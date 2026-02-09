import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, BitsAndBytesConfig
from typing import Literal, Dict, Any
from models.modules import TimeEncoder
from utils.utils import NeighborSampler


class TIDE(nn.Module):
    """
    TIDE: Temporal Interaction as Descriptive Events
    A unified model for temporal graph representation learning using Large Language Models.
    This model handles both Future Link Prediction (FLP) and Dynamic Node Classification (DNC) tasks.
    """
    def __init__(self,
                 node_raw_features: np.ndarray,
                 edge_raw_features: np.ndarray,
                 neighbor_sampler: Any, 
                 time_feat_dim: int,
                 num_neighbors: int,
                 llm_ckpt_dir: str,
                 output_dim: int, 
                 llm_hidden_dim: int = None, 
                 device: str = 'cpu',
                 llm_quantization_config: Dict[str, Any] = None,
                 use_flash_attention: bool = True):
        super().__init__()

        self.device = device
        self.num_neighbors = num_neighbors

        # 1. Feature and Sampler Initialization
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32))
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32))
        self.neighbor_sampler = neighbor_sampler

        node_feat_dim = self.node_raw_features.shape[1]
        edge_feat_dim = self.edge_raw_features.shape[1]

        # 2. Encoders and Projectors
        self.time_encoder = TimeEncoder(time_dim=time_feat_dim).to(device)
        
        # This projector translates our custom STE embeddings into the LLM's latent space
        # STE: Semantic-Temporal Event (v_node_feat + u_edge_feat + time_feat)
        ste_raw_dim = node_feat_dim + edge_feat_dim + time_feat_dim
        # The embedding for the central node `u` in the sequence
        u_token_raw_dim = node_feat_dim 

        # 3. Large Language Model Backbone
        self.llm, self.config = self._load_llm(llm_ckpt_dir, llm_quantization_config, use_flash_attention, device)
        
        if llm_hidden_dim is None:
            llm_hidden_dim = self.config.hidden_size

        self.projector_ste = nn.Linear(ste_raw_dim, llm_hidden_dim).to(device)
        self.projector_u = nn.Linear(u_token_raw_dim, llm_hidden_dim).to(device)

        # 4. Special Tokens for Task Prompting
        # 0: DNC_TASK, 1: FLP_TASK
        # self.task_tokens = nn.Embedding(2, llm_hidden_dim).to(device)
        self.task_tokens = nn.Embedding(1, llm_hidden_dim).to(device)
        # Separator token for FLP task
        self.sep_token = nn.Embedding(1, llm_hidden_dim).to(device)
        
        self.output_head = nn.Linear(llm_hidden_dim, output_dim).to(device, dtype=self.llm.dtype)
        
        # Set LLM to evaluation mode and freeze its parameters
        self.llm.eval()
        for param in self.llm.parameters():
            param.requires_grad = False

    def _load_llm(self, ckpt_dir, quantization_config, use_flash_attention, device):
        """Helper function to load LLM with optional quantization and flash attention."""
        config_args = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16, 
        }
        if quantization_config:
            config_args["quantization_config"] = BitsAndBytesConfig(**quantization_config)
        if use_flash_attention:
            config_args["attn_implementation"] = "flash_attention_2"
        
        llm = AutoModel.from_pretrained(ckpt_dir, **config_args, device_map={'': device})
        return llm, llm.config

    def _prepare_narrative_sequence(self, node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        Prepares the interleaved narrative sequence [v1, u1, v2, u2, ...] for a batch of nodes.
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        """
        batch_size = node_ids.shape[0]

        # 1. Sample historical neighbors
        # neighbor_node_ids, ndarray, shape (batch_size, num_neighbors)
        # neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors)
        # neighbor_times, ndarray, shape (batch_size, num_neighbors)
        neighbor_node_ids, neighbor_edge_ids, neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(
                node_ids=node_ids,
                node_interact_times=node_interact_times,
                num_neighbors=self.num_neighbors
            )

        # 2. Fetch raw features for nodes and edges
        # Neighbor features (become v tokens)
        # Tensor, shape (batch_size, num_neighbors, node_feat_dim)
        neighbor_node_feats = self.node_raw_features[torch.from_numpy(neighbor_node_ids)].to(self.device)
        # Tensor, shape (batch_size, num_neighbors, edge_feat_dim)
        neighbor_edge_feats = self.edge_raw_features[torch.from_numpy(neighbor_edge_ids)].to(self.device)
        # Central node features (become u tokens)
        # Tensor, shape (batch_size, node_feat_dim)
        central_node_feats = self.node_raw_features[torch.from_numpy(node_ids)].to(self.device)

        # 3. Compute time features (recency)
        # Tensor, shape (batch_size, num_neighbors)
        time_diffs = torch.from_numpy(node_interact_times[:, np.newaxis] - neighbor_times).float().to(self.device)
        # Tensor, shape (batch_size, num_neighbors, time_feat_dim)
        time_feats = self.time_encoder(time_diffs)
        # Handle padding
        time_feats[torch.from_numpy(neighbor_node_ids == 0).to(self.device)] = 0.0

        # 4. Create STE (Semantic-Temporal Event) embeddings for v tokens
        # Tensor, shape (batch_size, num_neighbors, node_feat_dim + edge_feat_dim + time_feat_dim)
        ste_cat_feats = torch.cat([neighbor_node_feats, neighbor_edge_feats, time_feats], dim=-1)
        # Tensor, shape (batch_size, num_neighbors, llm_hidden_dim)
        v_token_embeds = self.projector_ste(ste_cat_feats)

        # 5. Create embeddings for u tokens
        # We tile the central node features to interleave with v tokens
        # Tensor, shape (batch_size, num_neighbors, node_feat_dim)
        u_token_feats_tiled = central_node_feats.unsqueeze(1).repeat(1, self.num_neighbors, 1)
        # Tensor, shape (batch_size, num_neighbors, llm_hidden_dim)
        u_token_embeds = self.projector_u(u_token_feats_tiled)

        # 6. Interleave to form the narrative: [v1, u1, v2, u2, ...]
        # Reshape to (B, N, 2, D) and then to (B, N*2, D)
        # Tensor, shape (batch_size, num_neighbors * 2, llm_hidden_dim)
        narrative_embeds = torch.stack([v_token_embeds, u_token_embeds], dim=2).reshape(
            batch_size, self.num_neighbors * 2, -1
        )
        
        return narrative_embeds
    
    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()

    def forward(self,
                src_node_ids: np.ndarray, 
                node_interact_times: np.ndarray,
                dst_node_ids: np.ndarray = None):
        """
        :param src_node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :param dst_node_ids: ndarray, shape (batch_size, ), node ids
        """
        # 1. Prepare Narrative Embeddings
        # Tensor, shape (batch_size, num_neighbors * 2, llm_hidden_dim)
        src_narrative = self._prepare_narrative_sequence(src_node_ids, node_interact_times)
        
        # 2. Construct Task-Specific Prompt
        batch_size = src_node_ids.shape[0]
        
        # Prompt: [FLP_TASK] src_narrative [SEP] dst_narrative
        assert dst_node_ids is not None, "dst_node_ids must be provided for FLP task"
        # Tensor, shape (batch_size, num_neighbors * 2, llm_hidden_dim)
        dst_narrative = self._prepare_narrative_sequence(dst_node_ids, node_interact_times)
        # Tensor, shape (batch_size, 1, llm_hidden_dim)
        task_embed = self.task_tokens(torch.tensor([0], device=self.device)).unsqueeze(0).repeat(batch_size, 1, 1)
        # Tensor, shape (batch_size, 1, llm_hidden_dim)
        sep_embed = self.sep_token(torch.tensor([0], device=self.device)).unsqueeze(0).repeat(batch_size, 1, 1)
        # Tensor, shape (batch_size, num_neighbors * 2 + 2 + num_neighbors * 2, llm_hidden_dim)
        inputs_embeds = torch.cat([src_narrative, sep_embed, dst_narrative, task_embed], dim=1)
            
        inputs_embeds = inputs_embeds.to(self.llm.dtype)

        # 3. LLM Inference
        outputs = self.llm(inputs_embeds=inputs_embeds)
        # We take the representation of the first token, which is our [TASK] token.
        # This vector serves as the final, context-rich representation for the task.
        # Tensor, shape (batch_size, llm_hidden_dim)
        task_representation = outputs.last_hidden_state[:, -1, :]

        logits = self.output_head(task_representation)
            
        # Tensor, shape (batch_size, output_dim)
        return logits