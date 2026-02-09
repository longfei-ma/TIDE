import argparse
import time
import random
import json
import sys
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
import shutil
import torch
import torch.nn as nn
import itertools

from utils.utils import set_random_seed, get_parameter_sizes
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler, NeighborSampler
from utils.metrics import get_link_prediction_metrics
from utils.AblationDataLoader import get_idx_data_loader
from utils.DataLoader import Data
from torch.utils.data import DataLoader
from models.TIDE import TIDE



os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class EarlyStopping(object):

    def __init__(self, patience: int, save_model_folder: str, save_model_name: str):
        """
        Early stop strategy.
        :param patience: int, max patience
        :param save_model_folder: str, save model folder
        :param save_model_name: str, save model name
        :param logger: Logger
        :param model_name: str, model name
        """
        self.patience = patience
        self.counter = 0
        self.best_metrics = {}
        self.early_stop = False
        self.train_non_parameter_flag = False
        self.save_model_path = os.path.join(save_model_folder, f"{save_model_name}.pkl")


    def step(self, metrics: list, weights_to_save: dict):
        metrics_compare_results = []
        for metric_tuple in metrics:
            metric_name, metric_value, higher_better = metric_tuple[0], metric_tuple[1], metric_tuple[2]

            if higher_better:
                if self.best_metrics.get(metric_name) is None or metric_value >= self.best_metrics.get(metric_name):
                    metrics_compare_results.append(True)
                else:
                    metrics_compare_results.append(False)
            else:
                if self.best_metrics.get(metric_name) is None or metric_value <= self.best_metrics.get(metric_name):
                    metrics_compare_results.append(True)
                else:
                    metrics_compare_results.append(False)
        # all the computed metrics are better than the best metrics
        if torch.all(torch.tensor(metrics_compare_results)):
            for metric_tuple in metrics:
                metric_name, metric_value = metric_tuple[0], metric_tuple[1]
                self.best_metrics[metric_name] = metric_value
            self.save_checkpoint(weights_to_save)
            self.counter = 0
            self.train_non_parameter_flag = True
        # metrics are not better at the epoch
        else:
            self.train_non_parameter_flag = False
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, weights_to_save: dict):
        """
        saves trained weights at self.save_model_path
        :param weights_to_save: dict of trained params
        :return:
        """
        print(f"save trained weights {self.save_model_path}")
        torch.save(weights_to_save, self.save_model_path)
        

    def load_checkpoint(self, model: nn.Module, map_location: str = None):
        print(f"load model from {self.save_model_path}")
        checkpoint = torch.load(self.save_model_path, map_location=map_location)
        # 分别加载各个模块的权重
        model.projector_ste.load_state_dict(checkpoint['projector_ste'])
        model.projector_u.load_state_dict(checkpoint['projector_u'])
        model.task_tokens.load_state_dict(checkpoint['task_tokens'])
        model.sep_token.load_state_dict(checkpoint['sep_token'])
        model.output_head.load_state_dict(checkpoint['output_head'])

        print("model的权重加载完毕。")

            
def touch(file_path): 
    with open(file_path, 'w'):  
        pass  

def get_link_prediction_args():
    """
    get the args for the link prediction task
    process                                                                                                                                  
    :return:
    """
    # arguments
    parser = argparse.ArgumentParser('Interface for the link prediction task')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='FOOD',
                        choices=['FOOD', 'IMDB', 'Librarything', 'Beeradvocate', 'Amazon-Kindle', 'Ratebeer'])
    parser.add_argument('--llm_embedding', type=str, help='large language model to be used for embedding', default='sbert')
    parser.add_argument('--llm_predictor', type=str, help='large language model to be used for predictor', default='Llama-3.2-1B')
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--model_name', type=str, default='TIDE', help='name of the model, note that EdgeBank is only applicable for evaluation',
                        choices=['JODIE', 'DyRep', 'TGAT', 'TGN', 'CAWN', 'EdgeBank', 'TCL', 'GraphMixer', 'DyGFormer', 'FreeDyG'])
    parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use')
    parser.add_argument('--num_neighbors', type=int, default=2, help='number of neighbors to sample for each node')
    parser.add_argument('--sample_neighbor_strategy', type=str, default='recent', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--num_walk_heads', type=int, default=8, help='number of heads used for the attention in walk encoder')
    parser.add_argument('--num_heads', type=int, default=2, help='number of heads used in attention layer')
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--walk_length', type=int, default=1, help='length of each random walk')
    parser.add_argument('--time_gap', type=int, default=2000, help='time gap for neighbors to compute node features')
    parser.add_argument('--time_feat_dim', type=int, default=100, help='dimension of the time embedding')
    parser.add_argument('--position_feat_dim', type=int, default=172, help='dimension of the position embedding')
    parser.add_argument('--edge_bank_memory_mode', type=str, default='unlimited_memory', help='how memory of EdgeBank works',
                        choices=['unlimited_memory', 'time_window_memory', 'repeat_threshold_memory'])
    parser.add_argument('--time_window_mode', type=str, default='fixed_proportion', help='how to select the time window size for time window memory',
                        choices=['fixed_proportion', 'repeat_interval'])
    parser.add_argument('--patch_size', type=int, default=1, help='patch size')
    parser.add_argument('--channel_embedding_dim', type=int, default=50, help='dimension of each channel embedding')
    parser.add_argument('--max_input_sequence_length', type=int, default=32, help='maximal length of the input sequence of each node')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
    parser.add_argument('--edges_truncate', type=int, default=10, help='number to do scalability')
    parser.add_argument('--train_ratio', type=float, default=0.4, help='ratio of training set')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='ratio of validation set')
    parser.add_argument('--num_runs', type=int, default=5, help='number of runs')
    parser.add_argument('--run', type=int, default=0, help='number of run to run')
    parser.add_argument('--test_interval_epochs', type=int, default=10, help='how many epochs to perform testing once')
    parser.add_argument('--negative_sample_strategy', type=str, default='random', choices=['random', 'historical', 'inductive'],
                        help='strategy for the negative edge sampling')
    parser.add_argument('--transductive', action='store_true', default=False, help='whether or not transductive')
    parser.add_argument('--load_best_configs', action='store_true', default=False, help='whether to load the best configurations')
    parser.add_argument('--walklm', action='store_true', default=False, help='whether to use walklm')
    parser.add_argument('--empty', action='store_true', default=False, help='whether to remove texts')
    parser.add_argument('--empty_ndim', type=int, default=64, help='ndim of empty')
    parser.add_argument('--empty_type', type=str, default='zero', help='empty type, zero, uniform or normal')

    try:
        args = parser.parse_args()
        args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    except:
        parser.print_help()
        sys.exit()

    return args


def get_link_prediction_data(dataset_name: str, train_ratio: float, val_ratio: float, llm_name: str, transductive=False, args=None):
    """
    generate data for link prediction task (inductive & transductive settings)
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, (Data object)
    """
    
    graph_df = pd.read_csv(f'./processed_data/{dataset_name}/ml_{dataset_name}.csv')
    edge_raw_features = np.load(f'./processed_data/{dataset_name}/{llm_name}_{dataset_name}.npy')
    node_raw_features = np.load(f'./processed_data/{dataset_name}/{llm_name}_{dataset_name}_node.npy')

    # get the timestamp of validate and test set
    val_time, test_time = list(np.quantile(graph_df.ts, [train_ratio, (train_ratio+val_ratio)]))

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels)

    # the setting of seed follows previous works
    random.seed(2020)

    # union to get node set
    node_set = set(src_node_ids) | set(dst_node_ids)
    num_total_unique_node_ids = len(node_set)

    # compute nodes which appear at test time
    test_node_set = set(src_node_ids[node_interact_times > val_time]).union(set(dst_node_ids[node_interact_times > val_time]))
    # sample nodes which we keep as new nodes (to test inductiveness), so then we have to remove all their edges from training
    new_test_node_set = set(random.sample(list(test_node_set), int(0.1 * num_total_unique_node_ids)))

    # mask for each source and destination to denote whether they are new test nodes
    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

    # mask, which is true for edges with both destination and source not being new test nodes (because we want to remove all edges involving any new test node)
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    # for train data, we keep edges happening before the validation time which do not involve any new node, used for inductiveness
    train_mask = np.logical_and(node_interact_times <= val_time, observed_edges_mask)

    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask],
                      edge_ids=edge_ids[train_mask], labels=labels[train_mask])

    # define the new nodes sets for testing inductiveness of the model
    train_node_set = set(train_data.src_node_ids).union(train_data.dst_node_ids)
    assert len(train_node_set & new_test_node_set) == 0
    # new nodes that are not in the training set
    new_node_set = node_set - train_node_set

    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    # new edges with new nodes in the val and test set (for inductive evaluation)
    edge_contains_new_node_mask = np.array([(src_node_id in new_node_set or dst_node_id in new_node_set)
                                            for src_node_id, dst_node_id in zip(src_node_ids, dst_node_ids)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    # validation and test data
    if transductive:
        val_mask = np.logical_and(~edge_contains_new_node_mask, val_mask)
        test_mask = np.logical_and(~edge_contains_new_node_mask, test_mask)

    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels[val_mask])

    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], labels=labels[test_mask])

    # validation and test with edges that at least has one new node (not in training set)
    new_node_val_data = Data(src_node_ids=src_node_ids[new_node_val_mask], dst_node_ids=dst_node_ids[new_node_val_mask],
                             node_interact_times=node_interact_times[new_node_val_mask],
                             edge_ids=edge_ids[new_node_val_mask], labels=labels[new_node_val_mask])

    new_node_test_data = Data(src_node_ids=src_node_ids[new_node_test_mask], dst_node_ids=dst_node_ids[new_node_test_mask],
                              node_interact_times=node_interact_times[new_node_test_mask],
                              edge_ids=edge_ids[new_node_test_mask], labels=labels[new_node_test_mask])
    
    print(f'transductive testing interactions: {test_data.num_interactions}\ninductive testing interactions: {new_node_test_data.num_interactions}')

    print("The dataset has {} interactions, involving {} different nodes".format(full_data.num_interactions, full_data.num_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.num_interactions, train_data.num_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.num_interactions, val_data.num_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
        test_data.num_interactions, test_data.num_unique_nodes))
    print("The new node validation dataset has {} interactions, involving {} different nodes".format(
        new_node_val_data.num_interactions, new_node_val_data.num_unique_nodes))
    print("The new node test dataset has {} interactions, involving {} different nodes".format(
        new_node_test_data.num_interactions, new_node_test_data.num_unique_nodes))
    print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(len(new_test_node_set)))

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data


def create_optimizer(trainable_params: nn.Module, optimizer_name: str, learning_rate: float, weight_decay: float = 0.0):
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(params=trainable_params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(params=trainable_params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(params=trainable_params, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Wrong value for optimizer {optimizer_name}!")

    return optimizer


def evaluate_model_link_prediction(model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
        evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data, loss_func: nn.Module):
    """
    evaluate models on the link prediction task
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :return:
    """
    # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()
    model.set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]

            if evaluate_neg_edge_sampler.negative_sample_strategy != 'random':
                batch_neg_src_node_ids, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids),
                    batch_src_node_ids=batch_src_node_ids,
                    batch_dst_node_ids=batch_dst_node_ids,
                    current_batch_start_time=batch_node_interact_times[0],
                    current_batch_end_time=batch_node_interact_times[-1])
            else:
                _, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

            pos_logits = model(
                   src_node_ids=batch_src_node_ids, 
                   dst_node_ids=batch_dst_node_ids, 
                   node_interact_times=batch_node_interact_times)

            # Negative samples
            neg_logits = model(
                            src_node_ids=batch_neg_src_node_ids,
                            dst_node_ids=batch_neg_dst_node_ids,
                            node_interact_times=batch_node_interact_times)
            
            # pos_logits.mean() and neg_logits.mean()

            predicts = torch.cat([pos_logits, neg_logits], dim=0)
            labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)

            loss = loss_func(input=predicts, target=labels)
            evaluate_losses.append(loss.item())
            evaluate_metrics.append(get_link_prediction_metrics(predicts=predicts.squeeze(dim=-1).sigmoid(), labels=labels))

            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

    return evaluate_losses, evaluate_metrics

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_link_prediction_args()

    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
        get_link_prediction_data(dataset_name=args.dataset_name, train_ratio=args.train_ratio, val_ratio=args.val_ratio, llm_name=args.llm_embedding, args=args)
    
    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    # in the inductive setting, negatives are sampled only amongst other new nodes
    # train negative edge sampler does not need to specify the seed, but evaluation samplers need to do so
    train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids)
    val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=0)
    if args.negative_sample_strategy != 'random':
        test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids,
                interact_times=full_data.node_interact_times, last_observed_time=val_data.node_interact_times[-1],
                negative_sample_strategy=args.negative_sample_strategy, seed=2)
        new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids,
                interact_times=new_node_test_data.node_interact_times, last_observed_time=val_data.node_interact_times[-1],
                negative_sample_strategy=args.negative_sample_strategy, seed=3)
    else:
        test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2)
        new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids, seed=3)

    # get data loaders
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor, seed=0)
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor, seed=1)
    
    run = args.run
    set_random_seed(seed=run)
    args.seed = run
    save_model_name = f'seed{args.seed}'
    
    args.save_model_name = save_model_name

    run_start_time = time.time()
    print(f"********** Run {run + 1} Training starts. **********")
    print(f'configuration is {args}')
    ckpt = f'ckpt/{args.llm_predictor}'

    if "Qwen3-30B-A3B" in args.llm_predictor or "deepseek-moe" in args.llm_predictor:
        quant_config = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16}
    elif "8B" in args.llm_predictor: # 
        quant_config = {"load_in_8bit": True,}
    else:
        quant_config = None

    model = TIDE(
        node_raw_features=node_raw_features,
        edge_raw_features=edge_raw_features,
        neighbor_sampler=train_neighbor_sampler,
        time_feat_dim=args.time_feat_dim,
        num_neighbors=args.num_neighbors,
        llm_ckpt_dir=ckpt,
        output_dim=1, 
        device=args.device,
        llm_quantization_config=quant_config
    )

    print(f'model -> {model}')
    print(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
        f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')
    
    save_model_folder = f"saved_tide_link/{args.dataset_name}/{args.llm_predictor}-{args.save_model_name}"

    save_model_path = os.path.join(save_model_folder, f"{save_model_name}.pkl")
    early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
        save_model_name=args.save_model_name)
    
    DONE_file = os.path.join(save_model_folder, "DONE_")
    loss_func = nn.BCEWithLogitsLoss()
    
    if not os.path.exists(DONE_file):
        print(f'######## train new {save_model_path} ########')
        trainable_params = [
            {'params': model.projector_ste.parameters()},
            {'params': model.projector_u.parameters()},
            {'params': model.task_tokens.parameters()},
            {'params': model.sep_token.parameters()},
            {'params': model.output_head.parameters()} 
        ]
        optimizer = create_optimizer(trainable_params=trainable_params, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        for epoch in range(args.num_epochs):
            model.train()
            model.set_neighbor_sampler(train_neighbor_sampler)
            # store train losses and metrics
            train_losses, train_metrics = [], []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                    train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]
                
                _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

                # Tensor, shape (batch_size, 1, output_dim)
                pos_logits = model(
                   src_node_ids=batch_src_node_ids, 
                   dst_node_ids=batch_dst_node_ids, 
                   node_interact_times=batch_node_interact_times)

                # Negative samples
                # Tensor, shape (batch_size, 1, output_dim)
                neg_logits = model(
                                src_node_ids=batch_neg_src_node_ids,
                                dst_node_ids=batch_neg_dst_node_ids,
                                node_interact_times=batch_node_interact_times)

                # Calculate loss
                predicts = torch.cat([pos_logits, neg_logits], dim=0)
                labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)

                loss = loss_func(input=predicts, target=labels)
                train_losses.append(loss.item())
                train_metrics.append(get_link_prediction_metrics(predicts=predicts.squeeze(dim=-1).sigmoid(), labels=labels))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')

            val_losses, val_metrics = evaluate_model_link_prediction(model=model,
                neighbor_sampler=full_neighbor_sampler,
                evaluate_idx_data_loader=val_idx_data_loader,
                evaluate_neg_edge_sampler=val_neg_edge_sampler,
                evaluate_data=val_data,
                loss_func=loss_func)


            print(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
            for metric_name in train_metrics[0].keys():
                print(f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
            print(f'validate loss: {np.mean(val_losses):.4f}')
            for metric_name in val_metrics[0].keys():
                print(f'validate {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}')

            # select the best model based on all the validate metrics
            val_metric_indicator = []
            for metric_name in val_metrics[0].keys():
                val_metric_indicator.append((metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), True))
            
            weights_to_save = {
                'projector_ste': model.projector_ste.state_dict(),
                'projector_u': model.projector_u.state_dict(),
                'task_tokens': model.task_tokens.state_dict(),
                'sep_token': model.sep_token.state_dict(),
                'output_head': model.output_head.state_dict()
            }

            early_stop = early_stopping.step(val_metric_indicator, weights_to_save)
        
            if early_stop:
                break

        touch(DONE_file)
        single_run_time = time.time() - run_start_time
        print(f'Run {run + 1} Trainging cost {single_run_time:.2f} seconds.')

    else:
        print(f'{save_model_path} already exists, skip the {run+1}-th Run...')


    run_start_time = time.time()
    print(f"********** Run {run + 1} Testing starts. **********")
    torch.cuda.empty_cache()
    test_metric_all_runs, new_node_test_metric_all_runs = [], []
    early_stopping.load_checkpoint(model=model, map_location='cpu')

    # evaluate the best model
    print(f'get final performance on dataset {args.dataset_name}...')

    test_losses, test_metrics = evaluate_model_link_prediction(model=model,
                neighbor_sampler=full_neighbor_sampler,
                evaluate_idx_data_loader=test_idx_data_loader,
                evaluate_neg_edge_sampler=test_neg_edge_sampler,
                evaluate_data=test_data,
                loss_func=loss_func)
    
    new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model=model,
                neighbor_sampler=full_neighbor_sampler,
                evaluate_idx_data_loader=new_node_test_idx_data_loader,
                evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                evaluate_data=new_node_test_data,
                loss_func=loss_func)

    # store the evaluation metrics at the current run
    test_metric_dict, new_node_test_metric_dict = {}, {}

    print(f'test loss: {np.mean(test_losses):.4f}')
    for metric_name in test_metrics[0].keys():
        average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
        print(f'test {metric_name}, {average_test_metric:.4f}')
        test_metric_dict[metric_name] = average_test_metric

    print(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
    for metric_name in new_node_test_metrics[0].keys():
        average_new_node_test_metric = np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics])
        print(f'new node test {metric_name}, {average_new_node_test_metric:.4f}')
        new_node_test_metric_dict[metric_name] = average_new_node_test_metric

    single_run_time = time.time() - run_start_time
    print(f'Run {run + 1} Testing cost {single_run_time:.2f} seconds.')

    test_metric_all_runs.append([test_metric_dict['average_precision'], test_metric_dict['roc_auc']])
    new_node_test_metric_all_runs.append([new_node_test_metric_dict['average_precision'], new_node_test_metric_dict['roc_auc']])

    # save model result
    save_result_folder = f"./saved_results_tide"
    os.makedirs(save_result_folder, exist_ok=True)

    test_metric_all_runs = np.array(test_metric_all_runs)*100

    filename = f'{save_result_folder}/{args.dataset_name}-transductive.csv'
    print(f"Saving results to {filename}") 
    model_name = f'TIDE,{args.llm_predictor},{run}'
    
    with open(f"{filename}", 'a+') as write_obj:
        if len(test_metric_all_runs) > 1:
            write_obj.write(f"{model_name}," + 
                            f"{test_metric_all_runs[:, 0].mean():.2f} ± {test_metric_all_runs[:, 0].std():.2f}," +
                            f"{test_metric_all_runs[:, 1].mean():.2f} ± {test_metric_all_runs[:, 1].std():.2f}\n")
        else:
            write_obj.write(f"{model_name}," + 
                            f"{test_metric_all_runs[:, 0].mean():.2f}," +
                            f"{test_metric_all_runs[:, 1].mean():.2f}\n")
    print(f"{args.dataset_name}-{model_name}-{args.llm_predictor}-transductive:\n" + 
        f"AP: {test_metric_all_runs[:, 0].mean():.2f} ± {test_metric_all_runs[:, 0].std():.2f}\n" +
        f"AUROC: {test_metric_all_runs[:, 1].mean():.2f} ± {test_metric_all_runs[:, 1].std():.2f}\n")

    if new_node_test_metric_all_runs is not None: 
        new_node_test_metric_all_runs = np.array(new_node_test_metric_all_runs)*100
        filename = f'{save_result_folder}/{args.dataset_name}-{args.model_name}-inductive.csv'
        filename = f'{save_result_folder}/{args.dataset_name}-inductive.csv'
        print(f"Saving results to {filename}")
        with open(f"{filename}", 'a+') as write_obj:
            if len(test_metric_all_runs) > 1:
                write_obj.write(f"{model_name}," + 
                            f"{new_node_test_metric_all_runs[:, 0].mean():.2f} ± {new_node_test_metric_all_runs[:, 0].std():.2f}," +
                            f"{new_node_test_metric_all_runs[:, 1].mean():.2f} ± {new_node_test_metric_all_runs[:, 1].std():.2f}\n")
            else:
                write_obj.write(f"{model_name}," + 
                            f"{new_node_test_metric_all_runs[:, 0].mean():.2f}," +
                            f"{new_node_test_metric_all_runs[:, 1].mean():.2f}\n")
        
        print(f"{args.dataset_name}-{model_name}-{args.llm_predictor}-inductive:\n" + 
            f"AP: {new_node_test_metric_all_runs[:, 0].mean():.2f} ± {new_node_test_metric_all_runs[:, 0].std():.2f}\n" +
            f"AUROC: {new_node_test_metric_all_runs[:, 1].mean():.2f} ± {new_node_test_metric_all_runs[:, 1].std():.2f}\n")
