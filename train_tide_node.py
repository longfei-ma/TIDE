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
from utils.metrics import get_node_classification_metrics
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
        
        model.projector_ste.load_state_dict(checkpoint['projector_ste'])
        model.projector_u.load_state_dict(checkpoint['projector_u'])
        model.task_tokens.load_state_dict(checkpoint['task_tokens'])
        model.sep_token.load_state_dict(checkpoint['sep_token'])
        model.output_head.load_state_dict(checkpoint['output_head'])

        print("model weights load over!")

            
def touch(file_path):    
    with open(file_path, 'w'):  
        pass  

def get_node_classification_args():
    """
    get the args for the node classification task
    :return:
    """
    # arguments
    parser = argparse.ArgumentParser('Interface for the node classification task')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='FOOD',
                        choices=['FOOD', 'IMDB', 'Librarything', 'Beeradvocate', 'Amazon-Kindle', 'Ratebeer'])
    parser.add_argument('--llm_embedding', type=str, help='large language model to be used for embedding', default='sbert')
    parser.add_argument('--llm_predictor', type=str, help='large language model to be used for predictor', default='Llama-3.2-1B')
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--model_name', type=str, default='DyGFormer', help='name of the model',
                        choices=['JODIE', 'DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer', 'FreeDyG'])
    parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use')
    parser.add_argument('--num_neighbors', type=int, default=2, help='number of neighbors to sample for each node')
    parser.add_argument('--sample_neighbor_strategy', type=str, default='recent', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--time_feat_dim', type=int, default=100, help='dimension of the time embedding')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
    parser.add_argument('--train_ratio', type=float, default=0.4, help='ratio of training set')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='ratio of validation set')
    parser.add_argument('--num_runs', type=int, default=5, help='number of runs')
    parser.add_argument('--run', type=int, default=0, help='number of run to run')
    parser.add_argument('--empty', action='store_true', default=False, help='whether to remove texts')
    parser.add_argument('--empty_ndim', type=int, default=384, help='ndim of empty')

    try:
        args = parser.parse_args()
        args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    except:
        parser.print_help()
        sys.exit()

    return args


def get_node_classification_data(dataset_name: str, train_ratio: float, val_ratio: float, llm_embedding: str, llm_predictor: str,args=None):
    """
    generate data for node classification task
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, (Data object)
    """
    # Load data and train val test split
    graph_df = pd.read_csv(f'./processed_data/{dataset_name}/ml_{dataset_name}.csv') 
    edge_raw_features = np.load(f'./processed_data/{dataset_name}/{llm_embedding}_{dataset_name}.npy')
    node_raw_features = np.load(f'./processed_data/{dataset_name}/{llm_embedding}_{dataset_name}_node.npy')

    with open(f'DG_data/{dataset_name}/{dataset_name}_unique_labels.json', 'r', encoding="utf-8") as f: 
        unique_labels = json.load(f)

    with open(f'DG_data/{dataset_name}/{dataset_name}_labels.json', 'r', encoding="utf-8") as f: 
        labels = json.load(f)
    if dataset_name in ['FOOD', 'IMDB']: # multi-label, 
        fixed_labels = np.zeros((len(labels), len(unique_labels)), dtype=int)
        for i, label in enumerate(labels):
            fixed_labels[i, label] = 1
        raw_labels = labels
        labels = fixed_labels
    else:
        raw_labels = labels
        labels = np.array(labels)

    
    # get the timestamp of validate and test set
    val_time, test_time = list(np.quantile(graph_df.ts, [train_ratio, (train_ratio+val_ratio)]))

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)

    # The setting of seed follows previous works
    random.seed(2020)

    train_mask = node_interact_times <= val_time
    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels)
    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask],
                      edge_ids=edge_ids[train_mask], labels=labels[train_mask])
    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels[val_mask])
    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], labels=labels[test_mask])

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, unique_labels, raw_labels


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


def evaluate_model_node_classification(model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader, evaluate_data: Data, loss_func: nn.Module):
    model.set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses, trues and predicts
        evaluate_total_loss, evaluate_y_trues, evaluate_y_predicts = 0.0, [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices], evaluate_data.labels[evaluate_data_indices]
            if batch_labels.ndim<2: 
                valid_mask = batch_labels>=0 
                batch_labels = batch_labels[valid_mask]
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = batch_src_node_ids[valid_mask], batch_dst_node_ids[valid_mask], batch_node_interact_times[valid_mask], batch_edge_ids[valid_mask]
            
            # get predicted probabilities, shape (batch_size, c)
            predicts = model(
                   src_node_ids=batch_src_node_ids, 
                   dst_node_ids=batch_dst_node_ids, 
                   node_interact_times=batch_node_interact_times)

            labels = torch.from_numpy(batch_labels).to(predicts.device)
            if labels.ndim == predicts.ndim: # multi-label
                labels = labels.float()

            loss = loss_func(input=predicts, target=labels)

            evaluate_total_loss += loss.item()

            evaluate_y_trues.append(labels)
            evaluate_y_predicts.append(predicts)

            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

        evaluate_total_loss /= (batch_idx + 1)
        evaluate_y_trues = torch.cat(evaluate_y_trues, dim=0)
        evaluate_y_predicts = torch.cat(evaluate_y_predicts, dim=0)

        evaluate_metrics = get_node_classification_metrics(predicts=evaluate_y_predicts, labels=evaluate_y_trues)

    return evaluate_total_loss, evaluate_metrics


if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_node_classification_args()

    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, unique_labels,raw_labels = \
        get_node_classification_data(dataset_name=args.dataset_name, train_ratio=args.train_ratio, val_ratio=args.val_ratio, llm_embedding=args.llm_embedding, llm_predictor=args.llm_predictor, args=args)

    c = len(unique_labels)
    sample_labels = np.concatenate(np.array(raw_labels,dtype=object)) if args.dataset_name in ['FOOD', 'IMDB'] else train_data.labels #
    class_sample_count = torch.bincount(torch.as_tensor(sample_labels[sample_labels>=0], dtype=int))
    weights = 1.0 / class_sample_count.float()
    weights = weights.to(args.device) 
    
    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # get data loaders
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    if args.dataset_name in ['FOOD', 'IMDB']: #multi-label
        loss_func = nn.BCEWithLogitsLoss()
    else: 
        loss_func = nn.CrossEntropyLoss()

    val_metric_all_runs, test_metric_all_runs = [], []
    
    run = args.run
    set_random_seed(seed=run)
    args.seed = run
    save_model_name = f'seed{args.seed}'
    
    args.save_model_name = save_model_name

    run_start_time = time.time()
    print(f"********** Run {run + 1} Training starts. **********")
    print(f'configuration is {args}')

    ckpt = f'ckpt/{args.llm_predictor}'

    # 1. Instantiate the single, unified TIDE model
    # For FLP, dnc_output_dim can be a placeholder
    if "Qwen3-30B-A3B" in args.llm_predictor or "deepseek-moe" in args.llm_predictor or "8B" in args.llm_predictor:
        quant_config = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16}
    else:
        quant_config = None

    model = TIDE(
        node_raw_features=node_raw_features,
        edge_raw_features=edge_raw_features,
        neighbor_sampler=full_neighbor_sampler,
        time_feat_dim=args.time_feat_dim,
        num_neighbors=args.num_neighbors,
        llm_ckpt_dir=ckpt,
        output_dim=c, 
        device=args.device,
        llm_quantization_config=quant_config
    )

    print(f'model -> {model}')
    print(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
        f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')
    
    save_model_folder = f"saved_models{args.train_ratio}-n{args.num_neighbors}_{args.llm_name}/{args.dataset_name}/{args.model_name}/{args.save_model_name}"

    save_model_path = os.path.join(save_model_folder, f"{save_model_name}.pkl")
    early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
        save_model_name=args.save_model_name)
    
    DONE_file = os.path.join(save_model_folder, "DONE_")

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
            model.set_neighbor_sampler(full_neighbor_sampler)

            # store train losses, trues and predicts
            train_total_loss, train_y_trues, train_y_predicts = 0.0, [], []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels = \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], train_data.node_interact_times[train_data_indices], \
                    train_data.edge_ids[train_data_indices], train_data.labels[train_data_indices]
                if batch_labels.ndim<2: 
                    valid_mask = batch_labels>=0 
                    batch_labels = batch_labels[valid_mask]
                    batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = batch_src_node_ids[valid_mask], batch_dst_node_ids[valid_mask], batch_node_interact_times[valid_mask], batch_edge_ids[valid_mask]
                
                # get predicted probabilities, shape (batch_size, c)
                predicts = model(
                   src_node_ids=batch_src_node_ids, 
                   dst_node_ids=batch_dst_node_ids, 
                   node_interact_times=batch_node_interact_times)
                labels = torch.from_numpy(batch_labels).to(predicts.device)
                if labels.ndim == predicts.ndim: # multi-label
                    labels = labels.float()

                loss = loss_func(input=predicts, target=labels)

                train_total_loss += loss.item()

                train_y_trues.append(labels)
                train_y_predicts.append(predicts)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')

            train_total_loss /= (batch_idx + 1)
            train_y_trues = torch.cat(train_y_trues, dim=0)
            train_y_predicts = torch.cat(train_y_predicts, dim=0)

            train_metrics = get_node_classification_metrics(predicts=train_y_predicts, labels=train_y_trues)

            val_total_loss, val_metrics = evaluate_model_node_classification(model=model,
                    neighbor_sampler=full_neighbor_sampler,
                    evaluate_idx_data_loader=val_idx_data_loader,
                    evaluate_data=val_data,
                    loss_func=loss_func)

            print(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {train_total_loss:.4f}')
            for metric_name in train_metrics.keys():
                print(f'train {metric_name}, {train_metrics[metric_name]:.4f}')
            print(f'validate loss: {val_total_loss:.4f}')
            for metric_name in val_metrics.keys():
                print(f'validate {metric_name}, {val_metrics[metric_name]:.4f}')

            # select the best model based on all the validate metrics
            val_metric_indicator = []
            for metric_name in val_metrics.keys():
                val_metric_indicator.append((metric_name, val_metrics[metric_name], True))

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
    test_total_loss, test_metrics = evaluate_model_node_classification(model=model,
                    neighbor_sampler=full_neighbor_sampler,
                    evaluate_idx_data_loader=test_idx_data_loader,
                    evaluate_data=test_data,
                    loss_func=loss_func)
    

    # store the evaluation metrics at the current run
    test_metric_list = []
    single_run_time = time.time() - run_start_time
    print(f'Run {run + 1} Testing cost {single_run_time:.2f} seconds.')

    if args.dataset_name in ['FOOD', 'IMDB']: #multi-label
        test_metric_all_runs.append([test_metrics['jacc'], test_metrics['micro_f1'], test_metrics['macro_f1']])
    else:
        test_metric_all_runs.append([test_metrics['acc'], test_metrics['f1']])

    # save model result
    save_result_folder = f"./saved_results_tide"
    os.makedirs(save_result_folder, exist_ok=True)

    test_metric_all_runs = np.array(test_metric_all_runs)*100

    filename = f'{save_result_folder}/{args.dataset_name}-node.csv'
    print(f"Saving results to {filename}")  
    model_name = f'TIDE,{args.llm_predictor},{run}'
    with open(f"{filename}", 'a+') as write_obj:
        if args.dataset_name in ['FOOD', 'IMDB']: #multi-label
            if len(test_metric_all_runs)>1:
                write_obj.write(f"{model_name}," + 
                                f"{test_metric_all_runs[:, 0].mean():.2f} ± {test_metric_all_runs[:, 0].std():.2f}," +
                                f"{test_metric_all_runs[:, 1].mean():.2f} ± {test_metric_all_runs[:, 1].std():.2f}," +
                                f"{test_metric_all_runs[:, 2].mean():.2f} ± {test_metric_all_runs[:, 2].std():.2f}\n")
            else:
                write_obj.write(f"{model_name}," + 
                                f"{test_metric_all_runs[:, 0].mean():.2f}," +
                                f"{test_metric_all_runs[:, 1].mean():.2f}," +
                                f"{test_metric_all_runs[:, 2].mean():.2f}\n")
        else:
            if len(test_metric_all_runs)>1:
                write_obj.write(f"{model_name}," + 
                                f"{test_metric_all_runs[:, 0].mean():.2f} ± {test_metric_all_runs[:, 0].std():.2f}," +
                                f"{test_metric_all_runs[:, 1].mean():.2f} ± {test_metric_all_runs[:, 1].std():.2f}\n") 
            else:
                write_obj.write(f"{model_name}," + 
                                f"{test_metric_all_runs[:, 0].mean():.2f}," +
                                f"{test_metric_all_runs[:, 1].mean():.2f}\n")