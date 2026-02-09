import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, jaccard_score, f1_score

def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().float().numpy()
    labels = labels.cpu().float().numpy()
    y_pred = (predicts >= 0.5).astype(int)

    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'average_precision': average_precision, 'roc_auc': roc_auc}


def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, n_classes)
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    y_pred_logit = predicts.cpu().detach().float()
    y_true = labels.cpu().numpy()
    if labels.shape != predicts.shape: # multi-class
        y_pred = y_pred_logit.argmax(axis=1).numpy()
        acc = accuracy_score(y_true, y_pred)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')

        return {
            'acc': acc,
            'f1': weighted_f1
        }
    
    else: #multi-label
        y_pred_probs = torch.sigmoid(y_pred_logit)
        y_pred = (y_pred_probs >= 0.5).int().numpy()

        jacc = jaccard_score(y_true, y_pred, average='samples')
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        macro_f1 = f1_score(y_true, y_pred, average='macro')

        return {
            'jacc': jacc,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
        }