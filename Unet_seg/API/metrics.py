import numpy as np
from skimage.metrics import structural_similarity as cal_ssim
import torch

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

def calculate_iou(predicted, target):
    # Flatten the batch dimension
    predicted_flat = predicted.view(predicted.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    # Filter out batches where all target elements are zero
    non_empty_mask = target_flat.any(dim=1)
    non_empty_predicted = predicted_flat[non_empty_mask]
    non_empty_target = target_flat[non_empty_mask]
    if non_empty_target.numel() == 0:
        return None
    intersection = torch.logical_and(non_empty_predicted, non_empty_target).sum(dim=1)
    union = torch.logical_or(non_empty_predicted, non_empty_target).sum(dim=1)
    
    iou = intersection.float() / union.float()
    
    # Check if all IoU values are NaN, if so, set mean IoU to 0
    if torch.isnan(iou).all():
        return 0.0
    
    # Calculate mean IoU excluding NaN values
    valid_iou = iou[~torch.isnan(iou)]
    mean_iou = valid_iou.mean().item() if valid_iou.numel() > 0 else 0.0
    
    return mean_iou




def metric_classification(pred, true):
    pred[pred<0.5] = 0
    pred[pred>=0.5] = 1
    # conf_matrix = confusion_matrix(y_true=true, y_pred=pred)
    # acc = accuracy_score(true, pred)
    # prec = precision_score(true, pred)
    # recall = recall_score(true, pred)
    # f1s = f1_score(true, pred)
    iou = jaccard_score(true, pred)
    return iou

def getConfusionMatrix(pred, true):
    pred[pred<0.5] = 0
    pred[pred>=0.5] = 1
    cf_matrix = confusion_matrix(y_true=true, y_pred=pred)
    return sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues').get_figure()