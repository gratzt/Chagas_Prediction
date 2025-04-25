import os
import sys
from datetime import datetime
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from helper_team_code import *
from Preprocess_training import * 

#########################################################################################
# Create Evaluation Functions to be used on training and validations sets
#########################################################################################

def calculate_auroc(model, data_paths, labels, batch_size, ecg_filter, device, data_folder):
    """
    Calculate AUROC for a dataset
    
    Parameters:
    -----------
    model : torch.nn.Module
        The trained model
    data_paths : list
        List of paths to data files
    labels : list or numpy.ndarray
        True labels
    batch_size : int
        Batch size for processing
    ecg_filter : ECGPreprocess
        Preprocessing object
    device : torch.device
        Device to run calculations on
    data_folder : str
        Path to data folder
        
    Returns:
    --------
    float
        AUROC score
    """
    model.eval()
    all_probs = []
    all_batch_labels = []
    
    # Create batches
    batched_paths = [data_paths[i:i + batch_size] for i in range(0, len(data_paths), batch_size)]
    batched_labels = [labels[i:i + batch_size] for i in range(0, len(labels), batch_size)]
    
    with torch.no_grad():
        for b_i in range(len(batched_paths)):
            batch = batched_paths[b_i]
            batch_labels = batched_labels[b_i]
            
            # Extract record names from paths
            batch_records = [os.path.relpath(path, data_folder) for path in batch]
            
            try:
                # Process demographic data
                sex_indices, age_groups = process_demographic_data(batch_records, data_folder, batch_size)
                sex_indices = sex_indices.to(device)
                age_groups = age_groups.to(device)
                
                processed_data = ecg_filter.process_wfdb_files(
                    batch, 
                    pad_to_length=4096,
                    apply_resample=False,
                    apply_highpass=True,
                    apply_lowpass=True,
                    device=device
                )
            
                outputs = model(ecg=processed_data, sex=sex_indices, age_group=age_groups)
                probabilities = torch.sigmoid(outputs).cpu().numpy()
                
                all_probs.extend(probabilities)
                all_batch_labels.extend(batch_labels)
            
            except Exception as e:
                print(f"Error processing batch {b_i} for AUROC calculation: {e}")
                continue
    
    if len(all_probs) == 0:
        return 0.0
    
    # Calculate AUROC
    try:
        auroc = roc_auc_score(all_batch_labels, all_probs)
        return auroc
    except ValueError as e:
        print(f"Error calculating AUROC: {e}")
        return 0.0

def plot_metrics(train_losses, val_losses, train_aurocs, val_aurocs, epoch, plots_folder):
    """
    Plot training and validation metrics
    
    Parameters:
    -----------
    train_losses : list
        Training losses
    val_losses : list
        Validation losses
    train_aurocs : list
        Training AUROC scores
    val_aurocs : list
        Validation AUROC scores
    epoch : int
        Current epoch
    plots_folder : str
        Folder to save plots
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epoch + 2), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, epoch + 2), val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot AUROC
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epoch + 2), train_aurocs, 'b-', label='Training AUROC')
    plt.plot(range(1, epoch + 2), val_aurocs, 'r-', label='Validation AUROC')
    plt.xlabel('Epochs')
    plt.ylabel('AUROC')
    plt.title('Training and Validation AUROC')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.05)  # AUROC is between 0 and 1
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, f'metrics_epoch_{epoch+1}_{timestamp}.png'))
    plt.close()

# Add this function to calculate confusion matrix
def calculate_confusion_matrix(model, data_paths, labels, batch_size, ecg_filter, device, data_folder):
    """
    Calculate confusion matrix for a dataset
    
    Parameters:
    -----------
    model : torch.nn.Module
        The trained model
    data_paths : list
        List of paths to data files
    labels : list or numpy.ndarray
        True labels
    batch_size : int
        Batch size for processing
    ecg_filter : ECGPreprocess
        Preprocessing object
    device : torch.device
        Device to run calculations on
    data_folder : str
        Path to data folder
        
    Returns:
    --------
    tuple
        (tn, fp, fn, tp) - elements of confusion matrix
    """
    model.eval()
    all_preds = []
    all_batch_labels = []
    
    # Create batches
    batched_paths = [data_paths[i:i + batch_size] for i in range(0, len(data_paths), batch_size)]
    batched_labels = [labels[i:i + batch_size] for i in range(0, len(labels), batch_size)]
    
    with torch.no_grad():
        for b_i in range(len(batched_paths)):
            batch = batched_paths[b_i]
            batch_labels = batched_labels[b_i]
            
            # Extract record names from paths
            batch_records = [os.path.relpath(path, data_folder) for path in batch]
            
            try:
                # Process demographic data
                sex_indices, age_groups = process_demographic_data(batch_records, data_folder, batch_size)
                sex_indices = sex_indices.to(device)
                age_groups = age_groups.to(device)
                
                processed_data = ecg_filter.process_wfdb_files(
                    batch, 
                    pad_to_length=4096,
                    apply_resample=False,
                    apply_highpass=True,
                    apply_lowpass=True,
                    device=device
                )
            
                outputs = model(ecg=processed_data, sex=sex_indices, age_group=age_groups)
                probabilities = torch.sigmoid(outputs).cpu().numpy()
                predictions = (probabilities > 0.5).astype(int)
                
                all_preds.extend(predictions)
                all_batch_labels.extend(batch_labels)
            
            except Exception as e:
                print(f"Error processing batch {b_i} for confusion matrix calculation: {e}")
                continue
    
    if len(all_preds) == 0:
        return 0, 0, 0, 0
    
    # Calculate confusion matrix elements
    all_preds = np.array(all_preds)
    all_batch_labels = np.array(all_batch_labels)
    
    # Calculate TP, TN, FP, FN
    tp = np.sum((all_preds == 1) & (all_batch_labels == 1))
    tn = np.sum((all_preds == 0) & (all_batch_labels == 0))
    fp = np.sum((all_preds == 1) & (all_batch_labels == 0))
    fn = np.sum((all_preds == 0) & (all_batch_labels == 1))
    
    return tn, fp, fn, tp

def plot_confusion_matrix(tn, fp, fn, tp, epoch, plots_folder):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    tn : int
        True negatives
    fp : int
        False positives
    fn : int
        False negatives
    tp : int
        True positives
    epoch : int
        Current epoch
    plots_folder : str
        Folder to save plots
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    cm = np.array([[tn, fp], [fn, tp]])
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Create figure and axis
    fig, ax = plt.figure(figsize=(8, 8)), plt.gca()
    
    # Plot confusion matrix as heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    classes = ['Negative', 'Positive']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    # Add metrics as text
    plt.figtext(0.5, 0.01, f'Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}', 
                ha='center', fontsize=10)
    
    fig.tight_layout()
    plt.savefig(os.path.join(plots_folder, f'confusion_matrix_epoch_{epoch+1}_{timestamp}.png'))
    plt.close()
    
    return fig