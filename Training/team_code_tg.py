

################################################################################
#
# Training loop for ECG Chagas Classification using a Deep Neural Network
#
################################################################################

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from model_CNN import ECGTransformer, ConvBlock, ConvTrack
from Preprocess_training import ECGPreprocess
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
from helper_code import *
from helper_team_code import * 
from Evaluation import plot_metrics, plot_confusion_matrix, calculate_confusion_matrix, calculate_auroc


def train_model(data_folder, model_folder, verbose, batch_size=64, n_epochs=10, val_split=0.2):
    """
    Train the ECG classification model with validation monitoring and checkpointing.
    
    Parameters:
    -----------
    data_folder : str
        Folder containing the dataset
    model_folder : str
        Folder to save the trained model
    verbose : bool
        Whether to print progress messages
    batch_size : int
        Mini-batch size for training
    n_epochs : int
        Number of training epochs
    val_split : float
        Fraction of data to use for validation (0-1)
    """
    
    #####################################
    # Set Globals for Training Loop
    #####################################
    # Get device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")

    # Instantiate preprocessing class
    ecg_filter = ECGPreprocess(sampling_rate=400)

    # Create a folder for the model if it does not already exist
    os.makedirs(model_folder, exist_ok=True)
    
    # Create a folder for checkpoints
    checkpoint_folder = os.path.join(model_folder, 'checkpoints')
    os.makedirs(checkpoint_folder, exist_ok=True)
    
    # Create a folder for plots
    plots_folder = os.path.join(model_folder, 'plots')
    os.makedirs(plots_folder, exist_ok=True)

    ####################################
    # Load Data
    ####################################
    
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    random.seed(42)  # Fixed seed for reproducibility
    random.shuffle(records)
    num_records = len(records)
    
    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the labels from the data.
    if verbose:
        print('Extracting labels from the data...')

    labels = np.zeros(num_records, dtype=bool)

    # Iterate over the records to get labels
    for i in range(num_records):
        if verbose and i % 100 == 0:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])
        labels[i] = load_label(record)

    # Split data into training and validation sets
    val_size = int(num_records * val_split)
    train_records = records[val_size:]
    val_records = records[:val_size]
    train_labels = labels[val_size:]
    val_labels = labels[:val_size]
    
    if verbose:
        print(f'Training set size: {len(train_records)}')
        print(f'Validation set size: {len(val_records)}')

    # Create paths for records
    train_paths = [os.path.join(data_folder, rec) for rec in train_records]
    val_paths = [os.path.join(data_folder, rec) for rec in val_records]
    
    # Batch the training data
    train_batched_paths = [train_paths[i:i + batch_size] for i in range(0, len(train_paths), batch_size)]
    train_batched_labels = [train_labels[i:i + batch_size] for i in range(0, len(train_labels), batch_size)]
    train_batched_records = [train_records[i:i + batch_size] for i in range(0, len(train_records), batch_size)]
    
    # Batch the validation data
    val_batched_paths = [val_paths[i:i + batch_size] for i in range(0, len(val_paths), batch_size)]
    val_batched_labels = [val_labels[i:i + batch_size] for i in range(0, len(val_labels), batch_size)]
    val_batched_records = [val_records[i:i + batch_size] for i in range(0, len(val_records), batch_size)]
    
    #####################################
    # Initialize Model
    #####################################
    if verbose:
        print('Initializing model...')
    
    # Instantiate ECG model - with reduced parameters for efficiency
    model = ECGTransformer(
        d_model=64,             # Reduced from 128
        nhead=4,                # Reduced from 8
        num_encoder_layers=3,   # Reduced from 6
        dim_feedforward=256,    # Reduced from 512
        cnn_channels=32         # Reduced from 64
    ).to(device)

    # Training loop globals
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # For tracking metrics
    train_losses = []
    val_losses = []
    train_aurocs = []
    val_aurocs = []
    best_val_loss = float('inf')
    best_val_auroc = 0.0
    
    
    #####################################
    # Training Loop
    #####################################
    if verbose:
        print('Starting training...')
    
    for epoch in range(n_epochs):
        model.train()
        epoch_train_loss = 0.0
        batch_count = 0
        
        # Training phase
        for b_i in range(len(train_batched_paths)):
            if verbose and b_i % 5 == 0:
                print(f'Epoch {epoch+1}/{n_epochs}, Batch {b_i+1}/{len(train_batched_paths)}')
            
            batch = train_batched_paths[b_i]
            batch_records = train_batched_records[b_i]
            targets = torch.tensor(train_batched_labels[b_i], dtype=torch.float32).to(device)
            
            
            try:
                # Process demographic data
                sex_indices, age_groups = process_demographic_data(batch_records, data_folder, len(batch_records))
                sex_indices = sex_indices.to(device)
                age_groups = age_groups.to(device)
                
                processed_data = ecg_filter.process_wfdb_files(
                    batch, 
                    pad_to_length=4096,
                    apply_resample=False,
                    apply_highpass=True,  # Use filtering for better signal quality
                    apply_lowpass=True,
                    device=device
                )
            
                optimizer.zero_grad()
                outputs = model(ecg=processed_data, sex=sex_indices, age_group=age_groups)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_train_loss += loss.item()
                batch_count += 1
            
            except Exception as e:
                print(f"Error processing batch {b_i}: {e}")
                continue
        
        # Calculate average training loss for this epoch
        avg_train_loss = epoch_train_loss / max(1, batch_count)
        train_losses.append(avg_train_loss)
        
        # Validation phase after each epoch
        model.eval()
        epoch_val_loss = 0.0
        val_batch_count = 0
        
        with torch.no_grad():
            for b_i in range(len(val_batched_paths)):
                batch = val_batched_paths[b_i]
                batch_records = val_batched_records[b_i]
                targets = torch.tensor(val_batched_labels[b_i], dtype=torch.float32).to(device)
                
                try:
                    # Process demographic data
                    sex_indices, age_groups = process_demographic_data(batch_records, data_folder, len(batch_records))
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
                    loss = criterion(outputs, targets)
                    
                    epoch_val_loss += loss.item()
                    val_batch_count += 1
                
                except Exception as e:
                    print(f"Error processing validation batch {b_i}: {e}")
                    continue
        
        # Calculate average validation loss for this epoch
        avg_val_loss = epoch_val_loss / max(1, val_batch_count)
        val_losses.append(avg_val_loss)
        
        # Calculate AUROC for training and validation sets
        train_auroc = calculate_auroc(model, train_paths, train_labels, batch_size, ecg_filter, device, data_folder)
        val_auroc = calculate_auroc(model, val_paths, val_labels, batch_size, ecg_filter, device, data_folder)
        
        train_aurocs.append(train_auroc)
        val_aurocs.append(val_auroc)

        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        if verbose:
            print(f'Epoch {epoch+1}/{n_epochs}, '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, '
                  f'Train AUROC: {train_auroc:.4f}, '
                  f'Val AUROC: {val_auroc:.4f}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save model if it's the best so far (based on validation loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_auroc': train_auroc,
                'val_auroc': val_auroc,
            }, os.path.join(model_folder, 'best_model_by_loss.pt'))
            
            if verbose:
                print(f'New best model saved with validation loss: {avg_val_loss:.4f}')

        # Also save model if it has the best AUROC
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            
            # Generate confusion matrices for both training and validation sets
            train_tn, train_fp, train_fn, train_tp = calculate_confusion_matrix(
                model, train_paths, train_labels, batch_size, ecg_filter, device, data_folder
            )
            
            val_tn, val_fp, val_fn, val_tp = calculate_confusion_matrix(
                model, val_paths, val_labels, batch_size, ecg_filter, device, data_folder
            )
            
            # Plot the confusion matrices
            train_cm_folder = os.path.join(plots_folder, 'train_confusion_matrices')
            os.makedirs(train_cm_folder, exist_ok=True)

            train_cm_fig = plot_confusion_matrix(
                train_tn, train_fp, train_fn, train_tp, epoch, 
                os.path.join(plots_folder, 'train_confusion_matrices')
            )
            
            val_cm_folder = os.path.join(plots_folder, 'val_confusion_matrices')
            os.makedirs(val_cm_folder, exist_ok=True)

            val_cm_fig = plot_confusion_matrix(
                val_tn, val_fp, val_fn, val_tp, epoch, 
                os.path.join(plots_folder, 'val_confusion_matrices')
            )
            
            # Save confusion matrix values in the model checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_auroc': train_auroc,
                'val_auroc': val_auroc,
                'train_confusion_matrix': {
                    'tn': train_tn, 'fp': train_fp, 'fn': train_fn, 'tp': train_tp
                },
                'val_confusion_matrix': {
                    'tn': val_tn, 'fp': val_fp, 'fn': val_fn, 'tp': val_tp
                }
            }, os.path.join(model_folder, 'best_model_by_auroc.pt'))
            
            # Save as best_model.pt for loading
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_auroc': train_auroc,
                'val_auroc': val_auroc,
                'train_confusion_matrix': {
                    'tn': train_tn, 'fp': train_fp, 'fn': train_fn, 'tp': train_tp
                },
                'val_confusion_matrix': {
                    'tn': val_tn, 'fp': val_fp, 'fn': val_fn, 'tp': val_tp
                }
            }, os.path.join(model_folder, 'best_model.pt'))
            
            if verbose:
                print(f'New best model saved with validation AUROC: {val_auroc:.4f}')
                print(f'Confusion matrix summary:')
                print(f'  Training: TN={train_tn}, FP={train_fp}, FN={train_fn}, TP={train_tp}')
                print(f'  Validation: TN={val_tn}, FP={val_fp}, FN={val_fn}, TP={val_tp}')
        
        # Plot losses every 5 epochs
        if (epoch + 1) % 5 == 0:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, epoch + 2), train_losses, label='Training Loss')
            plt.plot(range(1, epoch + 2), val_losses, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Training and Validation Loss up to Epoch {epoch+1}')
            plt.legend()
            plt.grid(True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(os.path.join(plots_folder, f'loss_plot_epoch_{epoch+1}_{timestamp}.png'))
            plt.close()
            
            if verbose:
                print(f'Loss plot saved at epoch {epoch+1}')
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_folder, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_losses,
                'val_loss': val_losses,
                'train_auroc': train_aurocs,
                'val_auroc': val_aurocs,
            }, checkpoint_path)
            
            if verbose:
                print(f'Checkpoint saved at epoch {epoch+1}')
    
    # Save the final model
    save_model(model_folder, model)
    
    # Save the final metrics plots
    plot_metrics(train_losses, val_losses, train_aurocs, val_aurocs, n_epochs-1, plots_folder)

    if verbose:
        print('Training completed.')
        print(f'Best validation loss: {best_val_loss:.4f}')
        print(f'Best validation AUROC: {best_val_auroc:.4f}')
        print()

