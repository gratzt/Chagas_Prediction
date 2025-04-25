import os
import numpy as np
import torch
import joblib
from torch import nn
from Preprocess_training import *
from helper_code import *
from model_CNN import ECGTransformer  

def load_model(model_folder, verbose):
    """
    Load the trained model.
    
    Parameters:
    -----------
    model_folder : str
        Folder containing the model
    verbose : bool
        Whether to print progress messages
        
    Returns:
    --------
    dict
        Dictionary containing the loaded model
    """
    if verbose:
        print('Loading model...')
    
    model_path = os.path.join(model_folder, 'best_model.pt')
    
    # Check if PyTorch model exists
    if os.path.exists(model_path):
        if verbose:
            print(f'Loading PyTorch model from {model_path}')
        
        # Get device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model instance
        model = ECGTransformer(
            d_model=64,
            nhead=4,
            num_encoder_layers=3,
            dim_feedforward=256,
            cnn_channels=32
        ).to(device)
        
        # Load state dict
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set to evaluation mode
        
        # Return the model, device, and preprocessing object
        ecg_filter = ECGPreprocess(sampling_rate=400)
        return {'model': model, 'device': device, 'ecg_filter': ecg_filter}
    else:
        # Fall back to joblib if PyTorch model not found
        model_filename = os.path.join(model_folder, 'model.sav')
        if verbose:
            print(f'Loading joblib model from {model_filename}')
        model = joblib.load(model_filename)
        return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    """
    Run the trained model on a single record.
    
    Parameters:
    -----------
    record : str
        Path to the record
    model : dict
        Dictionary containing the model and other information
    verbose : bool
        Whether to print progress messages
        
    Returns:
    --------
    tuple
        Binary prediction and probability
    """
    # Check if we loaded a PyTorch or sklearn model
    if 'device' in model:
        pytorch_model = model['model']
        device = model['device']
        ecg_filter = model['ecg_filter']
        
        # Extract header to get demographics
        header = load_header(record)
        age = get_age(header)
        sex = get_sex(header)
        
        # Convert demographics to indices for embedding layers
        age_group = convert_age_to_group(age)
        sex_index = convert_sex_to_index(sex)
        
        # Convert to tensors and move to device
        sex_tensor = torch.tensor([sex_index], dtype=torch.long).to(device)
        age_tensor = torch.tensor([age_group], dtype=torch.long).to(device)
        
        # Process the ECG data
        processed_data = ecg_filter.process_wfdb_files(
            [record], 
            pad_to_length=4096,
            apply_resample=False,
            apply_highpass=True,
            apply_lowpass=True,
            device=device
        )
        
        # Make prediction
        with torch.no_grad():
            outputs = pytorch_model(ecg=processed_data, sex=sex_tensor, age_group=age_tensor)
            probability_output = torch.sigmoid(outputs).item()
            binary_output = probability_output > 0.5
        
        return binary_output, probability_output
    else:
        # Fall back to the old sklearn model
        model = model['model']
        
        # Extract the features
        features = extract_features(record)
        features = features.reshape(1, -1)
        
        # Get the model outputs
        binary_output = model.predict(features)[0]
        probability_output = model.predict_proba(features)[0][1]
        
        return binary_output, probability_output

# Extract your features.
def extract_features(record):
    header = load_header(record)
    age = get_age(header)
    sex = get_sex(header)

    one_hot_encoding_sex = np.zeros(3, dtype=bool)
    if sex == 'Female':
        one_hot_encoding_sex[0] = 1
    elif sex == 'Male':
        one_hot_encoding_sex[1] = 1
    else:
        one_hot_encoding_sex[2] = 1

    signal, fields = load_signals(record)

    # TO-DO: Update to compute per-lead features. Check lead order and update and use functions for reordering leads as needed.

    num_finite_samples = np.size(np.isfinite(signal))
    if num_finite_samples > 0:
        signal_mean = np.nanmean(signal)
    else:
        signal_mean = 0.0
    if num_finite_samples > 1:
        signal_std = np.nanstd(signal)
    else:
        signal_std = 0.0

    features = np.concatenate(([age], one_hot_encoding_sex, [signal_mean, signal_std]))

    return np.asarray(features, dtype=np.float32)

# Save your trained model.
def save_model(model_folder, model):
    """
    Save the trained model.
    
    Parameters:
    -----------
    model_folder : str
        Folder to save the model
    model : nn.Module or object
        PyTorch model or sklearn model to save
    """
    if isinstance(model, torch.nn.Module):
        # Save PyTorch model
        model_path = os.path.join(model_folder, 'model.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
        }, model_path)
        
        # Also save in joblib format for compatibility
        d = {'model': model}
        filename = os.path.join(model_folder, 'model.sav')
        joblib.dump(d, filename, protocol=0)
    else:
        # Save sklearn model
        d = {'model': model}
        filename = os.path.join(model_folder, 'model.sav')
        joblib.dump(d, filename, protocol=0)
        
def convert_age_to_group(age):
    """
    Convert numerical age to age group index for embedding layer.
    Groups: 0-9, 10-19, 20-29, ..., 90+
    
    Parameters:
    -----------
    age : float
        Numerical age
        
    Returns:
    --------
    int
        Age group index (0-9)
    """
    if np.isnan(age):
        return 9  # Default to oldest group if age is NaN
    
    age_group = min(int(age // 10), 9)  # Cap at group 9 (90+)
    return age_group

def convert_sex_to_index(sex):
    """
    Convert sex string to index for embedding layer.
    
    Parameters:
    -----------
    sex : str
        Sex string ('Male', 'Female', or None)
        
    Returns:
    --------
    int
        Sex index (0 for Female, 1 for Male)
    """
    if sex == 'Female':
        return 0
    elif sex == 'Male':
        return 1
    else:
        return 2  # Default to Female if unknown

def process_demographic_data(records, data_folder, batch_size):
    """
    Process demographic data for a batch of records.
    
    Parameters:
    -----------
    records : list
        List of record names
    data_folder : str
        Path to data folder
    batch_size : int
        Batch size
        
    Returns:
    --------
    tuple
        Tensors for sex and age_group
    """
    age_groups = []
    sex_indices = []
    
    for record in records:
        full_record_path = os.path.join(data_folder, record)
        header = load_header(full_record_path)
        
        # Extract age and sex
        age = get_age(header)
        sex = get_sex(header)
        
        # Convert to indices for embedding layers
        age_group = convert_age_to_group(age)
        sex_index = convert_sex_to_index(sex)
        
        age_groups.append(age_group)
        sex_indices.append(sex_index)
    
    # Convert to tensors
    age_groups_tensor = torch.tensor(age_groups, dtype=torch.long)
    sex_indices_tensor = torch.tensor(sex_indices, dtype=torch.long)
    
    return sex_indices_tensor, age_groups_tensor
