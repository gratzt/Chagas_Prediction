import torch
import numpy as np
import os
from Preprocess import ECGPreprocess
from Training.model_CNN import ECGTransformer

def load_model(device):
    """
    Load the pretrained ECG Transformer model
    
    Args:
        device: Computing device (cpu or cuda)
        
    Returns:
        Loaded model
    """
    # Instantiate model architecture (same as pretraining)
    model = ECGTransformer(
        input_channels=12,      # 12-lead ECG
        seq_length=4096,        # ECG sequence length
        d_model=64,             # Reduced embedding dimension (was 128)
        nhead=4,                # Reduced number of attention heads (was 8)
        num_encoder_layers=3,   # Reduced number of layers (was 6)
        dim_feedforward=256,    # Reduced feed-forward dimension (was 512)
        dropout=0.1,
        cnn_channels=32         # Reduced number of CNN filters (was 64)
    ).to(device)

    # Get current directory
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    pretrain_path = os.path.join(curr_dir, 'best_model_by_auroc.pt')
    
    # Load pretrained weights
    try:
        checkpoint = torch.load(pretrain_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Model loaded successfully from {pretrain_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {pretrain_path}. Please ensure the model file exists.")
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")
    
    # Set model to evaluation mode
    model.eval()
    
    return model

def predict(model, record, sex, age, sampling_rate, device):
    """
    Process ECG data and predict Chagas disease probability
    
    Args:
        model: Loaded ECGTransformer model
        record: WFDB record object
        sex: Tensor containing sex value
        age: Tensor containing age group value
        sampling_rate: Sampling rate of the ECG data
        device: Computing device (cpu or cuda)
        
    Returns:
        Probability of Chagas disease
    """
    # Prepare inputs
    sex = torch.unsqueeze(sex, 0).to(device)
    age = torch.unsqueeze(age, 0).to(device)
    
    # Preprocess ECG data
    ecg_filter = ECGPreprocess(sampling_rate=sampling_rate)
    data = ecg_filter.process_wfdb_files(
        record=record,
        apply_resample=True,
        apply_highpass=True,
        apply_lowpass=True,
        pad_to_length=4096,
        #apply_wavelet=False,
        device=device
    )
    
    # Add batch dimension
    data = torch.unsqueeze(data, 0)
    
    # Print shapes for debugging
    print(f"Data shape: {data.shape}")
    print(f"Sex shape: {sex.shape}")
    print(f"Age shape: {age.shape}")
    
    # Run inference
    with torch.no_grad():
        try:
            outputs = model(ecg=data, sex=sex, age_group=age)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            return float(probabilities[0][0])
        except Exception as e:
            print(f"Error during model inference: {str(e)}")
            raise