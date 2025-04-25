import torch
from Preprocess import ECGPreprocess
from Training.model_CNN import ECGTransformer
import os

# Change directory to parent folder
wd = os.path.dirname(__file__)
os.chdir(wd)


record = r'C:\Users\trevo\OneDrive\Documents\Georgia Tech\Courses\CS7643\Project\springCS7643project_data\extracted_data\extracted_data\samitrop\4991.data'
sampling_rate = 400
sex = 1
age = 3

sex = torch.tensor(sex, dtype=torch.long)
age = torch.tensor(age, dtype=torch.long)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


pretrain_path = r'best_model_by_auroc.pt'
checkpoint = torch.load(pretrain_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

ecg_filter = ECGPreprocess(sampling_rate=400)

'''
Age
Groups: 0-9, 10-19, 20-29, ..., 90+
maps to 0-9 with 9 also including nan

Sex
0 = Female
1 = Male
2 = Unknown

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

model.eval()
with torch.no_grad():
    outputs = model(ecg=processed_data, sex=sex_indices, age_group=age_groups)
    probabilities = torch.sigmoid(outputs).cpu().numpy()
'''