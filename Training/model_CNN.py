import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    A lightweight convolutional block with skip connection and fewer parameters
    """
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.1):
        super(ConvBlock, self).__init__()
        
        # Calculate padding to maintain sequence length
        padding = (kernel_size - 1) // 2
        
        # Use depthwise separable convolution for efficiency
        self.depthwise_conv = nn.Conv1d(
            in_channels, 
            in_channels, 
            kernel_size, 
            padding=padding, 
            groups=in_channels
        )
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, 1)
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection if input and output channels are different
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        # Input shape: [batch_size, channels, sequence_length]
        identity = x
        
        # Identity (skip) path with optional conv
        identity = self.skip(identity)

        # Conv path with depthwise separable convolution
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Add skip connection
        x = x + identity
        
        return x

class ConvTrack(nn.Module):
    """
    A track of two sequential LightConvBlocks (reduced from three)
    """
    def __init__(self, in_channels, mid_channels, kernel_size, dropout=0.1):
        super(ConvTrack, self).__init__()
        
        self.block1 = ConvBlock(in_channels, mid_channels, kernel_size, dropout)
        self.block2 = ConvBlock(mid_channels, mid_channels, kernel_size, dropout)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

class ECGTransformer(nn.Module):
    """
    Lightweight Transformer-based model for ECG binary classification with efficient CNN preprocessor.
    """
    def __init__(self, 
                 input_channels=12,      # 12-lead ECG
                 seq_length=4096,        # ECG sequence length
                 d_model=64,             # Reduced embedding dimension (was 128)
                 nhead=4,                # Reduced number of attention heads (was 8)
                 num_encoder_layers=3,   # Reduced number of layers (was 6)
                 dim_feedforward=256,    # Reduced feed-forward dimension (was 512)
                 dropout=0.1,
                 cnn_channels=32         # Reduced number of CNN filters (was 64)
                 ):
        super(ECGTransformer, self).__init__()

        # Demographic Embeddings
        self.sex_embedding = nn.Embedding(3, d_model // 4)
        self.age_embedding = nn.Embedding(10, d_model // 4)

        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Use only three parallel CNN tracks with different kernel sizes (reduced from 4)
        self.track1 = ConvTrack(input_channels, cnn_channels, kernel_size=15, dropout=dropout)
        self.track2 = ConvTrack(input_channels, cnn_channels, kernel_size=31, dropout=dropout)
        self.track3 = ConvTrack(input_channels, cnn_channels, kernel_size=63, dropout=dropout)
        
        # After concatenation, we'll have 3 * cnn_channels features
        total_cnn_channels = 3 * cnn_channels
        
        # Project concatenated CNN features to transformer dimension
        self.cnn_projection = nn.Linear(total_cnn_channels, d_model)
        
        # Skip every other position to reduce sequence length by half
        self.downsample_factor = 2
        self.effective_seq_length = (seq_length + self.downsample_factor - 1) // self.downsample_factor
        
        # Positional embedding with reduced size
        self.pos_embedding = nn.Embedding(self.effective_seq_length, d_model)
        
        # Transformer encoder with fewer parameters
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_encoder_layers
        )
        
        # Create a fusion network for non-linear interactions
        # Combines sex and age with the classificaiton token output
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model + d_model // 2, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Smaller classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),  # Reduced from 256
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # Binary classification
        )
        
    def forward(self, ecg, sex='', age_group=''):
        """
        Forward pass of the model.
        
        Args:
            ecg: Tensor of shape [batch_size, seq_length, channels]
            
        Returns:
            logits: Tensor of shape [batch_size, 1]
        """
        
        batch_size, seq_length, channels = ecg.size()
        
        # Transpose ECG to [batch_size, channels, seq_length] for CNN operations
        ecg_cnn = ecg.transpose(1, 2)
        
        # Apply each track to the ECG data
        track1_out = self.track1(ecg_cnn)  # [batch_size, cnn_channels, seq_length]
        track2_out = self.track2(ecg_cnn)
        track3_out = self.track3(ecg_cnn)
        
        # Concatenate along the channel dimension
        multi_scale_features = torch.cat([track1_out, track2_out, track3_out], dim=1)
        
        # Downsample sequence length by taking every n-th sample
        multi_scale_features = multi_scale_features[:, :, ::self.downsample_factor]
        
        # Transpose back to [batch_size, seq_length, channels] for transformer
        multi_scale_features = multi_scale_features.transpose(1, 2)
        
        # Project to d_model dimension
        x = self.cnn_projection(multi_scale_features)
        
        # Add positional encoding
        positions = torch.arange(self.effective_seq_length, device=x.device)
        pos = self.pos_embedding(positions).unsqueeze(0)
        pos = pos.expand(batch_size, -1, -1)
        
        # Combine embeddings
        x = x + pos
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Extract CLS token output for classification
        cls_features = x[:, 0, :]
        
        # Incorporate demographic features if provided
        if sex is not None and age_group is not None:
            sex_emb = self.sex_embedding(sex)
            age_emb = self.age_embedding(age_group)
            demo_features = torch.cat([sex_emb, age_emb], dim=1)
            
            # Concatenate CLS features with demographic features
            combined_features = torch.cat([cls_features, demo_features], dim=1)
            
            # Pass through fusion network for non-linear interactions
            cls_features = self.feature_fusion(combined_features)
            
        # Classification head
        logits = self.classifier(cls_features)
        
        return logits.squeeze(1)
    
    def predict(self, ecg, sex, age_group, threshold=0.5):
        """
        Make a binary prediction.
        
        Returns:
            predictions: Binary predictions (0 or 1)
        """
        with torch.no_grad():
            logits = self.forward(ecg, sex, age_group)
            predictions = torch.sigmoid(logits) > threshold
        return predictions.long()