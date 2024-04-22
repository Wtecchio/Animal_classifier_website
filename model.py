import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer


class AnimalClassifier(nn.Module):
    def __init__(self, num_classes, feature_size=2048, num_layers=2, num_heads=8, dropout=0.1):
        super(AnimalClassifier, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()  # Replace fully connected layer
        self.feature_size = feature_size
        
        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        
        # Transformer Decoder
        decoder_layer = TransformerDecoderLayer(d_model=feature_size, nhead=num_heads, dropout=dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers)
        
        # Final linear layer
        self.fc = nn.Linear(feature_size, num_classes)

    def forward(self, src, tgt):
        # Extract features from ResNet
        src_features = self.resnet(src)
        tgt_features = self.resnet(tgt)

        # Ensure both source and target features have batch dimension
        src_features = src_features.unsqueeze(0)  # Add batch dimension
        tgt_features = tgt_features.unsqueeze(0)  # Add batch dimension

        # Encoder forward pass
        src_encoded = self.transformer_encoder(src_features)

        # Decoder forward pass
        tgt_encoded = self.transformer_decoder(tgt_features, src_encoded)

        # Remove batch dimension before classification
        src_encoded = src_encoded.squeeze(0)
        tgt_encoded = tgt_encoded.squeeze(0)

        # Classification layer
        output = self.fc(tgt_encoded)
        return output