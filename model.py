import torch
import torch.nn as nn  # Make sure this import is present
from torchvision.models import resnet50
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer

class AnimalClassifier(nn.Module):
    def __init__(self, num_classes, feature_size=2048, num_layers=2, num_heads=8, dropout=0.1):
        super(AnimalClassifier, self).__init__()
        self.resnet = resnet50(pretrained=True)
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

    def forward(self, src, tgt=None):
        src_features = self.resnet(src)
        src_features = src_features.unsqueeze(0)

        if tgt is not None:
            tgt_features = self.resnet(tgt)
            tgt_features = tgt_features.unsqueeze(0)
            tgt_encoded = self.transformer_decoder(tgt_features, src_features)
            tgt_encoded = tgt_encoded.squeeze(0)
            output = self.fc(tgt_encoded)
        else:
            # Inference mode: only use the encoder part
            src_encoded = self.transformer_encoder(src_features)
            src_encoded = src_encoded.squeeze(0)
            output = self.fc(src_encoded)

        return output