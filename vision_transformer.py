import torch
from torch import nn
from patch_embedding import PatchEmbedding
from kan import KANLinear

class VisionTransformer(nn.Module):
    def __init__(self, num_patches, img_size, num_classes, patch_size, embed_dim, num_encoders, num_heads, hidden_dim, dropout, activation, in_channels):
        super().__init__()
        
        self.embeddings_block = PatchEmbedding(embed_dim, patch_size, num_patches, dropout, in_channels)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model = embed_dim, nhead = num_heads, dropout = dropout, activation = activation, batch_first = True, norm_first = True)
        
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers = num_encoders)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape = embed_dim),
            nn.Linear(in_features = embed_dim, out_features = num_classes)
        )
        
    def forward(self, x):
        x = self.embeddings_block(x)
        
        x = self.encoder_blocks(x)
        
        x = self.mlp_head(x[:, 0, :]) # taking only cls_token
        
        return x
    
class VisionTransformerKAN(nn.Module):
    def __init__(self, num_patches, img_size, num_classes, patch_size, embed_dim, num_encoders, num_heads, hidden_dim, dropout, activation, in_channels):
        super().__init__()
        
        self.embeddings_block = PatchEmbedding(embed_dim, patch_size, num_patches, dropout, in_channels)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model = embed_dim, nhead = num_heads, dropout = dropout, activation = activation, batch_first = True, norm_first = True)
        
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers = num_encoders)
        
        self.kan_head = nn.Sequential(
            nn.LayerNorm(normalized_shape = embed_dim),
            KANLinear(in_features = embed_dim, out_features = num_classes)
        )
        
    def forward(self, x):
        x = self.embeddings_block(x)
        
        x = self.encoder_blocks(x)
        
        x = self.kan_head(x[:, 0, :]) # taking only cls_token
        
        return x