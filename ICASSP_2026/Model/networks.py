import torch
import torch.nn as nn

class audiocnn(nn.Module):
    def __init__(self, num_classes=2):
        super(audiocnn, self).__init__()
        self.conv_block = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((4,4))  # 최종 -> (B,32,4,4)
    )
        self.fc_block = nn.Sequential(
            nn.Linear(32*4*4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        # x.shape: (B,32,new_freq,new_time)

        # 1) Flatten
        B, C, H, W = x.shape  # 동적 shape
        x = x.view(B, -1)     # (B, 32*H*W)

        # 2) FC
        x = self.fc_block(x)
        return x
    
class AudioCNN(nn.Module):
    def __init__(self, embed_dim=512):
        super(AudioCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4))  # 최종 -> (B, 32, 4, 4)
        )
        self.projection = nn.Linear(32 * 4 * 4, embed_dim)

    def forward(self, x):
        x = self.conv_block(x)
        B, C, H, W = x.shape
        x = x.view(B, -1)  # Flatten (B, C * H * W)
        x = self.projection(x)  # Project to embed_dim
        return x

class ViTDecoder(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_layers=6, num_classes=2):
        super(ViTDecoder, self).__init__()

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # Transformer expects input of shape (seq_len, batch, embed_dim)
        x = x.unsqueeze(1).permute(1, 0, 2)  # Add sequence dim (1, B, embed_dim)
        x = self.transformer(x)  # Pass through Transformer
        x = x.mean(dim=0)  # Take the mean over the sequence dimension (B, embed_dim)

        x = self.classifier(x)  # Classification head
        return x

class AudioCNNWithViTDecoder(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_layers=6, num_classes=2):
        super(AudioCNNWithViTDecoder, self).__init__()
        self.encoder = AudioCNN(embed_dim=embed_dim)
        self.decoder = ViTDecoder(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, num_classes=num_classes)

    def forward(self, x):
        x = self.encoder(x)  # Pass through AudioCNN encoder
        x = self.decoder(x)  # Pass through ViT decoder
        return x


# class AudioCNN(nn.Module):
#     def __init__(self, num_classes=2):
#         super(AudioCNN, self).__init__()
#         self.conv_block = nn.Sequential(
#         nn.Conv2d(1, 16, kernel_size=3, padding=1),
#         nn.ReLU(),
#         nn.MaxPool2d(2),
#         nn.Conv2d(16, 32, kernel_size=3, padding=1),
#         nn.ReLU(),
#         nn.MaxPool2d(2),
#         nn.AdaptiveAvgPool2d((4,4))  # 최종 -> (B,32,4,4)
#     )
#         self.fc_block = nn.Sequential(
#             nn.Linear(32*4*4, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_classes)
#         )


#     def forward(self, x):
#         x = self.conv_block(x)
#         # x.shape: (B,32,new_freq,new_time)

#         # 1) Flatten
#         B, C, H, W = x.shape  # 동적 shape
#         x = x.view(B, -1)     # (B, 32*H*W)

#         # 2) FC
#         x = self.fc_block(x)
#         return x



class audio_crossattn(nn.Module):
    def __init__(self, embed_dim=512):
        super(audio_crossattn, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4))  # 최종 출력 -> (B, 32, 4, 4)
        )
        self.projection = nn.Linear(32 * 4 * 4, embed_dim)

    def forward(self, x):
        x = self.conv_block(x)  # Convolutional feature extraction
        B, C, H, W = x.shape
        x = x.view(B, -1)  # Flatten (B, C * H * W)
        x = self.projection(x)  # Linear projection to embed_dim
        return x
    

class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x, cross_input):
        # Cross-attention between x and cross_input
        attn_output, _ = self.multihead_attn(query=x, key=cross_input, value=cross_input)
        x = self.layer_norm(x + attn_output)  # Add & Norm
        feed_forward_output = self.feed_forward(x)
        x = self.layer_norm(x + feed_forward_output)  # Add & Norm
        return x

class ViTDecoderWithCrossAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_layers=6, num_classes=2):
        super(ViTDecoderWithCrossAttention, self).__init__()

        # Cross-Attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(embed_dim, num_heads) for _ in range(num_layers)
        ])

        # Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x, cross_attention_input):
        # Pass through Cross-Attention layers
        for layer in self.cross_attention_layers:
            x = layer(x, cross_attention_input)

        # Transformer expects input of shape (seq_len, batch, embed_dim)
        x = x.unsqueeze(1).permute(1, 0, 2)  # Add sequence dim (1, B, embed_dim)
        x = self.transformer(x)  # Pass through Transformer
        embedding = x.mean(dim=0)  # Take the mean over the sequence dimension (B, embed_dim)

        # Classification head
        x = self.classifier(embedding)
        return x, embedding

# class AudioCNNWithViTDecoderAndCrossAttention(nn.Module):
#     def __init__(self, embed_dim=512, num_heads=8, num_layers=6, num_classes=2):
#         super(AudioCNNWithViTDecoderAndCrossAttention, self).__init__()
#         self.encoder = audio_crossattn(embed_dim=embed_dim)
#         self.decoder = ViTDecoderWithCrossAttention(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, num_classes=num_classes)

#     def forward(self, x, cross_attention_input):
#         # Pass through AudioCNN encoder
#         x = self.encoder(x)

#         # Pass through ViTDecoder with Cross-Attention
#         x = self.decoder(x, cross_attention_input)
#         return x
class CCV(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_layers=6, num_classes=2, freeze_feature_extractor=True):
        super(CCV, self).__init__()
        self.encoder = AudioCNN(embed_dim=embed_dim)
        self.decoder = ViTDecoderWithCrossAttention(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, num_classes=num_classes)
        if freeze_feature_extractor:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
    def forward(self, x, cross_attention_input=None):
        # Pass through AudioCNN encoder
        x = self.encoder(x)

        # If cross_attention_input is not provided, use the encoder output
        if cross_attention_input is None:
            cross_attention_input = x

        # Pass through ViTDecoder with Cross-Attention
        x, embedding = self.decoder(x, cross_attention_input)
        return x, embedding

#---------------------------------------------------------
'''
audiocnn weight frozen
crossatten decoder -lora tuning
'''

