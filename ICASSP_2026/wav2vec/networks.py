import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

'''
freeze_feature_extractor=True 시 Feature Extractor를 동결 (Pretraining)
unfreeze_feature_extractor()를 호출하면 Fine-Tuning 가능
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import Wav2Vec2Model

class cnn(nn.Module):
    def __init__(self, embed_dim=512):
        super(cnn, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4))  
        )
        self.projection = nn.Linear(32 * 4 * 4, embed_dim)

    def forward(self, x):
        x = self.conv_block(x)
        B, C, H, W = x.shape
        x = x.view(B, -1)
        x = self.projection(x)
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
        self.attention_weights = None  

    def forward(self, x, cross_input):
        # Cross-attention between x and cross_input
        attn_output, attn_weights = self.multihead_attn(query=x, key=cross_input, value=cross_input)
        self.attention_weights = attn_weights 
        x = self.layer_norm(x + attn_output)
        feed_forward_output = self.feed_forward(x)
        x = self.layer_norm(x + feed_forward_output)
        return x

class CrossAttentionViT(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_layers=6, num_classes=2):
        super(CrossAttentionViT, self).__init__()

        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(embed_dim, num_heads) for _ in range(num_layers)
        ])

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x, cross_attention_input):
        self.attention_maps = []  
        for layer in self.cross_attention_layers:
            x = layer(x, cross_attention_input)
            self.attention_maps.append(layer.attention_weights)  

        x = x.unsqueeze(1).permute(1, 0, 2)
        x = self.transformer(x)
        x = x.mean(dim=0)
        x = self.classifier(x)
        return x

class CCV(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_layers=6, num_classes=2):
        super(CCV, self).__init__()
        self.encoder = cnn(embed_dim=embed_dim)
        self.decoder = CrossAttentionViT(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, num_classes=num_classes)

    def forward(self, x, cross_attention_input=None):
        x = self.encoder(x)

        if cross_attention_input is None:
            cross_attention_input = x

        x = self.decoder(x, cross_attention_input)

        # Attention Map 저장
        self.attention_maps = self.decoder.attention_maps

        return x

    def get_attention_maps(self):
        return self.attention_maps

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class Wav2Vec2ForFakeMusic(nn.Module):
    def __init__(self, num_classes=2, freeze_feature_extractor=True):
        super(Wav2Vec2ForFakeMusic, self).__init__()
        
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

        if freeze_feature_extractor:
            for param in self.wav2vec.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(self.wav2vec.config.hidden_size, 256),  # 768 → 256
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)  # 256 → 2 (Binary Classification)
        )

    def forward(self, x):
        x = x.squeeze(1)  
        output = self.wav2vec(x)
        features = output["last_hidden_state"]  # (batch_size, seq_len, feature_dim)
        pooled_features = features.mean(dim=1)  # ✅ Mean Pooling 적용 (batch_size, feature_dim)
        logits = self.classifier(pooled_features)  # (batch_size, num_classes)

        return logits, pooled_features


def visualize_attention_map(attn_map, mel_spec, layer_idx):
    attn_map = attn_map.mean(dim=1).squeeze().cpu().numpy()  # 여러 head 평균
    mel_spec = mel_spec.squeeze().cpu().numpy()

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # 1Log-Mel Spectrogram 시각화
    sns.heatmap(mel_spec, cmap='inferno', ax=axs[0])
    axs[0].set_title("Log-Mel Spectrogram")
    axs[0].set_xlabel("Time Frames")
    axs[0].set_ylabel("Mel Frequency Bins")

    # Attention Map 시각화
    sns.heatmap(attn_map, cmap='viridis', ax=axs[1])
    axs[1].set_title(f"Attention Map (Layer {layer_idx})")
    axs[1].set_xlabel("Time Frames")
    axs[1].set_ylabel("Query Positions")

    plt.tight_layout()
    plt.show()
    plt.savefig("/data/kym/AI_Music_Detection/Code/model/attention_map/crossattn.png")
