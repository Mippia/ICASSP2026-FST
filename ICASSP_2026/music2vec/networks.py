import torch
import torch.nn as nn
from transformers import Data2VecAudioModel, Wav2Vec2Processor

class Music2VecClassifier(nn.Module):
    def __init__(self, num_classes=2, freeze_feature_extractor=True):
        super(Music2VecClassifier, self).__init__()

        self.processor = Wav2Vec2Processor.from_pretrained("facebook/data2vec-audio-base-960h")
        self.music2vec = Data2VecAudioModel.from_pretrained("m-a-p/music2vec-v1")

        if freeze_feature_extractor:
            for param in self.music2vec.parameters():
                param.requires_grad = False

        # Conv1d for learnable weighted average across layers
        self.conv1d = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.music2vec.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_values):
        input_values = input_values.squeeze(1)  # Ensure shape [batch, time]

        with torch.no_grad():
            outputs = self.music2vec(input_values, output_hidden_states=True)
        hidden_states = torch.stack(outputs.hidden_states) 
        time_reduced = hidden_states.mean(dim=2)  
        time_reduced = time_reduced.permute(1, 0, 2)  
        weighted_avg = self.conv1d(time_reduced).squeeze(1)  

        return self.classifier(weighted_avg), weighted_avg

    def unfreeze_feature_extractor(self):
        for param in self.music2vec.parameters():
            param.requires_grad = True

class Music2VecFeatureExtractor(nn.Module):
    def __init__(self, freeze_feature_extractor=True):
        super(Music2VecFeatureExtractor, self).__init__()
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/data2vec-audio-base-960h")
        self.music2vec = Data2VecAudioModel.from_pretrained("m-a-p/music2vec-v1")
        
        if freeze_feature_extractor:
            for param in self.music2vec.parameters():
                param.requires_grad = False

        # Conv1d for learnable weighted average across layers
        self.conv1d = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1)

    def forward(self, input_values):
        # input_values: [batch, time]
        input_values = input_values.squeeze(1)
        with torch.no_grad():
            outputs = self.music2vec(input_values, output_hidden_states=True)
        hidden_states = torch.stack(outputs.hidden_states)  # [num_layers, batch, time, hidden_dim]
        time_reduced = hidden_states.mean(dim=2)             # [num_layers, batch, hidden_dim]
        time_reduced = time_reduced.permute(1, 0, 2)           # [batch, num_layers, hidden_dim]
        weighted_avg = self.conv1d(time_reduced).squeeze(1)    # [batch, hidden_dim]
        return weighted_avg

'''
music2vec+CCV
# '''
# import torch
# import torch.nn as nn
# from transformers import Data2VecAudioModel, Wav2Vec2Processor
# import torch.nn.functional as F


# ###  Music2Vec Feature Extractor (Pretrained Model)
# class Music2VecFeatureExtractor(nn.Module):
#     def __init__(self, freeze_feature_extractor=True):
#         super(Music2VecFeatureExtractor, self).__init__()

#         self.processor = Wav2Vec2Processor.from_pretrained("facebook/data2vec-audio-base-960h")
#         self.music2vec = Data2VecAudioModel.from_pretrained("m-a-p/music2vec-v1")

#         if freeze_feature_extractor:
#             for param in self.music2vec.parameters():
#                 param.requires_grad = False 

#         # Conv1d for learnable weighted average across layers
#         self.conv1d = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1)

#     def forward(self, input_values):
#         with torch.no_grad():
#             outputs = self.music2vec(input_values, output_hidden_states=True)

#         hidden_states = torch.stack(outputs.hidden_states)  # [13, batch, time, hidden_size]
#         time_reduced = hidden_states.mean(dim=2)  # 평균 풀링: [13, batch, hidden_size]
#         time_reduced = time_reduced.permute(1, 0, 2)  # [batch, 13, hidden_size]
#         weighted_avg = self.conv1d(time_reduced).squeeze(1)  # [batch, hidden_size]

#         return weighted_avg  # Extracted feature representation


#     def unfreeze_feature_extractor(self):
#         for param in self.music2vec.parameters():
#             param.requires_grad = True  # Unfreeze for Fine-tuning

# ###  CNN Feature Extractor for CCV
class CNNEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super(CNNEncoder, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,1)),  # 기존 MaxPool2d(2)를 MaxPool2d((2,1))으로 변경
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1,1)),  # 추가된 MaxPool2d(1,1)로 크기 유지
            nn.AdaptiveAvgPool2d((4, 4))  # 최종 크기 조정
        )
        self.projection = nn.Linear(32 * 4 * 4, embed_dim)

    def forward(self, x):
        # print(f"Input shape before CNNEncoder: {x.shape}")  # 디버깅용 출력
        x = self.conv_block(x)
        B, C, H, W = x.shape
        x = x.view(B, -1)
        x = self.projection(x)
        return x


###  Cross-Attention Module
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
        attn_output, attn_weights = self.multihead_attn(query=x, key=cross_input, value=cross_input)
        self.attention_weights = attn_weights 
        x = self.layer_norm(x + attn_output)
        feed_forward_output = self.feed_forward(x)
        x = self.layer_norm(x + feed_forward_output)
        return x

### Cross-Attention Transformer
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

###  CCV Model (Final Classifier)
# class CCV(nn.Module):
#     def __init__(self, embed_dim=512, num_heads=8, num_layers=6, num_classes=2, freeze_feature_extractor=True):
#         super(CCV, self).__init__()

#         self.music2vec_extractor = Music2VecClassifier(freeze_feature_extractor=freeze_feature_extractor)

#         # CNN Encoder for Image Representation
#         self.encoder = CNNEncoder(embed_dim=embed_dim)

#         # Transformer with Cross-Attention
#         self.decoder = CrossAttentionViT(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, num_classes=num_classes)

#     def forward(self, x, cross_attention_input=None):
#         x = self.music2vec_extractor(x)  
#         # print(f"After Music2VecExtractor: {x.shape}")  # (batch, 2) 출력됨

#         # CNNEncoder가 기대하는 입력 크기 맞추기
#         x = x.unsqueeze(1).unsqueeze(-1)  # (batch, 1, 2, 1) 형태로 변환
#         # print(f"Before CNNEncoder: {x.shape}")  # CNN 입력 확인

#         x = self.encoder(x)

#         if cross_attention_input is None:
#             cross_attention_input = x

#         x = self.decoder(x, cross_attention_input)

#         return x

class CCV(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, num_layers=6, num_classes=2, freeze_feature_extractor=True):
        super(CCV, self).__init__()
        self.feature_extractor = Music2VecFeatureExtractor(freeze_feature_extractor=freeze_feature_extractor)
        
        # Cross-Attention Transformer
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(embed_dim, num_heads) for _ in range(num_layers)
        ])

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, input_values):
        # Extract feature embeddings
        features = self.feature_extractor(input_values)  # [batch, feature_dim]
        # Average over layer dimension if necessary (여기서는 이미 [batch, hidden_dim])
        # Apply Cross-Attention Layers
        for layer in self.cross_attention_layers:
            features = layer(features.unsqueeze(1), features.unsqueeze(1)).squeeze(1)
        # Transformer Encoding
        encoded = self.transformer(features.unsqueeze(1))
        encoded = encoded.mean(dim=1)
        # Classification Head
        logits = self.classifier(encoded)
        return logits

    def get_attention_maps(self):
        # 만약 CrossAttentionLayer의 attention_maps를 사용하고 싶다면 구현
        return None
