import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class MERTFeatureExtractor(nn.Module):
    def __init__(self, freeze_feature_extractor=True):
        super(MERTFeatureExtractor, self).__init__()
        config = AutoConfig.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
        if not hasattr(config, "conv_pos_batch_norm"):
            setattr(config, "conv_pos_batch_norm", False)
        self.mert = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", config=config, trust_remote_code=True)

        if freeze_feature_extractor:
            self.freeze()

    def forward(self, input_values):
        # 입력: [batch, time]
        # 사전학습된 MERT의 hidden_states 추출 (예시로 모든 레이어의 hidden state 사용)
        with torch.no_grad():
            outputs = self.mert(input_values, output_hidden_states=True)
        # hidden_states: tuple of [batch, time, feature_dim]
        # 여러 레이어의 hidden state를 스택한 뒤 시간축에 대해 평균하여 feature를 얻음
        hidden_states = torch.stack(outputs.hidden_states)  # [num_layers, batch, time, feature_dim]
        hidden_states = hidden_states.detach().clone().requires_grad_(True)
        time_reduced = hidden_states.mean(dim=2)  # [num_layers, batch, feature_dim]
        time_reduced = time_reduced.permute(1, 0, 2)  # [batch, num_layers, feature_dim]
        return time_reduced

    def freeze(self):
        for param in self.mert.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.mert.parameters():
            param.requires_grad = True


class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x, cross_input):
        # x와 cross_input 간의 어텐션 수행
        attn_output, _ = self.multihead_attn(query=x, key=cross_input, value=cross_input)
        x = self.layer_norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        return x


class CCV(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, num_layers=6, num_classes=2, freeze_feature_extractor=True):
        super(CCV, self).__init__()
        # MERT 기반 feature extractor (pretraining weight로부터 유의미한 피쳐 추출)
        self.feature_extractor = MERTFeatureExtractor(freeze_feature_extractor=freeze_feature_extractor)
        # Cross-Attention 레이어 여러 층
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(embed_dim, num_heads) for _ in range(num_layers)
        ])
        # Transformer Encoder (배치 차원 고려)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 분류기
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )


    def forward(self, input_values):
        """
        input_values: Tensor [batch, time]
        1. MERT로부터 feature 추출 → [batch, num_layers, feature_dim]
        2. 임베딩 차원 맞추기 위해 transpose → [batch, feature_dim, num_layers]
        3. Cross-Attention 적용
        4. Transformer Encoding 후 평균 풀링
        5. 분류기 통과하여 최종 출력(logits) 반환
        """
        features = self.feature_extractor(input_values)  # [batch, num_layers, feature_dim]
        # embed_dim는 보통 feature_dim과 동일하게 맞춤 (예시: 768)
        # features = features.permute(0, 2, 1)  # [batch, embed_dim, num_layers]
        
        # Cross-Attention 적용 (여기서는 자기자신과의 어텐션으로 예시)
        for layer in self.cross_attention_layers:
            features = layer(features, features)
        
        # Transformer Encoder를 위해 시간 축(여기서는 num_layers 축)에 대해 평균
        features = features.mean(dim=1).unsqueeze(1)  # [batch, 1, embed_dim]
        encoded = self.transformer(features)  # [batch, 1, embed_dim]
        encoded = encoded.mean(dim=1)  # [batch, embed_dim]
        output = self.classifier(encoded)  # [batch, num_classes]
        return output, encoded

    def unfreeze_feature_extractor(self):
        self.feature_extractor.unfreeze()
