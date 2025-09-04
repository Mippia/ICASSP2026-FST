import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path
import math
# from deepspeed.ops.adam import FusedAdam  # 호환성 문제로 비활성화


class MusicAudioClassifier(pl.LightningModule):
    def __init__(self,
                input_dim: int,
                hidden_dim: int = 256,
                learning_rate: float = 1e-4,
                emb_model: Optional[nn.Module] = None,
                is_emb: bool = False,
                backbone: str = 'segment_transformer',
                num_classes: int = 2):
        super().__init__()
        self.save_hyperparameters()
        
        if backbone == 'segment_transformer':
            self.model = SegmentTransformer(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                mode = 'both'
            )
        elif backbone == 'fusion_segment_transformer':
            self.model = FusionSegmentTransformer(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes
            )
        # elif backbone == 'guided_segment_transformer':
        #     self.model = GuidedSegmentTransformer(
        #         input_dim=input_dim,
        #         hidden_dim=hidden_dim,
        #         num_classes=num_classes
        #     )
    
    def _process_audio_batch(self, x: torch.Tensor) -> torch.Tensor:
        B, S = x.shape[:2]  # [B, S, C, M, T] or [B, S, C, T] for wav, [B, S, 1?, embsize] for emb
        x = x.view(B*S, *x.shape[2:])  # [B*S, C, M, T] 
        if self.is_emb == False:
            _, embeddings = self.emb_model(x)  # [B*S, emb_dim]
        else:
            embeddings = x
        if embeddings.dim() == 3:
            pooled_features = embeddings.mean(dim=1) # transformer
        else:
            pooled_features = embeddings # CCV..? no need to pooling
        return pooled_features.view(B, S, -1)  # [B, S, emb_dim]
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self._process_audio_batch(x) # 이걸 freeze하고 쓰는게 사실상 윗버전임
        x = x.half()
        return self.model(x, mask)
    
    def _compute_loss_and_probs(self, y_hat: torch.Tensor, y: torch.Tensor):
        """Compute loss and probabilities based on number of classes"""
        if y_hat.size(0) == 1:
            y_hat_flat = y_hat.flatten()
            y_flat = y.flatten()
        else:
            y_hat_flat = y_hat.squeeze() if self.num_classes == 2 else y_hat
            y_flat = y
        
        if self.num_classes == 2:
            loss = F.binary_cross_entropy_with_logits(y_hat_flat, y_flat.float())
            probs = torch.sigmoid(y_hat_flat)
            preds = (probs > 0.5).long()
        else:
            loss = F.cross_entropy(y_hat_flat, y_flat.long())
            probs = F.softmax(y_hat_flat, dim=-1)
            preds = torch.argmax(y_hat_flat, dim=-1)
        
        return loss, probs, preds, y_flat.long()
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y, mask = batch
        x = x.half()
        y_hat = self(x, mask)
        
        loss, probs, preds, y_true = self._compute_loss_and_probs(y_hat, y)
        
        # 간단한 배치 손실만 로깅 (step 수준)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # 전체 에폭에 대한 메트릭 계산을 위해 예측과 실제값 저장
        if self.num_classes == 2:
            self.training_step_outputs.append({'preds': probs, 'targets': y_true, 'binary_preds': preds})
        else:
            self.training_step_outputs.append({'probs': probs, 'preds': preds, 'targets': y_true})
        
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y, mask = batch
        x = x.half()
        y_hat = self(x, mask)
        
        loss, probs, preds, y_true = self._compute_loss_and_probs(y_hat, y)
        
        # 간단한 배치 손실만 로깅 (step 수준)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # 전체 에폭에 대한 메트릭 계산을 위해 예측과 실제값 저장
        if self.num_classes == 2:
            self.validation_step_outputs.append({'preds': probs, 'targets': y_true, 'binary_preds': preds})
        else:
            self.validation_step_outputs.append({'probs': probs, 'preds': preds, 'targets': y_true})

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y, mask = batch
        x = x.half()
        y_hat = self(x, mask)
        
        loss, probs, preds, y_true = self._compute_loss_and_probs(y_hat, y)
        
        # 간단한 배치 손실만 로깅 (step 수준)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        
        # 전체 에폭에 대한 메트릭 계산을 위해 예측과 실제값 저장
        if self.num_classes == 2:
            self.test_step_outputs.append({'preds': probs, 'targets': y_true, 'binary_preds': preds})
        else:
            self.test_step_outputs.append({'probs': probs, 'preds': preds, 'targets': y_true})

    def on_train_epoch_start(self):
        # 에폭 시작 시 결과 저장용 리스트 초기화
        self.training_step_outputs = []

    def on_validation_epoch_start(self):
        # 에폭 시작 시 결과 저장용 리스트 초기화
        self.validation_step_outputs = []

    def on_test_epoch_start(self):
        # 에폭 시작 시 결과 저장용 리스트 초기화
        self.test_step_outputs = []

    def _compute_binary_metrics(self, outputs, prefix):
        """Binary classification metrics computation"""
        all_preds = torch.cat([x['preds'] for x in outputs])
        all_targets = torch.cat([x['targets'] for x in outputs])
        binary_preds = torch.cat([x['binary_preds'] for x in outputs])
        
        # 정확도 계산
        acc = (binary_preds == all_targets).float().mean()
        
        # 혼동 행렬 요소 계산
        tp = torch.sum((binary_preds == 1) & (all_targets == 1)).float()
        fp = torch.sum((binary_preds == 1) & (all_targets == 0)).float()
        tn = torch.sum((binary_preds == 0) & (all_targets == 0)).float()
        fn = torch.sum((binary_preds == 0) & (all_targets == 1)).float()
        
        # 메트릭 계산
        precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0).to(tp.device)
        recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0).to(tp.device)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0).to(tp.device)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else torch.tensor(0.0).to(tn.device)
        
        # 로깅
        self.log(f'{prefix}_acc', acc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{prefix}_precision', precision, on_epoch=True, sync_dist=True)
        self.log(f'{prefix}_recall', recall, on_epoch=True, sync_dist=True)
        self.log(f'{prefix}_f1', f1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{prefix}_specificity', specificity, on_epoch=True, sync_dist=True)
        
        if prefix in ['val', 'test']:
            # ROC-AUC 계산 (간단한 근사)
            sorted_indices = torch.argsort(all_preds, descending=True)
            sorted_targets = all_targets[sorted_indices]
            
            n_pos = torch.sum(all_targets)
            n_neg = len(all_targets) - n_pos
            
            if n_pos > 0 and n_neg > 0:
                tpr_curve = torch.cumsum(sorted_targets, dim=0) / n_pos
                fpr_curve = torch.cumsum(1 - sorted_targets, dim=0) / n_neg
                
                width = fpr_curve[1:] - fpr_curve[:-1]
                height = (tpr_curve[1:] + tpr_curve[:-1]) / 2
                auc_approx = torch.sum(width * height)
                
                self.log(f'{prefix}_auc', auc_approx, on_epoch=True, sync_dist=True)
        
        if prefix == 'test':
            balanced_acc = (recall + specificity) / 2
            self.log('test_balanced_acc', balanced_acc, on_epoch=True)

    def _compute_multiclass_metrics(self, outputs, prefix):
        """Multi-class classification metrics computation"""
        all_probs = torch.cat([x['probs'] for x in outputs])
        all_preds = torch.cat([x['preds'] for x in outputs])
        all_targets = torch.cat([x['targets'] for x in outputs])
        
        # 전체 정확도
        acc = (all_preds == all_targets).float().mean()
        self.log(f'{prefix}_acc', acc, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # 클래스별 메트릭 계산
        for class_idx in range(self.num_classes):
            # 각 클래스에 대한 이진 분류 메트릭
            class_targets = (all_targets == class_idx).long()
            class_preds = (all_preds == class_idx).long()
            
            tp = torch.sum((class_preds == 1) & (class_targets == 1)).float()
            fp = torch.sum((class_preds == 1) & (class_targets == 0)).float()
            tn = torch.sum((class_preds == 0) & (class_targets == 0)).float()
            fn = torch.sum((class_preds == 0) & (class_targets == 1)).float()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0).to(tp.device)
            recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0).to(tp.device)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0).to(tp.device)
            
            self.log(f'{prefix}_class_{class_idx}_precision', precision, on_epoch=True)
            self.log(f'{prefix}_class_{class_idx}_recall', recall, on_epoch=True)
            self.log(f'{prefix}_class_{class_idx}_f1', f1, on_epoch=True)
        
        # 매크로 평균 F1 스코어
        class_f1_scores = []
        for class_idx in range(self.num_classes):
            class_targets = (all_targets == class_idx).long()
            class_preds = (all_preds == class_idx).long()
            
            tp = torch.sum((class_preds == 1) & (class_targets == 1)).float()
            fp = torch.sum((class_preds == 1) & (class_targets == 0)).float()
            fn = torch.sum((class_preds == 0) & (class_targets == 1)).float()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0).to(tp.device)
            recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0).to(tp.device)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0).to(tp.device)
            
            class_f1_scores.append(f1)
        
        macro_f1 = torch.stack(class_f1_scores).mean()
        self.log(f'{prefix}_macro_f1', macro_f1, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_train_epoch_end(self):
        if not hasattr(self, 'training_step_outputs') or not self.training_step_outputs:
            return
        
        if self.num_classes == 2:
            self._compute_binary_metrics(self.training_step_outputs, 'train')
        else:
            self._compute_multiclass_metrics(self.training_step_outputs, 'train')

    def on_validation_epoch_end(self):
        if not hasattr(self, 'validation_step_outputs') or not self.validation_step_outputs:
            return
        
        if self.num_classes == 2:
            self._compute_binary_metrics(self.validation_step_outputs, 'val')
        else:
            self._compute_multiclass_metrics(self.validation_step_outputs, 'val')

    def on_test_epoch_end(self):
        if not hasattr(self, 'test_step_outputs') or not self.test_step_outputs:
            return
        
        if self.num_classes == 2:
            self._compute_binary_metrics(self.test_step_outputs, 'test')
        else:
            self._compute_multiclass_metrics(self.test_step_outputs, 'test')

    def configure_optimizers(self):
        # FusedAdam 대신 일반 AdamW 사용 (GLIBC 호환성 문제 해결)
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,  # Adjust based on your training epochs
            eta_min=1e-6
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss',
        }


def pad_sequence_with_mask(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for DataLoader that creates padded sequences and attention masks with fixed length (48)."""
    embeddings, labels = zip(*batch)
    fixed_len = 48  # 고정 길이

    batch_size = len(embeddings)
    feat_dim = embeddings[0].shape[-1]
    
    padded = torch.zeros((batch_size, fixed_len, feat_dim))  # 고정 길이로 패딩된 텐서
    mask = torch.ones((batch_size, fixed_len), dtype=torch.bool)  # True는 padding을 의미
    
    for i, emb in enumerate(embeddings):
        length = emb.shape[0]
        
        # 길이가 고정 길이보다 길면 자르고, 짧으면 패딩
        if length > fixed_len:
            padded[i, :] = emb[:fixed_len]  # fixed_len보다 긴 부분을 잘라서 채운다.
            mask[i, :] = False
        else:
            padded[i, :length] = emb  # 실제 데이터 길이에 맞게 채운다.
            mask[i, :length] = False  # 패딩이 아닌 부분은 False로 설정
    
    return padded, torch.tensor(labels), mask


class SegmentTransformer(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 max_sequence_length: int = 1000,
                 mode: str = 'both',
                 share_parameter: bool = False,
                 num_classes: int = 2):
        super().__init__()
        
        # Original sequence processing
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.mode = mode
        self.share_parameter = share_parameter
        self.num_classes = num_classes
        
        # Positional encoding
        position = torch.arange(max_sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-np.log(10000.0) / hidden_dim))
        pos_encoding = torch.zeros(max_sequence_length, hidden_dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pos_encoding)
        
        # Transformer for original sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.sim_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Self-similarity stream processing
        self.similarity_projection = nn.Sequential(
            nn.Conv1d(1, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Transformer for similarity stream
        self.similarity_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final classification head
        self.classification_head_dim = hidden_dim * 2 if mode == 'both' else hidden_dim
        
        # Output dimension based on number of classes
        output_dim = 1 if num_classes == 2 else num_classes
        
        self.classification_head = nn.Sequential(
            nn.Linear(self.classification_head_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # 1. Process original sequence
        x = x.half()
        x1 = self.input_projection(x)
        x1 = x1 + self.pos_encoding[:seq_len].unsqueeze(0)
        x1 = self.transformer(x1, src_key_padding_mask=padding_mask)  # padding_mask 사용

        # 2. Calculate and process self-similarity
        x_expanded = x.unsqueeze(2)
        x_transposed = x.unsqueeze(1)
        distances = torch.mean((x_expanded - x_transposed) ** 2, dim=-1)
        similarity_matrix = torch.exp(-distances)  # (batch_size, seq_len, seq_len)
        
        # 자기 유사도 마스크 생성 및 적용 (각 시점에 대한 마스크 개별 적용)
        if padding_mask is not None:
            similarity_mask = padding_mask.unsqueeze(1) | padding_mask.unsqueeze(2)  # (batch_size, seq_len, seq_len)
            similarity_matrix = similarity_matrix.masked_fill(similarity_mask, 0.0)

        # Process similarity matrix row by row using Conv1d
        x2 = similarity_matrix.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)
        x2 = x2.view(batch_size * seq_len, 1, seq_len)  # Reshape for Conv1d
        x2 = self.similarity_projection(x2)  # (batch_size * seq_len, hidden_dim, seq_len)
        x2 = x2.mean(dim=2)  # Pool across sequence dimension
        x2 = x2.view(batch_size, seq_len, -1)  # Reshape back

        x2 = x2 + self.pos_encoding[:seq_len].unsqueeze(0)
        if self.share_parameter:
            x2 = self.transformer(x2, src_key_padding_mask=padding_mask)
        else:
            x2 = self.sim_transformer(x2, src_key_padding_mask=padding_mask)  # padding_mask 사용

        # 3. Global average pooling for both streams
        if padding_mask is not None:
            mask_expanded = (~padding_mask).float().unsqueeze(-1)
            x1 = (x1 * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            x2 = (x2 * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            x1 = x1.mean(dim=1)
            x2 = x2.mean(dim=1)
        
        # 4. Combine both streams and classify
        if self.mode == 'only_emb':
            x = x1
        elif self.mode == 'only_structure':
            x = x2
        elif self.mode == 'both':
            x = torch.cat([x1, x2], dim=-1)
        x= x.half()
        return self.classification_head(x)
    

class PairwiseGuidedTransformer(nn.Module):
    """Pairwise similarity matrix를 활용한 범용 transformer layer
    
    Vision: patch간 유사도, NLP: token간 유사도, Audio: segment간 유사도 등에 활용 가능
    """
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Standard Q, K projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        
        # Pairwise-guided V projection
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, pairwise_matrix, padding_mask=None):
        """
        Args:
            x: (batch, seq_len, d_model) - sequence embeddings
            pairwise_matrix: (batch, seq_len, seq_len) - pairwise similarity/distance matrix
            padding_mask: (batch, seq_len) - padding mask
        """
        batch_size, seq_len, d_model = x.shape
        
        # Standard Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head
        Q = Q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        
        # Standard attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_model ** 0.5)
        
        # ✅ Combine with pairwise matrix
        #pairwise_expanded = pairwise_matrix.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        enhanced_scores = scores# + pairwise_expanded 이거 빼고 하기로 했죠?
        
        # Apply padding mask
        if padding_mask is not None:
            mask_4d = padding_mask.unsqueeze(1).unsqueeze(1).expand(-1, self.num_heads, seq_len, -1)
            enhanced_scores = enhanced_scores.masked_fill(mask_4d, float('-inf'))
        
        # Softmax and apply to V
        attn_weights = F.softmax(enhanced_scores, dim=-1)
        attended = torch.matmul(attn_weights, V)
        
        # Reshape and project
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.output_proj(attended)
        
        return self.norm(x + output)


class MultiScaleAdaptivePooler(nn.Module):
    """Multi-scale adaptive pooling - 다양한 도메인에서 활용 가능"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        
        # Attention-based pooling
        self.attention_pool = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Complementary pooling strategies
        self.max_pool_proj = nn.Linear(hidden_dim, hidden_dim)

        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim) 
        
        
    def forward(self, x, padding_mask=None):
        """
        Args:
            x: (batch, seq_len, hidden_dim) - sequence features
            padding_mask: (batch, seq_len) - padding mask
            actually not better than avg pooling haha
        """
        batch_size = x.size(0)
        
        # 1. Global average pooling
        if padding_mask is not None:
            mask_expanded = (~padding_mask).float().unsqueeze(-1)
            global_avg = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            global_avg = x.mean(dim=1)
        
        # # 2. Global max pooling
        # if padding_mask is not None:
        #     x_masked = x.clone()
        #     x_masked[padding_mask] = float('-inf')
        #     global_max = x_masked.max(dim=1)[0]
        # else:
        #     global_max = x.max(dim=1)[0]
        
        # global_max = self.max_pool_proj(global_max)
        
        # # 3. Attention-based pooling
        # query = self.query_token.expand(batch_size, -1, -1)
        # attn_pooled, _ = self.attention_pool(
        #     query, x, x, 
        #     key_padding_mask=padding_mask
        # )
        # attn_pooled = attn_pooled.squeeze(1)
        
        # # 4. Fuse all pooling results
        # #combined = torch.cat([global_avg, global_max, attn_pooled], dim=-1)
        # #output = self.fusion(combined)
        output = global_avg
        return output


class GuidedSegmentTransformer(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 max_sequence_length: int = 1000,
                 mode: str = 'only_emb',
                 share_parameter: bool = False,
                 num_classes: int = 2):
        super().__init__()
        
        # Original sequence processing
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.mode = mode
        self.share_parameter = share_parameter
        self.num_classes = num_classes
        
        # Positional encoding
        position = torch.arange(max_sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-np.log(10000.0) / hidden_dim))
        pos_encoding = torch.zeros(max_sequence_length, hidden_dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pos_encoding)
        
        # ✅ Pairwise-guided transformer layers (범용적 이름)
        self.pairwise_guided_layers = nn.ModuleList([
            PairwiseGuidedTransformer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Pairwise matrix processing (기존 similarity processing)
        self.pairwise_projection = nn.Sequential(
            nn.Conv1d(1, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ✅ Multi-scale adaptive pooling (범용적 이름)
        self.adaptive_pooler = MultiScaleAdaptivePooler(hidden_dim, num_heads)
        
        # Final classification head
        self.classification_head_dim = hidden_dim * 2 if mode == 'both' else hidden_dim
        output_dim = 1 if num_classes == 2 else num_classes
        
        self.classification_head = nn.Sequential(
            nn.Linear(self.classification_head_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # 1. Process sequence
        x1 = self.input_projection(x)
        x1 = x1 + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # 2. Calculate pairwise matrix (can be similarity, distance, correlation, etc.)
        x_expanded = x.unsqueeze(2)
        x_transposed = x.unsqueeze(1)
        distances = torch.mean((x_expanded - x_transposed) ** 2, dim=-1)
        pairwise_matrix = torch.exp(-distances)  # Convert distance to similarity
        
        # Apply padding mask to pairwise matrix
        if padding_mask is not None:
            pairwise_mask = padding_mask.unsqueeze(1) | padding_mask.unsqueeze(2)
            pairwise_matrix = pairwise_matrix.masked_fill(pairwise_mask, 0.0)

        # ✅ Pairwise-guided processing
        for layer in self.pairwise_guided_layers:
            x1 = layer(x1, pairwise_matrix, padding_mask)

        # 3. Process pairwise matrix as separate stream (optional)
        if self.mode in ['only_structure', 'both']:
            x2 = pairwise_matrix.unsqueeze(1)
            x2 = x2.view(batch_size * seq_len, 1, seq_len)
            x2 = self.pairwise_projection(x2)
            x2 = x2.mean(dim=2)
            x2 = x2.view(batch_size, seq_len, -1)
            x2 = x2 + self.pos_encoding[:seq_len].unsqueeze(0)

        # ✅ Multi-scale adaptive pooling
        if self.mode == 'only_emb':
            x = self.adaptive_pooler(x1, padding_mask)
        elif self.mode == 'only_structure':
            x = self.adaptive_pooler(x2, padding_mask)
        elif self.mode == 'both':
            x1_pooled = self.adaptive_pooler(x1, padding_mask)
            x2_pooled = self.adaptive_pooler(x2, padding_mask)
            x = torch.cat([x1_pooled, x2_pooled], dim=-1)
        
        x = x
        return self.classification_head(x)
    

class CrossModalFusionLayer(nn.Module):
    """Structure와 Embedding 정보를 점진적으로 융합"""
    
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        
        # Cross-attention: embedding이 structure를 query하고, structure가 embedding을 query
        self.emb_to_struct_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.struct_to_emb_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        
        # Fusion gate (어느 정보를 얼마나 믿을지)
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, emb_features, struct_features, padding_mask=None):
        """
        emb_features: (batch, seq_len, d_model) - 메인 embedding 정보
        struct_features: (batch, seq_len, d_model) - structure 정보
        """
        
        # 1. Embedding이 Structure 정보를 참조
        emb_enhanced, _ = self.emb_to_struct_attn(
            emb_features, struct_features, struct_features,
            key_padding_mask=padding_mask
        )
        emb_enhanced = self.norm1(emb_features + emb_enhanced)
        
        # 2. Structure가 Embedding 정보를 참조
        struct_enhanced, _ = self.struct_to_emb_attn(
            struct_features, emb_features, emb_features,
            key_padding_mask=padding_mask
        )
        struct_enhanced = self.norm2(struct_features + struct_enhanced)
        
        # 3. Adaptive fusion (둘 중 어느 것을 더 믿을지 학습)
        combined = torch.cat([emb_enhanced, struct_enhanced], dim=-1)
        gate_weight = self.fusion_gate(combined)  # (batch, seq_len, d_model)
        
        # Gated combination
        fused = gate_weight * emb_enhanced + (1 - gate_weight) * struct_enhanced
        
        return fused


class FusionSegmentTransformer(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 max_sequence_length: int = 1000,
                 mode: str = 'both',  # 기본값을 both로
                 share_parameter: bool = False,
                 num_classes: int = 2):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.mode = mode
        self.num_classes = num_classes
        
        # Positional encoding
        position = torch.arange(max_sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-np.log(10000.0) / hidden_dim))
        pos_encoding = torch.zeros(max_sequence_length, hidden_dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pos_encoding)
        
        # ✅ Embedding stream: Pairwise-guided transformer
        self.embedding_layers = nn.ModuleList([
            PairwiseGuidedTransformer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # ✅ Structure stream: Pairwise matrix processing
        self.pairwise_projection = nn.Sequential(
            nn.Conv1d(1, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Structure transformer layers
        self.structure_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers // 2)  # 절반만 사용
        ])
        
        # ✅ Cross-modal fusion layers (핵심!)
        self.fusion_layers = nn.ModuleList([
            CrossModalFusionLayer(hidden_dim, num_heads)
            for _ in range(1)  # fusion은 하나만 써야 gate가 유의미해질듯
        ])
        
        # Adaptive pooling
        self.adaptive_pooler = MultiScaleAdaptivePooler(hidden_dim, num_heads)
        
        # Final classification head (이제 단일 차원)
        output_dim = 1 if num_classes == 2 else num_classes
        
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # 더 이상 concat 안함
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # 1. Initialize both streams
        x_emb = self.input_projection(x)
        x_emb = x_emb + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # 2. Calculate pairwise matrix
        x_expanded = x.unsqueeze(2)
        x_transposed = x.unsqueeze(1)
        distances = torch.mean((x_expanded - x_transposed) ** 2, dim=-1)
        pairwise_matrix = torch.exp(-distances)
        
        if padding_mask is not None:
            pairwise_mask = padding_mask.unsqueeze(1) | padding_mask.unsqueeze(2)
            pairwise_matrix = pairwise_matrix.masked_fill(pairwise_mask, 0.0)

        # 3. Process structure stream
        x_struct = pairwise_matrix.unsqueeze(1)
        x_struct = x_struct.view(batch_size * seq_len, 1, seq_len)
        x_struct = self.pairwise_projection(x_struct)
        x_struct = x_struct.mean(dim=2)
        x_struct = x_struct.view(batch_size, seq_len, -1)
        x_struct = x_struct + self.pos_encoding[:seq_len].unsqueeze(0)
        
        for struct_layer in self.structure_layers:
            x_struct = struct_layer(x_struct, src_key_padding_mask=padding_mask)
        
        # 4. Process embedding stream (with pairwise guidance)
        for emb_layer in self.embedding_layers:
            x_emb = emb_layer(x_emb, pairwise_matrix, padding_mask)
        
        # ✅ 5. Progressive Cross-modal Fusion (핵심!)
        fused = x_emb  # 시작은 embedding에서
        for fusion_layer in self.fusion_layers:
            fused = fusion_layer(fused, x_struct, padding_mask)
            # 이제 fused는 embedding + structure 정보를 모두 포함
        
        # 6. Final pooling and classification
        pooled = self.adaptive_pooler(fused, padding_mask)
        
        pooled = pooled.half()
        return self.classification_head(pooled)
    
    import torch