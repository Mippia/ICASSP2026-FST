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
from transformers import AutoModel, AutoConfig

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x, cross_input):
        attn_output, _ = self.multihead_attn(query=x, key=cross_input, value=cross_input)
        attn_output = F.normalize(attn_output, p=2, dim=-1)
        x = self.layer_norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        ff_output = F.normalize(ff_output, p=2, dim=-1)
        x = self.layer_norm2(x + ff_output)
        return x


class CrossAttn_Transformer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_layers=6, num_classes=2):
        super(CrossAttn_Transformer, self).__init__()

        self.cross_attention_layers = nn.ModuleList([
            CrossAttention(embed_dim, num_heads) for _ in range(num_layers)
        ])

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.encoder =  nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, cross_attention_input):
        self.attention_maps = []  
        for layer in self.cross_attention_layers:
            x = layer(x, cross_attention_input)

        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.mean(dim=0)
        embedding  = self.encoder(x)
        x = self.classifier(embedding)
        return x, embedding

class MERT(nn.Module):
    def __init__(self, freeze_feature_extractor=True):
        super(MERT, self).__init__()
        config = AutoConfig.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
        if not hasattr(config, "conv_pos_batch_norm"):
            setattr(config, "conv_pos_batch_norm", False)
        self.mert = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", config=config, trust_remote_code=True)
        
        if freeze_feature_extractor:
            self.freeze()

    def forward(self, input_values):
        with torch.no_grad():
            outputs = self.mert(input_values, output_hidden_states=True)
        hidden_states = torch.stack(outputs.hidden_states)
        hidden_states = hidden_states.detach().clone().requires_grad_(True)
        
        # 먼저 정규화
        hidden_states = F.normalize(hidden_states, p=2, dim=-1)
        # 그 다음 clamp
        hidden_states = torch.clamp(hidden_states, -3.0, 3.0)
        
        time_reduced = hidden_states.mean(dim=2)
        time_reduced = time_reduced.permute(1, 0, 2)
        
        # 최종 출력도 정규화 후 clamp
        time_reduced = F.normalize(time_reduced, p=2, dim=-1)
        time_reduced = torch.clamp(time_reduced, -2.0, 2.0)
        
        return time_reduced

    def freeze(self):
        for param in self.mert.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.mert.parameters():
            param.requires_grad = True


class MERT_AudioCAT(pl.LightningModule):
    def __init__(self, embed_dim=768, num_heads=8, num_layers=6, num_classes=2, 
                 freeze_feature_extractor=False, learning_rate=2e-5, weight_decay=0.01):
        super(MERT_AudioCAT, self).__init__()
        self.save_hyperparameters()
        self.feature_extractor = MERT(freeze_feature_extractor=freeze_feature_extractor)
        self.cross_attention_layers = nn.ModuleList([
            CrossAttention(embed_dim, num_heads) for _ in range(num_layers)
        ])
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
    def forward(self, input_values):
        features = self.feature_extractor(input_values)  
        for layer in self.cross_attention_layers:
            features = layer(features, features)
    
        features = features.mean(dim=1).unsqueeze(1) 
        encoded = self.transformer(features) 
        encoded = encoded.mean(dim=1)  
        output = self.classifier(encoded) 
        return output, encoded

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        if torch.isnan(x).any() or torch.isinf(x).any():
            return None
        
        logits, encoded = self(x)
        
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            return None
            
        loss = F.cross_entropy(logits, y)
        
        if torch.isnan(loss) or torch.isinf(loss):
            return None
            
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        self.train_f1(preds, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
            
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        if torch.isnan(x).any() or torch.isinf(x).any():
            return None
        
        logits, _ = self(x)
        
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            return None
            
        loss = F.cross_entropy(logits, y)
        
        if torch.isnan(loss) or torch.isinf(loss):
            return None
        
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits,_ = self(x)
        loss = F.cross_entropy(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        self.log('test_f1', self.test_f1, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=2, 
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }
        
    def unfreeze_feature_extractor(self):
        self.feature_extractor.unfreeze()

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
        self.is_emb = is_emb
    
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
        x = self._process_audio_batch(x)
        x = x.half()
        return self.model(x, mask)
    
    def _compute_loss_and_probs(self, y_hat: torch.Tensor, y: torch.Tensor):
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
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
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
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        if self.num_classes == 2:
            self.validation_step_outputs.append({'preds': probs, 'targets': y_true, 'binary_preds': preds})
        else:
            self.validation_step_outputs.append({'probs': probs, 'preds': preds, 'targets': y_true})

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y, mask = batch
        x = x.half()
        y_hat = self(x, mask)
        
        loss, probs, preds, y_true = self._compute_loss_and_probs(y_hat, y)
        
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        
        if self.num_classes == 2:
            self.test_step_outputs.append({'preds': probs, 'targets': y_true, 'binary_preds': preds})
        else:
            self.test_step_outputs.append({'probs': probs, 'preds': preds, 'targets': y_true})

    def on_train_epoch_start(self):
        self.training_step_outputs = []

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def on_test_epoch_start(self):
        self.test_step_outputs = []

    def _compute_binary_metrics(self, outputs, prefix):
        """Binary classification metrics computation"""
        all_preds = torch.cat([x['preds'] for x in outputs])
        all_targets = torch.cat([x['targets'] for x in outputs])
        binary_preds = torch.cat([x['binary_preds'] for x in outputs])
        
        acc = (binary_preds == all_targets).float().mean()
        
        tp = torch.sum((binary_preds == 1) & (all_targets == 1)).float()
        fp = torch.sum((binary_preds == 1) & (all_targets == 0)).float()
        tn = torch.sum((binary_preds == 0) & (all_targets == 0)).float()
        fn = torch.sum((binary_preds == 0) & (all_targets == 1)).float()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0).to(tp.device)
        recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0).to(tp.device)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0).to(tp.device)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else torch.tensor(0.0).to(tn.device)
        
        self.log(f'{prefix}_acc', acc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{prefix}_precision', precision, on_epoch=True, sync_dist=True)
        self.log(f'{prefix}_recall', recall, on_epoch=True, sync_dist=True)
        self.log(f'{prefix}_f1', f1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{prefix}_specificity', specificity, on_epoch=True, sync_dist=True)
        
        if prefix in ['val', 'test']:
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
        all_probs = torch.cat([x['probs'] for x in outputs])
        all_preds = torch.cat([x['preds'] for x in outputs])
        all_targets = torch.cat([x['targets'] for x in outputs])
        
        acc = (all_preds == all_targets).float().mean()
        self.log(f'{prefix}_acc', acc, on_epoch=True, prog_bar=True, sync_dist=True)
        
        for class_idx in range(self.num_classes):
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
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,  
            eta_min=1e-6
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss',
        }


def pad_sequence_with_mask(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings, labels = zip(*batch)
    fixed_len = 48 

    batch_size = len(embeddings)
    feat_dim = embeddings[0].shape[-1]
    
    padded = torch.zeros((batch_size, fixed_len, feat_dim)) 
    mask = torch.ones((batch_size, fixed_len), dtype=torch.bool)  
    
    for i, emb in enumerate(embeddings):
        length = emb.shape[0]
        
        if length > fixed_len:
            padded[i, :] = emb[:fixed_len]  
            mask[i, :] = False
        else:
            padded[i, :length] = emb
            mask[i, :length] = False

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
        batch_size = x.size(0)
        
        if padding_mask is not None:
            mask_expanded = (~padding_mask).float().unsqueeze(-1)
            global_avg = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            global_avg = x.mean(dim=1)
        
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
        
        self.pairwise_guided_layers = nn.ModuleList([
            PairwiseGuidedTransformer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        self.pairwise_projection = nn.Sequential(
            nn.Conv1d(1, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
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
    
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        
        self.emb_to_struct_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.struct_to_emb_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, emb_features, struct_features, padding_mask=None):

        emb_enhanced, _ = self.emb_to_struct_attn(
            emb_features, struct_features, struct_features,
            key_padding_mask=padding_mask
        )
        emb_enhanced = self.norm1(emb_features + emb_enhanced)
        
        struct_enhanced, _ = self.struct_to_emb_attn(
            struct_features, emb_features, emb_features,
            key_padding_mask=padding_mask
        )
        struct_enhanced = self.norm2(struct_features + struct_enhanced)
        
        combined = torch.cat([emb_enhanced, struct_enhanced], dim=-1)
        gate_weight = self.fusion_gate(combined)  # (batch, seq_len, d_model)
        
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
                 mode: str = 'both', 
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
        
        self.embedding_layers = nn.ModuleList([
            PairwiseGuidedTransformer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        self.pairwise_projection = nn.Sequential(
            nn.Conv1d(1, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.structure_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers // 2)  
        ])
        
        self.fusion_layers = nn.ModuleList([
            CrossModalFusionLayer(hidden_dim, num_heads)
            for _ in range(1)  
        ])
        
        self.adaptive_pooler = MultiScaleAdaptivePooler(hidden_dim, num_heads)
        
        output_dim = 1 if num_classes == 2 else num_classes
        
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
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
        
        # 5. Progressive Cross-modal Fusion 
        fused = x_emb  
        for fusion_layer in self.fusion_layers:
            fused = fusion_layer(fused, x_struct, padding_mask)

        # 6. Final pooling and classification
        pooled = self.adaptive_pooler(fused, padding_mask)
        
        pooled = pooled.half()
        return self.classification_head(pooled)
    
    import torch