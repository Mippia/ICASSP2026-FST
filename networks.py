import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, AutoConfig
from transformers import Wav2Vec2Model, Wav2Vec2Processor, Data2VecAudioModel
import torchmetrics

class cnnblock(nn.Module):
    def __init__(self, embed_dim=512):
        super(cnnblock, self).__init__()
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
        x = self.layer_norm1(x + attn_output)
        ff_output = self.feed_forward(x)
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

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x, cross_attention_input):
        self.attention_maps = []  
        for layer in self.cross_attention_layers:
            x = layer(x, cross_attention_input)

        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.mean(dim=0)
        x = self.classifier(x)
        return x

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
        time_reduced = hidden_states.mean(dim=2) 
        time_reduced = time_reduced.permute(1, 0, 2)  
        return time_reduced

    def freeze(self):
        for param in self.mert.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.mert.parameters():
            param.requires_grad = True


class MERT_AudioCNN(pl.LightningModule):
    def __init__(self, embed_dim=768, num_heads=8, num_layers=6, num_classes=2, 
                 freeze_feature_extractor=False, learning_rate=2e-5, weight_decay=0.01):
        super(MERT_AudioCNN, self).__init__()
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
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        self.train_f1(preds, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
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
            verbose=True
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


class Wav2vec_AudioCNN(pl.LightningModule):
    def __init__(self, model_name="facebook/wav2vec2-base", embed_dim=512, num_heads=8, 
                 num_layers=6, num_classes=2, freeze_feature_extractor=True,
                 learning_rate=2e-5, weight_decay=0.01):
        super(Wav2vec_AudioCNN, self).__init__()
        self.save_hyperparameters()
        
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.feature_extractor = Wav2Vec2Model.from_pretrained(model_name)
        if freeze_feature_extractor:
            self.feature_extractor.freeze_feature_encoder()  

        self.projection = nn.Linear(self.feature_extractor.config.hidden_size, embed_dim)
        self.decoder = CrossAttn_Transformer(embed_dim=embed_dim, num_heads=num_heads, 
                                            num_layers=num_layers, num_classes=num_classes)
                                            
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x, cross_attention_input=None):
        x = x.squeeze(1) 

        # Wav2Vec2 Feature Extraction
        features = self.feature_extractor(x).last_hidden_state 
        features = self.projection(features) 

        if cross_attention_input is None:
            cross_attention_input = features

        x = self.decoder(features, cross_attention_input)

        return x
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        self.train_f1(preds, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
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
            verbose=True
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

class Music2vec_AudioCNN(pl.LightningModule):
    def __init__(self, embed_dim=512, num_heads=8, num_layers=6, num_classes=2,
                 learning_rate=2e-5, weight_decay=0.01):
        super(Music2vec_AudioCNN, self).__init__()
        self.save_hyperparameters()
        
        self.feature_extractor = Music2vec(freeze_feature_extractor=True)
        self.projection = nn.Linear(self.feature_extractor.music2vec.config.hidden_size, embed_dim)
        self.decoder = CrossAttn_Transformer(embed_dim=embed_dim, num_heads=num_heads, 
                                            num_layers=num_layers, num_classes=num_classes)
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x, cross_attention_input=None):
        x = x.squeeze(1)
        features = self.feature_extractor(x)  
        features = self.projection(features)  

        if cross_attention_input is None:
            cross_attention_input = features

        x = self.decoder(features.unsqueeze(1), cross_attention_input.unsqueeze(1))
        return x
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        self.train_f1(preds, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
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
            verbose=True
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

class AudioCNN(pl.LightningModule):
    def __init__(self, embed_dim=512, num_heads=8, num_layers=6, num_classes=2,
                 learning_rate=2e-5, weight_decay=0.01):
        super(AudioCNN, self).__init__()
        self.save_hyperparameters()
        
        self.encoder = cnnblock(embed_dim=embed_dim)
        self.decoder = CrossAttn_Transformer(embed_dim=embed_dim, num_heads=num_heads, 
                                            num_layers=num_layers, num_classes=num_classes)
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x, cross_attention_input=None):
        x = self.encoder(x)  
        x = x.unsqueeze(1)
        if cross_attention_input is None:
            cross_attention_input = x
        x = self.decoder(x, cross_attention_input)
        return x
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        self.train_f1(preds, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
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
            verbose=True
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


# 필요한 보조 클래스들
class Music2vec(nn.Module):
    def __init__(self, freeze_feature_extractor=True):
        super(Music2vec, self).__init__()
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/data2vec-audio-base-960h")
        self.music2vec = Data2VecAudioModel.from_pretrained("m-a-p/music2vec-v1")
        
        if freeze_feature_extractor:
            for param in self.music2vec.parameters():
                param.requires_grad = False
        self.conv1d = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1)

    def forward(self, input_values):
        input_values = input_values.squeeze(1)
        with torch.no_grad():
            outputs = self.music2vec(input_values, output_hidden_states=True)
        hidden_states = torch.stack(outputs.hidden_states)  
        time_reduced = hidden_states.mean(dim=2)            
        time_reduced = time_reduced.permute(1, 0, 2)           
        weighted_avg = self.conv1d(time_reduced).squeeze(1)    
        return weighted_avg