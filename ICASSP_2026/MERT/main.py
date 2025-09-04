import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score
import wandb
import argparse
from transformers import AutoModel, AutoConfig, Wav2Vec2FeatureExtractor
from ICASSP_2026.MERT.datalib import FakeMusicCapsDataset, train_files, train_labels, val_files, val_labels
from ICASSP_2026.MERT.networks import MERTFeatureExtractor
# Set device
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Initialize wandb
wandb.init(project="mert", name=f"hpfilter_pretrain_{args.pretrain_epochs}_finetune_{args.finetune_epochs}", config=args)

# Load datasets
print("🔍 Preparing datasets...")
train_dataset = FakeMusicCapsDataset(train_files, train_labels)
val_dataset = FakeMusicCapsDataset(val_files, val_labels)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=FakeMusicCapsDataset.collate_fn)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=FakeMusicCapsDataset.collate_fn)

# Model Checkpoint Paths
pretrain_ckpt = os.path.join(args.checkpoint_dir, f"mert_pretrain_{args.pretrain_epochs}.pth")
finetune_ckpt = os.path.join(args.checkpoint_dir, f"mert_finetune_{args.finetune_epochs}.pth")

# Load Music2Vec Model for Pretraining
print("🔍 Initializing MERT model for Pretraining...")

config = AutoConfig.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
if not hasattr(config, "conv_pos_batch_norm"):
    setattr(config, "conv_pos_batch_norm", False)  

mert_model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", config=config, trust_remote_code=True).to(device)
mert_model = MERTFeatureExtractor().to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mert_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Training function
def train(model, dataloader, optimizer, criterion, device, epoch, phase="Pretrain"):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    all_preds, all_labels = [], []

    for inputs, labels in tqdm(dataloader, desc=f"{phase} Training Epoch {epoch+1}"):
        labels = labels.to(device)
        inputs = inputs.to(device)

        # inputs = inputs.float()
        # output  = model(inputs)  
        output = model(inputs)

        # Check if the output is a tensor or an object with logits
        if isinstance(output, torch.Tensor):
            logits = output
        elif hasattr(output, "logits"):
            logits = output.logits
        elif isinstance(output, (tuple, list)):
            logits = output[0]
        else:
            raise ValueError("Unexpected model output type")

        loss = criterion(logits, labels)


        # loss = criterion(output, labels) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = output.argmax(dim=1) 
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    scheduler.step()

    accuracy = total_correct / total_samples
    f1 = f1_score(all_labels, all_preds, average="binary")
    precision = precision_score(all_labels, all_preds, average="binary")
    recall = recall_score(all_labels, all_preds, average="binary", pos_label=1)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)


    wandb.log({
        f"{phase} Train Loss": total_loss / len(dataloader),
        f"{phase} Train Accuracy": accuracy,
        f"{phase} Train F1 Score": f1,
        f"{phase} Train Precision": precision,
        f"{phase} Train Recall": recall,
        f"{phase} Train Balanced Accuracy": balanced_acc,
    })

    print(f"{phase} Train Epoch {epoch+1}: Train Loss: {total_loss / len(dataloader):.4f}, "
          f"Train Acc: {accuracy:.4f}, Train F1: {f1:.4f}, Train Prec: {precision:.4f}, Train Rec: {recall:.4f}, B_ACC: {balanced_acc:.4f}")

def validate(model, dataloader, optimizer, criterion, device, epoch, phase="Validation"):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    all_preds, all_labels = [], []

    for inputs, labels in tqdm(dataloader, desc=f"{phase} Validation Epoch {epoch+1}"):
        labels = labels.to(device)
        inputs = inputs.to(device)

        output = model(inputs)

        # Check if the output is a tensor or an object with logits
        if isinstance(output, torch.Tensor):
            logits = output
        elif hasattr(output, "logits"):
            logits = output.logits
        elif isinstance(output, (tuple, list)):
            logits = output[0]
        else:
            raise ValueError("Unexpected model output type")

        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1) 
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    scheduler.step() 
    accuracy = total_correct / total_samples
    val_f1 = f1_score(all_labels, all_preds, average="weighted")
    val_precision = precision_score(all_labels, all_preds, average="binary")
    val_recall = recall_score(all_labels, all_preds, average="binary")
    val_bal_acc = balanced_accuracy_score(all_labels, all_preds)

    wandb.log({
        f"{phase} Val Loss": total_loss / len(dataloader),
        f"{phase} Val Accuracy": accuracy,
        f"{phase} Val F1 Score": val_f1,
        f"{phase} Val Precision": val_precision,
        f"{phase} Val Recall": val_recall,
        f"{phase} Val Balanced Accuracy": val_bal_acc,
    })
    print(f"{phase} Val Loss: {total_loss / len(dataloader):.4f}, "
          f"Val Acc: {accuracy:.4f}, Val F1: {val_f1:.4f}, Val Prec: {val_precision:.4f}, Val Rec: {val_recall:.4f}, Val B_ACC: {val_bal_acc:.4f}")
    return total_loss / len(dataloader), accuracy, val_f1


print("\n🔍 Step 1: Self-Supervised Pretraining on REAL Data")
# for epoch in range(args.pretrain_epochs):
#     train(mert_model, train_loader, optimizer, criterion, device, epoch, phase="Pretrain")
# torch.save(mert_model.state_dict(), pretrain_ckpt)
# print(f"\nPretraining completed! Model saved at: {pretrain_ckpt}")

# print("\n🔍 Initializing CCV Model for Fine-Tuning...")
# mert_model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", config=config, trust_remote_code=True).to(device)
# mert_model.feature_extractor.load_state_dict(torch.load(pretrain_ckpt), strict=False)

# optimizer = optim.Adam(mert_model.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

print("\n🔍 Step 2: Fine-Tuning CCV Model")
for epoch in range(args.finetune_epochs):
    train(mert_model, train_loader, optimizer, criterion, device, epoch, phase="Fine-Tune")

torch.save(mert_model.state_dict(), finetune_ckpt)
print(f"\nFine-Tuning completed! Model saved at: {finetune_ckpt}")

print("\n🔍 Step 2: Fine-Tuning MERT Model")
mert_model.load_state_dict(torch.load(pretrain_ckpt), strict=False)

optimizer = optim.Adam(mert_model.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

for epoch in range(args.finetune_epochs):
    train(mert_model, train_loader, optimizer, criterion, device, epoch, phase="Fine-Tune")

torch.save(mert_model.state_dict(), finetune_ckpt)
print(f"\nFine-Tuning completed! Model saved at: {finetune_ckpt}")