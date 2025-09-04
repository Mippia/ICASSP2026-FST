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
from transformers import Wav2Vec2Processor
from datalib import FakeMusicCapsDataset, train_files, train_labels, val_files, val_labels
from networks import Music2VecClassifier, CCV

parser = argparse.ArgumentParser(description='AI Music Detection Training with Music2Vec + CCV')
parser.add_argument('--gpu', type=str, default='2', help='GPU ID')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--finetune_lr', type=float, default=1e-3, help='Fine-Tune Learning rate')
parser.add_argument('--pretrain_epochs', type=int, default=20, help='Pretraining epochs (REAL data only)')
parser.add_argument('--finetune_epochs', type=int, default=10, help='Fine-tuning epochs (REAL + FAKE data)')
parser.add_argument('--checkpoint_dir', type=str, default='', help='Checkpoint directory')
parser.add_argument('--weight_decay', type=float, default=0.001, help="Weight decay for optimizer")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

wandb.init(project="music2vec_ccv", name=f"pretrain_{args.pretrain_epochs}_finetune_{args.finetune_epochs}", config=args)

print("Preparing datasets...")
train_dataset = FakeMusicCapsDataset(train_files, train_labels)
val_dataset = FakeMusicCapsDataset(val_files, val_labels)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

pretrain_ckpt = os.path.join(args.checkpoint_dir, f"music2vec_pretrain_{args.pretrain_epochs}.pth")
finetune_ckpt = os.path.join(args.checkpoint_dir, f"music2vec_ccv_finetune_{args.finetune_epochs}.pth")

print("Initializing Music2Vec model for Pretraining...")
processor = Wav2Vec2Processor.from_pretrained("facebook/data2vec-audio-base-960h") 
model = Music2VecClassifier(freeze_feature_extractor=False).to(device)  # Pretraining에서는 freeze

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

def train(model, dataloader, optimizer, criterion, device, epoch, phase="Pretrain"):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    all_preds, all_labels = [], []

    for inputs, labels in tqdm(dataloader, desc=f"{phase} Training Epoch {epoch+1}"):
        labels = labels.to(device)
        inputs = inputs.to(device)

        logits = model(inputs)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    scheduler.step()
    accuracy = total_correct / total_samples
    f1 = f1_score(all_labels, all_preds, average="binary")
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="binary")
    recall = recall_score(all_labels, all_preds, average="binary")

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

def validate(model, dataloader, criterion, device, phase="Validation"):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"{phase}"):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.squeeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = total_correct / total_samples
    f1 = f1_score(all_labels, all_preds, average="weighted")
    val_bal_acc = balanced_accuracy_score(all_labels, all_preds)
    val_precision = precision_score(all_labels, all_preds, average="binary")
    val_recall = recall_score(all_labels, all_preds, average="binary")
    
    wandb.log({
        f"{phase} Val Loss": total_loss / len(dataloader),
        f"{phase} Val Accuracy": accuracy,
        f"{phase} Val F1 Score": f1,
        f"{phase} Val Precision": val_precision,
        f"{phase} Val Recall": val_recall,
        f"{phase} Val Balanced Accuracy": val_bal_acc,
    })
    print(f"{phase} Val Loss: {total_loss / len(dataloader):.4f}, "
          f"Val Acc: {accuracy:.4f}, Val F1: {f1:.4f}, Val Prec: {val_precision:.4f}, Val Rec: {val_recall:.4f}, Val B_ACC: {val_bal_acc:.4f}")
    return total_loss / len(dataloader), accuracy, f1

print("\nStep 1: Self-Supervised Pretraining on REAL Data")
for epoch in range(args.pretrain_epochs):
    train(model, train_loader, optimizer, criterion, device, epoch, phase="Pretrain")

torch.save(model.state_dict(), pretrain_ckpt)
print(f"\nPretraining completed! Model saved at: {pretrain_ckpt}")

print("\nInitializing Music2Vec + CCV Model for Fine-Tuning...")
model.load_state_dict(torch.load(pretrain_ckpt))

# model = CCV(embed_dim=768, num_heads=8, num_layers=6, num_classes=2, freeze_feature_extractor=True).to(device)
model = Music2VecClassifier(freeze_feature_extractor=False).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

print("\nStep 2: Fine-Tuning CCV Model using Music2Vec Features")
for epoch in range(args.finetune_epochs):
    train(model, train_loader, optimizer, criterion, device, epoch, phase="Fine-Tune")

torch.save(model.state_dict(), finetune_ckpt)
print(f"\nFine-Tuning completed! Model saved at: {finetune_ckpt}")
