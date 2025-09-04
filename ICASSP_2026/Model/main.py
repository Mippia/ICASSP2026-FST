import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import wandb
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score
from datalib import FakeMusicCapsDataset
from datalib import (
    FakeMusicCapsDataset,
    train_files, val_files, train_labels, val_labels,  
    closed_test_files, closed_test_labels, 
    open_test_files, open_test_labels,  
    preprocess_audio 
)
from datalib import preprocess_audio 
from networks import CCV
from attentionmap import visualize_attention_map 
from confusion_matrix import plot_confusion_matrix

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
'''
python3 main.py --model_name CCV --batch_size 32 --epochs 10 --loss_type ce --oversample True

audiocnn encoder - crossattn based decoder (ViT) model
'''
# Argument parsing
import argparse
parser = argparse.ArgumentParser(description='AI Music Detection Training')
parser.add_argument('--gpu', type=str, default='1', help='GPU ID')
parser.add_argument('--model_name', type=str, choices=['audiocnn', 'CCV'], default='CCV', help='Model name')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--audio_duration', type=float, default=10, help='Length of the audio slice in seconds')
parser.add_argument('--patience_counter', type=int, default=5, help='Early stopping patience')
parser.add_argument('--log_dir', type=str, default='', help='TensorBoard log directory')
parser.add_argument('--ckpt_path', type=str, default='', help='Checkpoint directory')
parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay (default: 0.0)")
parser.add_argument("--loss_type", type=str, choices=["ce", "weighted_ce", "focal"], default="ce", help="Loss function type")

parser.add_argument('--inference', type=str, help='Path to a .wav file for inference')  
parser.add_argument("--closed_test", action="store_true", help="Use Closed Test (FakeMusicCaps full dataset)")
parser.add_argument("--open_test", action="store_true", help="Use Open Set Test (SUNOCAPS_PATH included)")
parser.add_argument("--oversample", type=bool, default=True, help="Apply Oversampling to balance classes") # real data oversampling


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
wandb.init(project="", 
           name=f"{args.model_name}_lr{args.learning_rate}_ep{args.epochs}_bs{args.batch_size}", config=args)

if args.model_name == 'CCV':
    model = CCV(embed_dim=512, num_heads=8, num_layers=6, num_classes=2).cuda()
    feat_type = 'mel'
else:
    raise ValueError(f"Invalid model name: {args.model_name}")

model = model.to(device)
print(f"Using model: {args.model_name}, Parameters: {count_parameters(model)}")
print(f"weight_decay WD: {args.weight_decay}")

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

if args.loss_type == "ce":
    print("Using CrossEntropyLoss")
    criterion = nn.CrossEntropyLoss()

elif args.loss_type == "weighted_ce":
    print("Using Weighted CrossEntropyLoss")
    
    num_real = sum(1 for label in train_labels if label == 0) 
    num_fake = sum(1 for label in train_labels if label == 1)  

    total_samples = num_real + num_fake
    weight_real = total_samples / (2 * num_real)  
    weight_fake = total_samples / (2 * num_fake)  
    class_weights = torch.tensor([weight_real, weight_fake]).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

elif args.loss_type == "focal":
    print("Using Focal Loss")

    class FocalLoss(torch.nn.Module):
        def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, inputs, targets):
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)  
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  

            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss

    criterion = FocalLoss().to(device)

if not os.path.exists(args.ckpt_path):
    os.makedirs(args.ckpt_path)

train_dataset = FakeMusicCapsDataset(train_files, train_labels, feat_type=feat_type, target_duration=args.audio_duration)
val_dataset = FakeMusicCapsDataset(val_files, val_labels, feat_type=feat_type, target_duration=args.audio_duration)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

def train(model, train_loader, val_loader, optimizer, scheduler, criterion, device, args):
    writer = SummaryWriter(log_dir=args.log_dir)
    best_val_bal_acc = float('inf')
    early_stop_cnt = 0  
    log_interval = 1

    for epoch in range(args.epochs):
        print(f"\n[Epoch {epoch + 1}/{args.epochs}]")
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        all_train_preds= [] 
        all_train_labels = []
        attention_maps = []  

        train_pbar = tqdm(train_loader, desc="Train", leave=False)
        for batch_idx, (data, target) in enumerate(train_pbar):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)  
            preds = output.argmax(dim=1) 
            train_correct += (preds == target).sum().item()
            train_total += target.size(0)

            all_train_labels.extend(target.cpu().numpy())
            all_train_preds.extend(preds.cpu().numpy())

            if hasattr(model, "get_attention_maps"):  
                attention_maps.append(model.get_attention_maps())

        train_loss /= train_total
        train_acc = train_correct / train_total
        train_bal_acc = balanced_accuracy_score(all_train_labels, all_train_preds)
        train_precision = precision_score(all_train_labels, all_train_preds, average="binary")
        train_recall = recall_score(all_train_labels, all_train_preds, average="binary")
        train_f1 = f1_score(all_train_labels, all_train_preds, average="binary")

        wandb.log({
            "Train Loss": train_loss, "Train Accuracy": train_acc,
            "Train Precision": train_precision, "Train Recall": train_recall,
            "Train F1 Score": train_f1, "Train B_ACC": train_bal_acc,
        })

        print(f"Train Epoch: {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
              f"Train B_ACC: {train_bal_acc:.4f} | Train Prec: {train_precision:.3f} | "
              f"Train Rec: {train_recall:.3f} | Train F1: {train_f1:.3f}")

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_val_preds, all_val_labels = [], []
        attention_maps = []  
        val_pbar = tqdm(val_loader, desc=" Val ", leave=False)
        with torch.no_grad():
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item() * data.size(0)
                preds = output.argmax(dim=1)  
                val_correct += (preds == target).sum().item()
                val_total += target.size(0)

                all_val_labels.extend(target.cpu().numpy())
                all_val_preds.extend(preds.cpu().numpy())

                if hasattr(model, "get_attention_maps"):  
                    attention_maps.append(model.get_attention_maps())

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_bal_acc = balanced_accuracy_score(all_val_labels, all_val_preds)
        val_precision = precision_score(all_val_labels, all_val_preds, average="binary")
        val_recall = recall_score(all_val_labels, all_val_preds, average="binary")
        val_f1 = f1_score(all_val_labels, all_val_preds, average="binary")

        wandb.log({
            "Validation Loss": val_loss, "Validation Accuracy": val_acc,
            "Validation Precision": val_precision, "Validation Recall": val_recall,
            "Validation F1 Score": val_f1, "Validation B_ACC": val_bal_acc,
        })

        print(f"Val Epoch: {epoch+1} [{batch_idx * len(data)}/{len(val_loader.dataset)} "
            f"({100. * batch_idx / len(val_loader):.0f}%)]\t"
            f"Val  Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} | "
            f"Val B_ACC: {val_bal_acc:.4f} | Val Prec: {val_precision:.3f} | "
            f"Val Rec: {val_recall:.3f} | Val F1: {val_f1:.3f}")
        
        if epoch % 1 == 0 and len(attention_maps) > 0:
            print(f"Visualizing Attention Map at Epoch {epoch+1}")

            if isinstance(attention_maps[0], list):
                attn_map_numpy = np.array([t.detach().cpu().numpy() for t in attention_maps[0]])  
            elif isinstance(attention_maps[0], torch.Tensor):
                attn_map_numpy = attention_maps[0].detach().cpu().numpy()
            else:
                attn_map_numpy = np.array(attention_maps[0]) 

            print(f"Attention Map Shape: {attn_map_numpy.shape}")

            if len(attn_map_numpy) > 0:
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(attn_map_numpy[0], cmap='viridis', interpolation='nearest')
                ax.set_title(f"Attention Map - Epoch {epoch+1}")
                plt.colorbar(ax.imshow(attn_map_numpy[0], cmap='viridis'))
                plt.savefig("")
                plt.show()
            else:
                print(f"Warning: attention_maps[0] is empty! Shape={attn_map_numpy.shape}")

        if val_bal_acc < best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            early_stop_cnt = 0  
            torch.save(model.state_dict(), os.path.join(args.ckpt_path, f"best_model_{args.model_name}.pth"))
            print("Best model saved.")
        else:
            early_stop_cnt += 1  
            print(f'PATIENCE {early_stop_cnt}/{args.patience_counter}')

        if early_stop_cnt >= args.patience_counter:
            print("Early stopping triggered.")
            break

        scheduler.step()
        plot_confusion_matrix(all_val_labels, all_val_preds, classes=["REAL", "FAKE"], writer=writer, epoch=epoch)

    wandb.finish()
    writer.close()

def predict(audio_path):
    print(f"Loading model from {args.ckpt_path}/celoss_best_model_{args.model_name}.pth")
    model.load_state_dict(torch.load(os.path.join(args.ckpt_path, f"best_model_{args.model_name}.pth"), map_location=device))
    model.eval()

    input_tensor = preprocess_audio(audio_path).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        ai_music_prob = probabilities[0, 1].item()

    if ai_music_prob > 0.5:
        print(f"FAKE MUSIC {ai_music_prob:.2%})")
    else:
        print(f"REAL MUSIC {100 - ai_music_prob * 100:.2f}%")

def Test(model, test_loader, criterion, device):
    model.load_state_dict(torch.load(os.path.join(args.ckpt_path, f"best_model_{args.model_name}.pth"), map_location=device))
    model.eval()
    test_loss, test_correct, test_total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc=" Test ", leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            test_loss += loss.item() * data.size(0)
            preds = output.argmax(dim=1)
            test_correct += (preds == target).sum().item()
            test_total += target.size(0)

            all_labels.extend(target.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    test_loss /= test_total
    test_acc = test_correct / test_total
    test_bal_acc = balanced_accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds, average="binary")
    test_recall = recall_score(all_labels, all_preds, average="binary")
    test_f1 = f1_score(all_labels, all_preds, average="binary")

    print(f"\nTest Results - Loss: {test_loss:.4f} | Test Acc: {test_acc:.3f} | "
          f"Test B_ACC: {test_bal_acc:.4f} | Test Prec: {test_precision:.3f} | "
          f"Test Rec: {test_recall:.3f} | Test F1: {test_f1:.3f}")


if __name__ == "__main__":
    train(model, train_loader, val_loader, optimizer, scheduler, criterion, device, args)
    if args.closed_test:
        print("\nRunning Closed Test (FakeMusicCaps Full Dataset)...")
        test_dataset = FakeMusicCapsDataset(closed_test_files, closed_test_labels, feat_type=feat_type, target_duration=args.audio_duration)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

    elif args.open_test:
        print("\nRunning Open Set Test (FakeMusicCaps + SunoCaps)...")
        test_dataset = FakeMusicCapsDataset(open_test_files, open_test_labels, feat_type=feat_type, target_duration=args.audio_duration)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

    else:
        print("\nRunning Validation Test (FakeMusicCaps 20% Validation Set)...")
        test_dataset = FakeMusicCapsDataset(val_files, val_labels, feat_type=feat_type, target_duration=args.audio_duration)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

    print("\nEvaluating Model on Test Set...")
    Test(model, test_loader, criterion, device)

    if args.inference:
        if not os.path.exists(args.inference):
            print(f"[ERROR] No File Found: {args.inference}")
        else:
            predict(args.inference)
