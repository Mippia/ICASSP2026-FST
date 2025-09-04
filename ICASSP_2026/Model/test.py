import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix
from datalib_f import (
    FakeMusicCapsDataset,
    closed_test_files, closed_test_labels,
    open_test_files, open_test_labels,
    val_files, val_labels
)
from networks_f import CCV_Wav2Vec2
import argparse

parser = argparse.ArgumentParser(description="AI Music Detection Testing")
parser.add_argument('--gpu', type=str, default='1', help='GPU ID')
parser.add_argument('--model_name', type=str, choices=['audiocnn', 'CCV'], default='CCV_Wav2Vec2', help='Model name')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--ckpt_path', type=str, default='/data/kym/AI_Music_Detection/Code/model/tensorboard/wav2vec', help='Checkpoint directory')
parser.add_argument('--closed_test', action="store_true", help="Use Closed Test (FakeMusicCaps full dataset)")
parser.add_argument('--open_test', action="store_true", help="Use Open Set Test (SUNOCAPS_PATH included)")
parser.add_argument('--output_path', type=str, default='/data/kym/AI_Music_Detection/Code/model/test_results/w_celoss_repreprocess/wav2vec', help='Path to save test results')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    num_classes = cm.shape[0]
    tick_labels = classes[:num_classes]

    ax.set(xticks=np.arange(num_classes),
           yticks=np.arange(num_classes),
           xticklabels=tick_labels,
           yticklabels=tick_labels,
           ylabel='True label',
           xlabel='Predicted label')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

if args.model_name == 'CCV_Wav2Vec2':
    model = CCV_Wav2Vec2(embed_dim=512, num_heads=8, num_layers=6, num_classes=2).to(device)
else:
    raise ValueError(f"Invalid model name: {args.model_name}")

ckpt_file = os.path.join(args.ckpt_path, f"best_model_{args.model_name}.pth")
if not os.path.exists(ckpt_file):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

print(f"\nLoading model from {ckpt_file}")

# model.load_state_dict(torch.load(ckpt_file, map_location=device))
# 병렬
state_dict = torch.load(ckpt_file, map_location=device)
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith("module.") else k 
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
# 병렬
model.eval()

torch.cuda.empty_cache()

if args.closed_test:
    print("\nRunning Closed Test (FakeMusicCaps Full Dataset)...")
    test_dataset = FakeMusicCapsDataset(closed_test_files, closed_test_labels, feat_type="mel", target_duration=10.0)
elif args.open_test:
    print("\nRunning Open Set Test (FakeMusicCaps + SunoCaps)...")
    test_dataset = FakeMusicCapsDataset(open_test_files, open_test_labels, feat_type="mel", target_duration=10.0)
else:
    print("\nRunning Validation Test (FakeMusicCaps 20% Validation Set)...")
    test_dataset = FakeMusicCapsDataset(val_files, val_labels, feat_type="mel", target_duration=10.0)

test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

def Test(model, test_loader, device):
    model.eval()
    test_loss, test_correct, test_total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)

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

    os.makedirs(args.output_path, exist_ok=True)
    conf_matrix_path = os.path.join(args.output_path, f"confusion_matrix_{args.model_name}.png")
    plot_confusion_matrix(all_labels, all_preds, classes=["real", "generative"], output_path=conf_matrix_path)

print("\nEvaluating Model on Test Set...")
Test(model, test_loader, device)
