import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix
from datalib import (
    FakeMusicCapsDataset,
    closed_test_files, closed_test_labels,
    open_test_files, open_test_labels,
    val_files, val_labels
)
from networks import Wav2Vec2ForFakeMusic
import tqdm
from tqdm import tqdm
import argparse
'''
python3 test.py --finetune_test --closed_test | --open_test
'''
parser = argparse.ArgumentParser(description="AI Music Detection Testing with Wav2Vec 2.0")
parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--ckpt_path', type=str, default='', help='Checkpoint directory')
parser.add_argument('--pretrain_test', action="store_true", help="Test Pretrained Wav2Vec2 Model")
parser.add_argument('--finetune_test', action="store_true", help="Test Fine-Tuned Wav2Vec2 Model")
parser.add_argument('--closed_test', action="store_true", help="Use Closed Test (FakeMusicCaps full dataset)")
parser.add_argument('--open_test', action="store_true", help="Use Open Set Test (SUNOCAPS_PATH included)")
parser.add_argument('--output_path', type=str, default='', help='Path to save test results')

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

if args.pretrain_test:
    ckpt_file = os.path.join(args.ckpt_path, "wav2vec2_pretrain_20.pth")
    print("\n🔍 Loading Pretrained Model:", ckpt_file)
    model = Wav2Vec2ForFakeMusic(num_classes=2, freeze_feature_extractor=True).to(device)

elif args.finetune_test:
    ckpt_file = os.path.join(args.ckpt_path, "wav2vec2_finetune_10.pth")
    print("\n🔍 Loading Fine-Tuned Model:", ckpt_file)
    model = Wav2Vec2ForFakeMusic(num_classes=2, freeze_feature_extractor=False).to(device)

else:
    raise ValueError("You must specify --pretrain_test or --finetune_test")

if not os.path.exists(ckpt_file):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

# model.load_state_dict(torch.load(ckpt_file, map_location=device))
# model.eval()

ckpt = torch.load(ckpt_file, map_location=device)

keys_to_remove = [key for key in ckpt.keys() if "masked_spec_embed" in key]
for key in keys_to_remove:
    print(f"Removing unexpected key: {key}")
    del ckpt[key]

try:
    model.load_state_dict(ckpt, strict=False) 
except RuntimeError as e:
    print("Model loading error:", e)
    print("Trying to load entire model...")
    model = torch.load(ckpt_file, map_location=device)  
model.to(device)
model.eval()

torch.cuda.empty_cache()

if args.closed_test:
    print("\nRunning Closed Test (FakeMusicCaps Full Dataset)...")
    test_dataset = FakeMusicCapsDataset(closed_test_files, closed_test_labels)
elif args.open_test:
    print("\nRunning Open Set Test (FakeMusicCaps + SunoCaps)...")
    test_dataset = FakeMusicCapsDataset(open_test_files, open_test_labels)
else:
    print("\nRunning Validation Test (FakeMusicCaps 20% Validation Set)...")
    test_dataset = FakeMusicCapsDataset(val_files, val_labels)

test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

def Test(model, test_loader, device, phase="Test"):
    model.eval()
    test_loss, test_correct, test_total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"{phase}"):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.squeeze(1)  # Ensure correct input shape

            output = model(inputs)
            loss = F.cross_entropy(output, labels)

            test_loss += loss.item() * inputs.size(0)
            preds = output.argmax(dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    test_loss /= test_total
    test_acc = test_correct / test_total
    test_bal_acc = balanced_accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds, average="binary")
    test_recall = recall_score(all_labels, all_preds, average="binary")
    test_f1 = f1_score(all_labels, all_preds, average="binary")

    print(f"\n{phase} Test Results - Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.3f} | "
          f"Test Balanced Acc: {test_bal_acc:.4f} | Test Precision: {test_precision:.3f} | "
          f"Test Recall: {test_recall:.3f} | Test F1: {test_f1:.3f}")

    os.makedirs(args.output_path, exist_ok=True)
    conf_matrix_path = os.path.join(args.output_path, f"confusion_matrix_{phase}_opentest.png")
    plot_confusion_matrix(all_labels, all_preds, classes=["real", "generative"], output_path=conf_matrix_path)

print("\nEvaluating Model on Test Set...")
Test(model, test_loader, device, phase="Pretrained Model" if args.pretrain_test else "Fine-Tuned Model")
