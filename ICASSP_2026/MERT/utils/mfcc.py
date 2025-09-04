import os
import glob
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import argparse
import wandb

class RealFakeDataset(Dataset):
    """
    audio/FakeMusicCaps/
       ├─ real/
       │   └─ MusicCaps/*.wav  (label=0)
       └─ generative/
           └─ .../*.wav        (label=1)
    """
    def __init__(self, root_dir, sr=16000, n_mels=64, target_duration=10.0):
      
        self.sr = sr
        self.n_mels = n_mels
        self.target_duration = target_duration
        self.target_samples = int(target_duration * sr)  # 10초 = 160,000 샘플

        self.file_paths = []
        self.labels = []

        # Real 데이터 (label=0)
        real_dir = os.path.join(root_dir, "real")
        real_wav_files = glob.glob(os.path.join(real_dir, "**", "*.wav"), recursive=True)
        for f in real_wav_files:
            self.file_paths.append(f)
            self.labels.append(0)

        # Generative 데이터 (label=1)
        gen_dir = os.path.join(root_dir, "generative")
        gen_wav_files = glob.glob(os.path.join(gen_dir, "**", "*.wav"), recursive=True)
        for f in gen_wav_files:
            self.file_paths.append(f)
            self.labels.append(1)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        # print(f"[DEBUG] Path: {audio_path}, Label: {label}")  # 추가

        waveform, sr = librosa.load(audio_path, sr=self.sr, mono=True)

        current_samples = waveform.shape[0]
        if current_samples > self.target_samples:
            waveform = waveform[:self.target_samples]
        elif current_samples < self.target_samples:
            stretch_factor = self.target_samples / current_samples
            waveform = librosa.effects.time_stretch(waveform, rate=stretch_factor)
            waveform = waveform[:self.target_samples]

        mfcc = librosa.feature.mfcc(
            y=waveform, sr=self.sr, n_mfcc=self.n_mels, n_fft=1024, hop_length=256
        )
        mfcc = librosa.util.normalize(mfcc)

        mfcc = np.expand_dims(mfcc, axis=0)
        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return mfcc_tensor, label_tensor


     
class AudioCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(AudioCNN, self).__init__()
        self.conv_block = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((4,4))  # 최종 -> (B,32,4,4)
    )
        self.fc_block = nn.Sequential(
            nn.Linear(32*4*4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )


    def forward(self, x):
        x = self.conv_block(x)
        # x.shape: (B,32,new_freq,new_time)

        # 1) Flatten
        B, C, H, W = x.shape  # 동적 shape
        x = x.view(B, -1)     # (B, 32*H*W)

        # 2) FC
        x = self.fc_block(x)
        return x


def my_collate_fn(batch):
    mel_list, label_list = zip(*batch) 

    max_frames = max(m.shape[2] for m in mel_list)

    padded = []
    for m in mel_list:
        diff = max_frames - m.shape[2]
        if diff > 0:
            print(f"Padding applied: Original frames = {m.shape[2]}, Target frames = {max_frames}")
            m = F.pad(m, (0, diff), mode='constant', value=0)
        padded.append(m)

        
    mel_batch = torch.stack(padded, dim=0)
    label_batch = torch.tensor(label_list, dtype=torch.long)
    return mel_batch, label_batch


class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='./ckpt/mfcc/early_stop_best_batch_{batch_size}_epochs_{epochs}_lr_{learning_rate}.pth', verbose=False):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self._save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self._save_checkpoint(val_loss, model)
            self.counter = 0

    def _save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model ...")
        torch.save(model.state_dict(), self.path)

def train(batch_size, epochs, learning_rate, root_dir="audio/FakeMusicCaps"):
    if not os.path.exists("./ckpt/mfcc/"):
        os.makedirs("./ckpt/mfcc/")

    wandb.init(
        project="AI Music Detection",
        name=f"mfcc_batch_{batch_size}_epochs_{epochs}_lr_{learning_rate}",
        config={"batch_size": batch_size, "epochs": epochs, "learning_rate": learning_rate},
    )

    dataset = RealFakeDataset(root_dir=root_dir)
    n_total = len(dataset)
    n_train = int(n_total * 0.8)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch}/{epochs}]")

        # Training
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        train_pbar = tqdm(train_loader, desc="Train", leave=False)
        for mel_batch, labels in train_pbar:
            mel_batch, labels = mel_batch.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(mel_batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * mel_batch.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_preds, all_labels = [], []
        val_pbar = tqdm(val_loader, desc=" Val ", leave=False)
        with torch.no_grad():
            for mel_batch, labels in val_pbar:
                mel_batch, labels = mel_batch.to(device), labels.to(device)
                outputs = model(mel_batch)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * mel_batch.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_precision = precision_score(all_labels, all_preds, average="macro")
        val_recall = recall_score(all_labels, all_preds, average="macro")
        val_f1 = f1_score(all_labels, all_preds, average="macro")

        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} "
              f"Precision: {val_precision:.3f} Recall: {val_recall:.3f} F1: {val_f1:.3f}")

        wandb.log({"train_loss": train_loss, "train_acc": train_acc,
                   "val_loss": val_loss, "val_acc": val_acc,
                   "val_precision": val_precision, "val_recall": val_recall, "val_f1": val_f1})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = f"./ckpt/mfcc/best_batch_{batch_size}_epochs_{epochs}_lr_{learning_rate}.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"[INFO] New best model saved: {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AI Music Detection model.")
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size for training")
    parser.add_argument('--epochs', type=int, required=True, help="Number of epochs")
    parser.add_argument('--learning_rate', type=float, required=True, help="Learning rate")
    parser.add_argument('--root_dir', type=str, default="audio/FakeMusicCaps", help="Root directory for dataset")

    args = parser.parse_args()

    train(batch_size=args.batch_size, epochs=args.epochs, learning_rate=args.learning_rate, root_dir=args.root_dir)
