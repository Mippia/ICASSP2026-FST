import os
import torch
import torch.nn.functional as F
import torchaudio
import argparse
from datalib import preprocess_audio
from networks import Wav2Vec2ForFakeMusic

# Argument Parsing
parser = argparse.ArgumentParser(description="Wav2Vec2 AI Music Detection Inference")
parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
parser.add_argument('--model_name', type=str, choices=['Wav2Vec2ForFakeMusic'], default='Wav2Vec2ForFakeMusic', help='Model name')
parser.add_argument('--ckpt_path', type=str, default='/data/kym/AI_Music_Detection/Code/model/wav2vec/ckpt/', help='Checkpoint directory')
parser.add_argument('--model_type', type=str, choices=['pretrain', 'finetune'], required=True, help='Choose between pretrained or fine-tuned model')
parser.add_argument('--inference', type=str, required=True, help='Path to a .wav file for inference')  
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model Checkpoint
if args.model_type == 'pretrain':
    model_file = os.path.join(args.ckpt_path, "wav2vec2_pretrain_10.pth")
elif args.model_type == 'finetune':
    model_file = os.path.join(args.ckpt_path, "wav2vec2_finetune_5.pth")
else:
    raise ValueError("Invalid model type. Choose between 'pretrain' or 'finetune'.")

if not os.path.exists(model_file):
    raise FileNotFoundError(f"Model checkpoint not found: {model_file}")

if args.model_name == 'Wav2Vec2ForFakeMusic':
    model = Wav2Vec2ForFakeMusic(num_classes=2, freeze_feature_extractor=(args.model_type == 'finetune'))
else:
    raise ValueError(f"Invalid model name: {args.model_name}")

def predict(audio_path):
    print(f"\n🔍 Loading model from {model_file}")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"[ERROR] Audio file not found: {audio_path}")

    model.to(device)
    model.eval()

    input_tensor = preprocess_audio(audio_path).to(device)  
    print(f"Input shape after preprocessing: {input_tensor.shape}")  

    with torch.no_grad():
        output = model(input_tensor)  
        print(f"Raw model output (logits): {output}")

        probabilities = F.softmax(output, dim=1)
        ai_music_prob = probabilities[0, 1].item()

        print(f"Softmax Probabilities: {probabilities}")
        print(f"AI Music Probability: {ai_music_prob:.4f}")

        if ai_music_prob > 0.5:
            print(f" FAKE MUSIC DETECTED ({ai_music_prob:.2%})")
        else:
            print(f" REAL MUSIC DETECTED ({100 - ai_music_prob * 100:.2f}%)")

if __name__ == "__main__":
    predict(args.inference)
