<div align="center">

# 🎵 Fusion Segment Transformer: Bi-directional Attention Guided Fusion Network for AI-Generated Music Detection

**Submitted @ ICASSP 2026**  
Yumin Kim*, Seonghyeon Go*  
*MIPPIA Inc.*

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://mippia.com/)
[![Demo Page](https://img.shields.io/badge/Demo-Page-blue)](https://mippia.com/)

</div>


<img width="2252" height="660" alt="Image" src="https://github.com/user-attachments/assets/275e2422-d9dd-4940-a102-b56e35e7900d" />


---

### Abstract
With the rise of generative AI technology, anyone can now easily create and deploy AI-generated music, which has heightened the need for technical solutions to address copyright and ownership issues. While prior works have largely focused on short-audio segments, the challenge of full-audio detection, which requires modeling long-term structure and context, remains insufficiently explored. To address this, we propose Fusion Segment Transformer, which improves to the model architecture using fusion layers to combine content and structure information better. As in our previous work, we employ diverse feature extractors to provide complementary insights. In addition, we implement the Muffin Encoder, a frequency-sensitive model that is specifically designed to address high-frequency artifacts characteristic of AI-generated music. Experiments on the SONICS and AIME datasets show that our approach consistently outperforms the previous Segment Transformer and recent baselines, achieving new state-of-the-art results in full-audio AI-generated music detection.

---

## 📖 Contents
- [Installation](#installation)
- [Checkpoints](#checkpoints)
- [Inference](#inference)
- [About Fusion Segment Transformer](#about-fusion-segment-transformer)
  - [Stage-1: Feature Embedding Extractor for Short Audio Segment Detection](#-stage-1-feature-embedding-extractor-for-short-audio-segment-detection)
  - [Stage-2: Fusion Segment Transformer for Full Audio Segment Detection](#-stage-2-fusion-segment-transformer-for-full-audio-segment-detection)
  - [Quantitative Results](#quantitative-results)

---

## Installation
```bash
git clone https://github.com/mineeuk/ICASSP2026-FST.git
cd ICASSP2026-FST

pip install -r requirements.txt
```

## Checkpoints

Download pretrained checkpoints from Google Drive:

- [Stage-1 (MERT-AudioCAT)](https://drive.google.com/file/d/1frT4Mn0l6rso407Sy3eWCKbZmgwuVceN/view?usp=sharing)  
- [Stage-2 (Fusion Segment Transformer)](https://drive.google.com/file/d/1ktCDD2y91Rp07qK9olpFwK02waBajYf7/view?usp=sharing)  


## Inference

Run inference on the song you want to analyze:

```bash
python inference.py --audio ./examples/test.wav
```

## 📖 About Fusion Segment Transformer

### 🎼 Stage-1: Feature Embedding Extractor for Short Audio Segment Detection

<img width="1511" height="527" alt="Image" src="https://github.com/user-attachments/assets/b9ab6120-ed00-4587-b0f6-9d1c0b11d052" />


Stage-1: **AudioCAT framework** for short-audio segment detection.  
Feature extractors (a–e) are variably selected:  

- (a–d) use publicly available pretrained weights  
- (e) is pre-trained and further fine-tuned as the AudioCAT encoder  

We reuse the feature extractors such as **Wav2vec, Music2vec, MERT, and FX-encoder** employed in our previous work.  
Additionally, we compare them with the **Muffin Encoder**, which is specialized for multi-band frequency, to evaluate their ability in detecting short audio segments.

### 🎼 Stage-2: Fusion Segment Transformer for Full Audio Segment Detection

<img width="2252" height="660" alt="Image" src="https://github.com/user-attachments/assets/275e2422-d9dd-4940-a102-b56e35e7900d" />

Stage-2: **Fusion-enhanced Segment Transformer** architecture with dual-stream processing, cross-modal fusion mechanism, and multi-scale adaptive pooling for full-audio AIGM detection.


## 📊 Quantitative Results

### SONICS

<p align="center">
  <img src="https://github.com/user-attachments/assets/62ef2b13-2792-426e-bfe8-30da1b4e8148" alt="SONICS results table" width="70%">
</p>

Overall performance comparison of full-audio segment detection by segment-level detectors on the SONICS dataset.  
*Best*: **Bold**; *Second best*: <u>Underline</u>.

### AIME

<p align="center">
  <img src="https://github.com/user-attachments/assets/b85ebe92-0c03-4b44-9762-401d07c0edbb" alt="AIME results table" width="70%">
</p>

Overall performance comparison of full-audio segment detection by segment-level detectors on the AIME dataset.  
*Best*: **Bold**; *Second best*: <u>Underline</u>.
