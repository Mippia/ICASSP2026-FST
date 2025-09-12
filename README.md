<div align="center">

# 🎵 Fusion Segment Transformer: Bi-directional Attention Guided Fusion Network for AI-Generated Music Detection

**Submitted @ ICASSP 2026**  
Yumin Kim*, Seonghyeon Go*  
*MIPPIA Inc.*

[![Project Page](https://img.shields.io/badge/Project-Page-blue)]([https://mippia.com/](https://mippia.github.io/ICASSP2026-FST/))
[![Demo Page](https://img.shields.io/badge/Demo-Page-red)](https://huggingface.co/spaces/mippia/AI-Music-Detection-FST)

</div>


<img width="2252" height="660" alt="Image" src="https://github.com/user-attachments/assets/275e2422-d9dd-4940-a102-b56e35e7900d" />


### Abstract
With the rise of generative AI technology, anyone can now easily create and deploy AI-generated music, which has heightened the need for technical solutions to address copyright and ownership issues. While existing works have largely focused on short-audio, the challenge of full-audio detection, which requires modeling long-term structure and context, remains insufficiently explored. To address this, we propose an improved version of the Segment Transformer, termed Fusion Segment Transformer. As in our previous work, we extract content embeddings from short music segments using diverse feature extractors. Furthermore, we enhance the architecture for full-audio AI-generated music detection by introducing a Gated Fusion Layer that effectively integrates content and structural information, enabling the capture of long-term context. Experiments on the SONICS and AIME datasets show that our approach consistently outperforms the previous model and recent baselines, achieving state-of-the-art results in full-audio segment detection. 

## 📖 Contents
- [Installation](#installation)
- [Checkpoints](#checkpoints)
- [Inference](#inference)


## Installation
```bash
git clone https://github.com/mineeuk/ICASSP2026-FST.git
cd ICASSP2026-FST

pip install -r requirements.txt
```

## Checkpoints

Download pretrained checkpoints from Google Drive:

- [Stage-1 (MERT-AudioCAT)](https://drive.google.com/file/d/1frT4Mn0l6rso407Sy3eWCKbZmgwuVceN/view?usp=sharing))  
- [Stage-2 (Fusion Segment Transformer)](https://drive.google.com/file/d/1E_xPsosYWI4UjKT8XQCbZW4ILvsWnmda/view?usp=sharing))  


## Inference

Check the audio you want to analyze using the inference.py:

```bash
python inference.py --audio ./examples/test.wav
```
