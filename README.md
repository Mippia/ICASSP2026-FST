<div align="center">

# 🎵 Fusion Segment Transformer: Bi-directional Attention Guided Fusion Network for AI-Generated Music Detection

**Submitted @ ICASSP 2026**  
Yumin Kim*, Seonghyeon Go*  
*MIPPIA Inc.*

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://mippia.com/)
[![Demo Page](https://img.shields.io/badge/Demo-Page-red)](https://mippia.com/)

</div>


<img width="2252" height="660" alt="Image" src="https://github.com/user-attachments/assets/275e2422-d9dd-4940-a102-b56e35e7900d" />


### Abstract
With the rise of generative AI technology, anyone can now easily create and deploy AI-generated music, which has heightened the need for technical solutions to address copyright and ownership issues. While prior works have largely focused on short-audio segments, the challenge of full-audio detection, which requires modeling long-term structure and context, remains insufficiently explored. To address this, we propose Fusion Segment Transformer, which improves to the model architecture using fusion layers to combine content and structure information better. As in our previous work, we employ diverse feature extractors to provide complementary insights. In addition, we implement the Muffin Encoder, a frequency-sensitive model that is specifically designed to address high-frequency artifacts characteristic of AI-generated music. Experiments on the SONICS and AIME datasets show that our approach consistently outperforms the previous Segment Transformer and recent baselines, achieving new state-of-the-art results in full-audio AI-generated music detection.


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
