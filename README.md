<div align="center">

#  MUSIC PLAGIARISM DETECTION: PROBLEM FORMULATION AND A SEGMENT-BASED SOLUTION

<h3>Submitted @ ICCASP 2026</h3>

<p>
  <b>Seonghyeon Go*</b> · <b>Yumin Kim*</b> 
</p>

<p>MIPPIA Inc.</p>

[![Project Page](https://img.shields.io/badge/Project-Website-blue)](https://mippia.github.io/ICASSP2026-FST/)
[![Demo Page](https://img.shields.io/badge/Demo-Page-red)](https://huggingface.co/spaces/mippia/AI-Music-Detection-FST)

</div>


<img width="2247" height="655" alt="Image" src="https://github.com/user-attachments/assets/2ec1b58a-0f61-40cd-969e-3061cdb7b74a" />

## Abstract
With the rise of generative AI technology, anyone can now easily create and deploy AI-generated music, which has heightened the need for technical solutions to address copyright and ownership issues. While existing works have largely focused on short-audio, the challenge of full-audio detection, which requires modeling long-term structure and context, remains insufficiently explored. To address this, we propose an improved version of the Segment Transformer, termed Fusion Segment Transformer. As in our previous work, we extract content embeddings from short music segments using diverse feature extractors. Furthermore, we enhance the architecture for full-audio AI-generated music detection by introducing a Gated Fusion Layer that effectively integrates content and structural information, enabling the capture of long-term context. Experiments on the SONICS and AIME datasets show that our approach consistently outperforms the previous model and recent baselines, achieving state-of-the-art results in full-audio segment detection. 

## 📖 Contents
- [🎵 Fusion Segment Transformer: Bi-directional Attention Guided Fusion Network for AI-Generated Music Detection](#-fusion-segment-transformer-bi-directional-attention-guided-fusion-network-for-ai-generated-music-detection)
  - [Abstract](#abstract)
  - [📖 Contents](#-contents)
  - [Installation](#installation)
  - [Requirements](#requirements)
  - [Download Checkpoints and Datasets](#download-checkpoints-and-datasets)
  - [Usage Inference](#usage-inference)
  - [License](#license)


## Installation

## Requirements
To set up their environment, please run:
```bash
pip install -r requirements.txt
```

## Download Checkpoints and Datasets
To get started, download the our pre-trained checkpoints from Google Drive:
- [Stage-1 (MERT-AudioCAT)](https://drive.google.com/file/d/1frT4Mn0l6rso407Sy3eWCKbZmgwuVceN/view?usp=sharing)
- [Stage-2 (Fusion Segment Transformer)](https://drive.google.com/file/d/1E_xPsosYWI4UjKT8XQCbZW4ILvsWnmda/view?usp=sharing)


## Usage Inference
Use the inference.py script to check if music is AI-generated or human-made. For example,

```bash
python inference.py --audio ./examples/test.wav
```

## License
Our code and demo website are licensed under a 
  <a href="https://creativecommons.org/licenses/by-nc/4.0/" 
     class="text-blue-500 hover:underline">
    Creative Commons Attribution-NonCommercial 4.0 International License
  </a>.
