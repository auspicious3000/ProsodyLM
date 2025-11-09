# 🗣️ ProsodyLM

**ProsodyLM** — a speech language model  
→ With novel **prosody tokenization** (not audio tokenization)  
→ Achieves superior prosody capabilities with **pre-training only** (no alignment)

---

### 📄 Paper
[**ProsodyLM: Uncovering the Emerging Prosody Processing Capabilities in Speech Language Models**](https://arxiv.org/abs/2507.20091)

### 🔊 Demo
[https://auspicious3000.github.io/ProsodyLM-Demo](https://auspicious3000.github.io/ProsodyLM-Demo)

### 🤗 Model Checkpoints
[https://huggingface.co/auspicious3000/prosodylm](https://huggingface.co/auspicious3000/prosodylm)

---

## ⚙️ Environment Setup

- **Python:** 3.10  
- **CUDA:** 12.8  

Create a conda environment using the provided `requirements.txt`:

```bash
conda create -n prosodylm python=3.10
conda activate prosodylm
pip install -r requirements.txt
```

Then, clone this repository and download the model checkpoints:

```bash
# Clone this repository
git clone https://github.com/auspicious3000/prosodylm.git
cd prosodylm

# Create a folder for checkpoints
mkdir hf_ckpt

# Download the Hugging Face model into this folder
git lfs install
git clone https://huggingface.co/auspicious3000/prosodylm hf_ckpt/
```

---

## 🚀 Inference

```bash
inference.ipynb
```

---

## 🧠 Training

```bash
finetune_val.sh
```

---

## 🎬 Something Fun

A short clip generated from a model fine-tuned with **ProsodyLM** — showing expressive, natural speech synthesis.

<p align="center">
  <video src="https://github.com/user-attachments/assets/62df197a-960f-4051-bb60-f34805aac23a" width="480" controls></video>
</p>


---

## 🧾 Citation

If you find this work useful, please cite:

```
@inproceedings{
qian2025prosodylm,
title={Prosody{LM}: Uncovering the Emerging Prosody Processing Capabilities in Speech Language Models},
author={Kaizhi Qian and Xulin Fan and Junrui Ni and Slava Shechtman and Mark A. Hasegawa-Johnson and Chuang Gan and Yang Zhang},
booktitle={Second Conference on Language Modeling},
year={2025},
url={https://openreview.net/forum?id=uBg8PClMUu}
}
```

---
 
**License:** CC BY-NC 4.0
