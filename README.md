# 🎵 Language-Queried Audio Source Separation (LASS)

This is a lightweight and interpretable framework for **language-based audio source separation**.  
Given a mixture and a text query (e.g., *“accordion”*), it isolates the matching source using **Non-negative Matrix Factorization (NMF)** + **CLAP embeddings**.

---

## ✨ Features
- Natural language queries via CLAP (text & audio embeddings).
- Lightweight, training-free, interpretable framework.
- Optional temporal smoothing with **LMS**, **NLMS**, or **Kalman filter**.
- Evaluated with **SDR, SI-SDR, and SDRi**.

---

<!-- ## 📂 Code Overview
.
├── nmf_eval.py         # NMF + CLAP with dynamic merging
├── nmf_eval_lms.py     # NMF + CLAP + LMS smoothing
├── nmf_eval_nlms.py    # NMF + CLAP + NLMS smoothing
├── try_batanmf.py      # Baseline β-NMF (IS vs KL divergence)
├── try_lms.py          # NMF + LMS smoothing
├── try_nlms.py         # NMF + NLMS smoothing
├── try_kalman.py       # NMF + Kalman smoothing
└── README.md

--- -->

## 🧪 Experiments
- **Exp 1**: Baseline β-NMF → chose **β=1 (KL)**, **K=4**.  
- **Exp 2**: Add mask smoothing (**LMS / NLMS / Kalman**).  
- **Exp 3**: Full LASS → plain NMF, dynamic merging, NMF+filters, vs AudioSep.  

---

## 🚀 Usage
```bash
# Clone & install
git clone https://github.com/SiqWang/LASS.git
cd LASS
pip install numpy librosa soundfile padasip tqdm pykalman scikit-learn torch

# Run experiments
python try_batanmf.py      # Baseline NMF (IS (β=0) vs. KL (β=1))
python try_lms.py          # β-NMF + LMS
python try_nlms.py         # β-NMF + NLMS
python try_kalman.py       # β-NMF + Kalman
python nmf_eval.py         # NMF + CLAP + dynamic merging
python nmf_eval_lms.py     # NMF + CLAP + LMS
python nmf_eval_nlms.py    # NMF + CLAP + NLMS
```
