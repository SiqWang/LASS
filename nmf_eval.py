# --------------------------------------------------------------------
# nmf_eval.py  —  MUSIC dataset  Auto-K NMF + CLAP evaluation
# --------------------------------------------------------------------
import gc
import os, csv, librosa, numpy as np, torch, soundfile as sf
from tqdm import tqdm
from typing import Dict
from models.clap_encoder import CLAP_Encoder           # Your CLAP implementation
from utils import calculate_sdr, calculate_sisdr       # Baseline metrics
# --------------------------------------------------------------------
# ★ 1. Auto-K NMF Separator (integrated here)
# --------------------------------------------------------------------
from sklearn.decomposition import NMF
import sys
sys.path.append('..\\AudioSep-main\\')

class AutoK_NMF_Separator:
    """
    Workflow:
      1) Perform NMF with K_max.
      2) Optionally remove columns based on energy threshold.
      3) Use CLAP similarity to select the best-matching component,
         with dynamic expansion of top-N components if similarities are close.
    """
    def __init__(self, clap_encoder: CLAP_Encoder,
                 K_max: int = 10,
                 sr_mix: int = 32_000,
                 n_fft: int = 1024, hop: int = 256,
                 nmf_iter: int = 1000,
                 energy_thr: float = 0,
                 loudness_norm: bool = False):
        self.clap = clap_encoder.eval()
        self.Kmax, self.sr = K_max, sr_mix
        self.fft, self.hop, self.n_iter = n_fft, hop, nmf_iter
        self.E_thr = energy_thr
        self.loudness_norm = loudness_norm   # Whether to RMS normalize each component

    # ------- Compute CLAP audio embedding -------
    @torch.no_grad()
    def _emb(self, wav: np.ndarray) -> torch.Tensor:
        wav_t = torch.from_numpy(wav).unsqueeze(0)            # (1, T)
        emb = self.clap.get_query_embed('audio', audio=wav_t) # (1, D)
        return torch.nn.functional.normalize(emb, dim=-1).squeeze(0)

    # ------- Optional RMS normalization -------
    @staticmethod
    def _rms_norm(w):
        rms = np.sqrt(np.mean(w**2) + 1e-9)
        return w / rms

    # ------- Main separation -------
    def separate(self, mix: np.ndarray, q_emb: torch.Tensor) -> np.ndarray:
        # STFT
        beta = 1; K = 6; lambda_reg = 0.01; n_iter = 1000
        S = librosa.stft(mix, n_fft=self.fft, hop_length=self.hop)
        mag, phs = np.abs(S) + 1e-6, np.angle(S)

        # NMF(K) + sparse regularization
        np.random.seed(0)
        F, N = mag.shape
        W = np.abs(np.random.randn(F, K))
        H = np.abs(np.random.randn(K, N))
        eps = 1e-10

        for _ in range(n_iter):
            WH = W @ H + eps

            if beta == 1:  # KL divergence
                W *= ((mag * WH ** (beta - 2)) @ H.T) / ((WH ** (beta - 1)) @ H.T + lambda_reg * W @ (W.T @ W) + eps)
                H *= (W.T @ (mag * WH ** (beta - 2))) / (W.T @ (WH ** (beta - 1)) + eps)
            elif beta == 0:  # IS divergence
                W *= ((mag / WH**2) @ H.T) / ((1 / WH) @ H.T + lambda_reg * W @ (W.T @ W) + eps)
                H *= (W.T @ (mag / WH**2)) / (W.T @ (1 / WH) + eps)
            else:
                raise ValueError('Only beta=1 (KL) or beta=0 (IS) are supported.')

            # Normalize W to avoid scaling ambiguity
            norm = np.linalg.norm(W, axis=0) + eps
            W /= norm
            H *= norm[:, np.newaxis]
            M = mag / (mag.sum(0, keepdims=True) + 1e-9)

        WH = W @ H + 1e-10
        K = W.shape[1]

        # Extract separated components
        ys, sims = [], []
        for k in range(K):
            Vk = W[:, [k]] @ H[[k], :]
            Mk = Vk / WH
            Xk = Mk * S
            y = librosa.istft(Xk, hop_length=self.hop, length=len(mix)).astype(np.float32)
            if self.loudness_norm:
                y = self._rms_norm(y)
            sf.write(f'AudioSep-main/output1/{k}.wav', y, self.sr)

            sim = torch.cosine_similarity(q_emb, self._emb(y), dim=0).item()
            ys.append(y)
            sims.append(sim)

        sims = np.array(sims)
        sorted_indices = np.argsort(-sims)  # Sort descending
        top1_idx = sorted_indices[0]
        top1_sim = sims[top1_idx]

        # Dynamically expand Top-N group if similarities are close
        dynamic_top_indices = [top1_idx]
        for idx in sorted_indices[1:]:
            if abs(sims[top1_idx] - sims[idx]) <= 0.05:
                dynamic_top_indices.append(idx)
            else:
                break  # Stop expansion when similarity gap is too large

        # Top-1 baseline
        best_sim = sims[top1_idx]
        best_wav = ys[top1_idx]

        # Try combinations of close candidates
        from itertools import combinations
        for r in range(2, len(dynamic_top_indices) + 1):
            for combo in combinations(dynamic_top_indices, r):
                merged = sum(ys[i] for i in combo)
                if self.loudness_norm:
                    merged = self._rms_norm(merged)

                merged_sim = torch.cosine_similarity(q_emb, self._emb(merged), dim=0).item()

                if merged_sim > best_sim:
                    print(f"✅ Combination {combo} improved! Merged SI={merged_sim:.4f} > Best SI={best_sim:.4f}")
                    best_sim = merged_sim
                    best_wav = merged
                else:
                    print(f"❌ Combination {combo} no improvement. Merged SI={merged_sim:.4f} <= Best SI={best_sim:.4f}")
        
        return best_wav

# --------------------------------------------------------------------
# ★ 2. Evaluator
# --------------------------------------------------------------------
class MUSIC_NMF_Evaluator:
    def __init__(self, separator: AutoK_NMF_Separator,
                 sr=32_000,
                 meta=r'AudioSep-main\evaluation\metadata\music_eval.csv',
                 wavdir=r'AudioSep-main\evaluation\music'):

        self.sep, self.sr, self.dir = separator, sr, wavdir
        with open(meta) as f:
            rdr = csv.reader(f); next(rdr)
            self.meta = [row for row in rdr]

        self.classes = ["acoustic guitar","violin","accordion","xylophone","erhu",
                        "trumpet","tuba","cello","flute","saxophone"]

    @torch.no_grad()
    def __call__(self) -> Dict[str, float]:
        sisdrs = {c: [] for c in self.classes}
        sdris  = {c: [] for c in self.classes}
        results = []

        for idx, caption, *_ in tqdm(self.meta[0:5000:100]):
            src = librosa.load(f'{self.dir}\segment-{idx}.wav', sr=self.sr, mono=True)[0]
            mix = librosa.load(f'{self.dir}\mixture-{idx}.wav', sr=self.sr, mono=True)[0]

            """
            Example synthetic mixture (reference only, not executed):
                seg1, sr = librosa.load("AudioSep-main\\evaluation\\music\\segment-1636.wav", sr=None, mono=True)
                seg2, _  = librosa.load("AudioSep-main\\evaluation\\music\\segment-2990.wav", sr=sr, mono=True)
                snr_db = 0
                p1, p2 = np.mean(seg1**2), np.mean(seg2**2)
                scale  = np.sqrt(p1 / (p2 * 10**(snr_db/10)))
                mix    = seg1 + seg2 * scale

                # Avoid clipping
                mix /= max(1.05 * abs(mix).max(), 1.0)
                tgt    = seg1.copy()
                tgt   /= max(1.05 * abs(tgt).max(), 1.0)
                caption = "trumpet"
            """

            q_emb = self.sep.clap.get_query_embed('text', text=[caption])
            q_emb = torch.nn.functional.normalize(q_emb, dim=-1).squeeze(0)

            est = self.sep.separate(mix, q_emb)

            # Save separation results
            os.makedirs('AudioSep-main\\output_betanmf_1', exist_ok=True)
            sf.write(f'AudioSep-main\\output_betanmf_1\\{caption}_{idx}.wav', est, self.sr)

            sdr0 = calculate_sdr(src, mix)
            sdr  = calculate_sdr(src, est)
            sisdrs[caption].append(calculate_sisdr(src, est))
            sdris [caption].append(sdr - sdr0)
            results.append([idx, caption, sisdrs[caption][-1], sdris[caption][-1]])

        mean_sisdr = np.mean([np.mean(sisdrs[c]) for c in self.classes])
        mean_sdri  = np.mean([np.mean(sdris[c])  for c in self.classes])
        results.append(["mean", "mean", mean_sisdr, mean_sdri])

        with open('AudioSep-main\\output_betanmf_1\\results.csv', "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["idx", "caption", "SI-SDR", "SDRi"])
            for row in results:
                writer.writerow(row)

        return dict(SI_SDR=mean_sisdr, SDRi=mean_sdri)

# --------------------------------------------------------------------
# ★ 3. Main program
# --------------------------------------------------------------------
def main():
    clap = CLAP_Encoder()                       # 32 kHz
    separator = AutoK_NMF_Separator(clap, K_max=4,
                                    loudness_norm=False)   # Toggle RMS normalization
    evaluator = MUSIC_NMF_Evaluator(separator)
    scores = evaluator()
    print("\n======  MUSIC  Auto-K NMF + CLAP  ======")
    print(f"SDRi   : {scores['SDRi']:.3f} dB")
    print(f"SI-SDR : {scores['SI_SDR']:.3f} dB")

if __name__ == "__main__":
    main()
