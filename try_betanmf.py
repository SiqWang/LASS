import numpy as np
import librosa
import soundfile as sf
import os
import csv
from tqdm import tqdm

# ====== Custom evaluation metrics ======
def calculate_sdr(ref, est, eps=1e-10):
    reference = ref
    noise = est - reference
    numerator = np.clip(np.mean(reference ** 2), eps, None)
    denominator = np.clip(np.mean(noise ** 2), eps, None)
    return 10. * np.log10(numerator / denominator)

def calculate_sisdr(ref, est):
    eps = np.finfo(ref.dtype).eps
    reference = ref.reshape(-1, 1)
    estimate = est.reshape(-1, 1)
    Rss = np.dot(reference.T, reference)
    a = (eps + np.dot(reference.T, estimate)) / (Rss + eps)
    e_true = a * reference
    e_res = estimate - e_true
    Sss = (e_true**2).sum()
    Snn = (e_res**2).sum()
    return 10 * np.log10((eps + Sss) / (eps + Snn))

# ====== Original decomposition function (unchanged) ======
def minimum_volume_beta_nmf(V, K=6, beta=1, lambda_reg=0.01, n_iter=400):
    np.random.seed(0)
    F, N = V.shape
    W = np.abs(np.random.randn(F, K))
    H = np.abs(np.random.randn(K, N))
    eps = 1e-10
    for _ in range(n_iter):
        WH = W @ H + eps
        if beta == 1:  # KL divergence
            W *= ((V * WH ** (beta-2)) @ H.T) / ((WH ** (beta-1)) @ H.T + lambda_reg * W @ (W.T @ W) + eps)
            H *= (W.T @ (V * WH ** (beta-2))) / (W.T @ (WH ** (beta-1)) + eps)
        elif beta == 0:  # IS divergence
            W *= ((V / WH**2) @ H.T) / ((1 / WH) @ H.T + lambda_reg * W @ (W.T @ W) + eps)
            H *= (W.T @ (V / WH**2)) / (W.T @ (1 / WH) + eps)
        else:
            raise ValueError('Only beta=1 (KL) or beta=0 (IS) are supported.')
        norm = np.linalg.norm(W, axis=0) + eps
        W /= norm
        H *= norm[:, np.newaxis]
    return W, H

# ====== Original reconstruction function (unchanged) ======
def reconstruct_sources(V, X_complex, sig_length, W, H):
    WH = W @ H + 1e-10
    K = W.shape[1]
    estimated_sources = []
    for k in range(K):
        Vk = W[:, [k]] @ H[[k], :]
        Mk = Vk / WH
        Xk = Mk * X_complex
        xk = librosa.istft(Xk, n_fft=1024, hop_length=256, length=sig_length)
        estimated_sources.append(xk)
    return np.array(estimated_sources)

# ====== Main batch processing ======
output_dir = "estimated_sources_beta0_k6"
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "results.csv")
results = []

for idx in tqdm(range(0, 5000, 50)):
    mix_path = f"AudioSep-main/evaluation/music/mixture-{idx}.wav"
    tgt_path = f"AudioSep-main/evaluation/music/segment-{idx}.wav"
    if not os.path.exists(mix_path) or not os.path.exists(tgt_path):
        continue

    mixture, sr = librosa.load(mix_path, sr=None, mono=True)
    target, _ = librosa.load(tgt_path, sr=sr, mono=True)
    Zxx = librosa.stft(mixture, n_fft=1024, hop_length=256)
    V = np.abs(Zxx)

    # NMF decomposition
    W, H = minimum_volume_beta_nmf(V, K=6, beta=0, lambda_reg=0.01, n_iter=1000)
    estimated_sources = reconstruct_sources(V, Zxx, len(mixture), W, H)

    # Evaluate best separated source
    best_sdr_score = -np.inf
    best_idx = -1
    for i in range(estimated_sources.shape[0]):
        est = estimated_sources[i][:len(target)]
        score = calculate_sdr(target, est)
        if score > best_sdr_score:
            best_sdr_score = score
            best_idx = i

    best_est = estimated_sources[best_idx][:len(target)]
    best_sisdr = calculate_sisdr(target, best_est)

    out_path = os.path.join(output_dir, f"{idx}.wav")
    sf.write(out_path, best_est, sr)
    results.append([idx, best_sdr_score, best_sisdr])

# Save CSV file
avg_sdr = np.mean([r[1] for r in results])
avg_sisdr = np.mean([r[2] for r in results])
results.append(["mean", avg_sdr, avg_sisdr])
with open(csv_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["idx", "SDR", "SI-SDR"])
    writer.writerows(results)

print(f"Batch processing complete! Best results saved in {output_dir}/, metrics saved as {csv_path}")
