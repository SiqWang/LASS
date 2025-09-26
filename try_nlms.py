import numpy as np
np.complex = complex
import librosa
import soundfile as sf
import os
import csv
import padasip as pa
from tqdm import tqdm

# =================== Custom NMF ===================
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

# =================== Custom reconstruction ===================
def reconstruct_sources(V, X_complex, sig_length, W, H):
    WH = W @ H + 1e-10
    K = W.shape[1]
    estimated_sources = []
    for k in range(K):
        Vk = W[:, [k]] @ H[[k], :]
        Mk = Vk / WH
        Xk = Mk * X_complex
        xk = librosa.istft(Xk, n_fft=1024, hop_length=512, length=sig_length)
        estimated_sources.append(xk)
    return np.array(estimated_sources)

# =================== NLMS smoothing function ===================
def nlms_filter(series, mu=0.5):
    if len(series) < 2:
        return series
    x = series[:-1].reshape(-1, 1)
    d = series[1:]
    filt = pa.filters.FilterNLMS(n=1, mu=mu)
    y, _, _ = filt.run(d, x)
    y_full = np.concatenate(([series[0]], y))
    return np.maximum(y_full, 0.0)

# =================== Batch processing ===================
output_dir = "nmf_nlms"
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "results.csv")
results = []

for idx in tqdm(range(0, 5000, 50)):
    mix_path = f"AudioSep-main/evaluation/music/mixture-{idx}.wav"
    tgt_path = f"AudioSep-main/evaluation/music/segment-{idx}.wav"
    if not os.path.exists(mix_path) or not os.path.exists(tgt_path):
        continue

    mix, sr = librosa.load(mix_path, sr=16000, mono=True)
    target, _ = librosa.load(tgt_path, sr=sr, mono=True)

    S = librosa.stft(mix, n_fft=1024, hop_length=512)
    magnitude, phase = np.abs(S), np.angle(S)

    # NMF
    W, H = minimum_volume_beta_nmf(magnitude, K=4, beta=1, lambda_reg=0.01, n_iter=1000)
    V = np.array([np.outer(W[:, k], H[k]) for k in range(4)])
    V_sum = V.sum(axis=0) + 1e-8
    M = V / V_sum

    # NLMS-smoothed masks
    M_nlms = np.zeros_like(M)
    for k in range(4):
        for f in range(M.shape[1]):
            M_nlms[k, f, :] = nlms_filter(M[k, f, :], mu=0.01)
    M_nlms_sum = M_nlms.sum(axis=0) + 1e-8
    M_nlms /= M_nlms_sum

    # Reconstruct sources
    sources = []
    for k in range(4):
        S_k = M_nlms[k] * magnitude * np.exp(1j * phase)
        y_k = librosa.istft(S_k, hop_length=512, length=len(mix))
        sources.append(y_k)

    # Evaluate best source
    best_sdr, best_idx = -np.inf, -1
    for i in range(4):
        est = sources[i][:len(target)]
        score = calculate_sdr(target, est)
        if score > best_sdr:
            best_sdr = score
            best_idx = i

    best_est = sources[best_idx][:len(target)]
    best_sisdr = calculate_sisdr(target, best_est)

    sf.write(os.path.join(output_dir, f"{idx}.wav"), best_est, sr)
    results.append([idx, best_sdr, best_sisdr])

# Save CSV results
avg_sdr = np.mean([r[1] for r in results])
avg_sisdr = np.mean([r[2] for r in results])
results.append(["mean", avg_sdr, avg_sisdr])

with open(csv_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["idx", "SDR", "SI-SDR"])
    writer.writerows(results)

print(f"Batch processing complete! Best results saved in {output_dir}/, metrics saved as {csv_path}")
