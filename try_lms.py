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

# =================== LMS smoothing function ===================
def lms_filter(series, mu=0.01):
    if len(series) < 2:
        return series
    x = series[:-1].reshape(-1, 1)
    d = series[1:]
    filt = pa.filters.FilterLMS(n=1, mu=mu)
    y, _, _ = filt.run(d, x)
    y_full = np.concatenate(([series[0]], y))
    return np.maximum(y_full, 0.0)

# =================== Batch processing ===================
output_dir = "nmf_lms"
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

    W, H = minimum_volume_beta_nmf(magnitude, K=4, beta=1, lambda_reg=0.01, n_iter=1000)
    V = np.array([np.outer(W[:, k], H[k]) for k in range(4)])
    V_sum = V.sum(axis=0) + 1e-8
    M = V / V_sum

    # Apply LMS filter for smoothing
    M_lms = np.zeros_like(M)
    for k in range(4):
        for f in range(M.shape[1]):
            M_lms[k, f, :] = lms_filter(M[k, f, :], mu=0.01)
    M_lms_sum = M_lms.sum(axis=0) + 1e-8
    M_lms /= M_lms_sum

    # Reconstruct signals
    sources_lms = []
    for k in range(4):
        S_k = M_lms[k] * magnitude * np.exp(1j * phase)
        y_k = librosa.istft(S_k, hop_length=512, length=len(mix))
        sources_lms.append(y_k)

    # Evaluate best SDR source
    best_sdr, best_idx = -np.inf, -1
    for i in range(4):
        est = sources_lms[i][:len(target)]
        score = calculate_sdr(target, est)
        if score > best_sdr:
            best_sdr = score
            best_idx = i

    best_est = sources_lms[best_idx][:len(target)]
    best_sisdr = calculate_sisdr(target, best_est)
    sf.write(os.path.join(output_dir, f"{idx}.wav"), best_est, sr)
    results.append([idx, best_sdr, best_sisdr])

# Save results
avg_sdr = np.mean([r[1] for r in results])
avg_sisdr = np.mean([r[2] for r in results])
results.append(["mean", avg_sdr, avg_sisdr])

with open(csv_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["idx", "SDR", "SI-SDR"])
    writer.writerows(results)

print(f"Batch processing complete! Best results saved in {output_dir}/, statistics saved as {csv_path}")

'''
# ==================== Prepare NLMS input ====================
# ------- Choose target track index (starting from 0), e.g., select source2 ---------
target_index = 0  # choose the 2nd separated source as target

# Target signal
d = sources_raw[target_index]
# d = librosa.load("AudioSep-main/evaluation/music/segment-3100.wav", sr=16000, mono=True)[0]

# Interference signal: sum of all other tracks
x = np.sum([x for i, x in enumerate(sources_raw) if i != target_index], axis=0)

# Align lengths
min_len = min(len(d), len(x), len(mix))
d = d[:min_len]
x = x[:min_len]
mix = mix[:min_len]

# ==================== NLMS adaptive filtering ====================
filter_order = 16  # filter order
mu = 0.05          # step size

# Construct history input matrix
X = pa.input_from_history(d, filter_order)
d = mix[filter_order-1:]  # align length

# Create NLMS filter
filt = pa.filters.FilterNLMS(n=filter_order, mu=mu)

# Run filter
y, e, w = filt.run(d, X)

# Save filtered audio
sf.write(f"source{target_index+1}_nlms.wav", y, sr)
print(f"NLMS leakage-reduced audio saved: source{target_index+1}_nlms.wav")
'''
