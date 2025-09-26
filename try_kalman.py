import numpy as np
import librosa
import soundfile as sf
import os
import csv
from tqdm import tqdm
from pykalman import KalmanFilter

# User-defined minimum volume beta-NMF function (default beta=0 is IS divergence)
def minimum_volume_beta_nmf(V, K=4, beta=0, lambda_reg=0.01, n_iter=400):
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

# Evaluation functions
def calculate_sdr(ref, est, eps=1e-10):
    return 10 * np.log10(np.clip(np.mean(ref ** 2), eps, None) / np.clip(np.mean((est - ref) ** 2), eps, None))

def calculate_sisdr(ref, est):
    eps = np.finfo(ref.dtype).eps
    ref = ref.reshape(-1, 1)
    est = est.reshape(-1, 1)
    Rss = np.dot(ref.T, ref)
    a = (eps + np.dot(ref.T, est)) / (Rss + eps)
    e_true = a * ref
    e_res = est - e_true
    return 10 * np.log10((eps + (e_true**2).sum()) / (eps + (e_res**2).sum()))

# Main processing pipeline
output_dir = "nmf_kalman"
os.makedirs(output_dir, exist_ok=True)
results = []

for idx in tqdm(range(0, 5000, 100)):
    mix_path = f"AudioSep-main/evaluation/music/mixture-{idx}.wav"
    target_path = f"AudioSep-main/evaluation/music/segment-{idx}.wav"
    if not os.path.exists(mix_path) or not os.path.exists(target_path):
        continue

    x, sr = librosa.load(mix_path, sr=16000, mono=True)
    target, _ = librosa.load(target_path, sr=sr, mono=True)
    min_len = min(len(x), len(target))
    x = x[:min_len]
    target = target[:min_len]

    n_fft = 1024
    hop_length = 512
    S = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(S), np.angle(S)

    # NMF decomposition
    W, H = minimum_volume_beta_nmf(magnitude, K=4, beta=1, lambda_reg=0.01, n_iter=1000)
    V = np.array([np.outer(W[:, k], H[k]) for k in range(4)])
    V_sum = V.sum(axis=0) + 1e-8
    M = V / V_sum

    # Kalman smoothing
    M_kalman = np.zeros_like(M)
    for k in range(4):
        for f in range(M.shape[1]):
            series = M[k, f, :]
            kf = KalmanFilter(
                transition_matrices=1.0,
                observation_matrices=1.0,
                initial_state_mean=series[0],
                initial_state_covariance=1e-4,
                observation_covariance=1e-2,
                transition_covariance=1e-5
            )
            state_means, _ = kf.smooth(series)
            M_kalman[k, f, :] = np.maximum(state_means[:, 0], 0.0)

    M_kalman_sum = M_kalman.sum(axis=0) + 1e-8
    M_kalman /= M_kalman_sum

    # Reconstruct audio
    sources = []
    for k in range(4):
        S_k = M_kalman[k] * magnitude * np.exp(1j * phase)
        y_k = librosa.istft(S_k, hop_length=hop_length, length=len(x))
        sources.append(y_k)

    # Evaluate and save best source
    best_sdr = -np.inf
    best_est = None
    for y_k in sources:
        y_k = y_k[:len(target)]
        sdr = calculate_sdr(target, y_k)
        if sdr > best_sdr:
            best_sdr = sdr
            best_est = y_k

    sisdr = calculate_sisdr(target, best_est)
    sf.write(os.path.join(output_dir, f"{idx}.wav"), best_est, sr)
    results.append([idx, best_sdr, sisdr])

# Write CSV results
avg_sdr = np.mean([r[1] for r in results])
avg_sisdr = np.mean([r[2] for r in results])
results.append(["mean", avg_sdr, avg_sisdr])
with open(os.path.join(output_dir, "results.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["idx", "SDR", "SI-SDR"])
    writer.writerows(results)

'''
# ------------------ 2️⃣ STFT -------------------
n_fft = 1024
hop_length = 512

S = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
magnitude, phase = np.abs(S), np.angle(S)

# ------------------ 3️⃣ NMF separation -------------------
K = 4 # number of sources
nmf = NMF(K, max_iter=1000,
              beta_loss='itakura-saito', solver='mu',
              l1_ratio=0.7, random_state=0)
W = nmf.fit_transform(magnitude)
H = nmf.components_

# Reconstruct each source’s energy spectrogram
V = np.array([np.outer(W[:, k], H[k]) for k in range(K)])
V_sum = V.sum(axis=0) + 1e-8  # avoid division by zero
M = V / V_sum  # initial Wiener masks

# ========================== Raw mask reconstruction ==========================
sources = []
for k in range(K):
    S_k = M[k] * magnitude * np.exp(1j * phase)
    y_k = librosa.istft(S_k, hop_length=hop_length)
    sources.append(y_k)
    sf.write(f"source_{k+1}_raw.wav", y_k, sr)
    print(f"NMF separation result saved: source_{k+1}_raw.wav")

# ------------------ Kalman filter denoising -------------------
for k, y_k in enumerate(sources):
    # Ensure 1D signal
    y_k = y_k.flatten()

    # Create Kalman filter
    kf = KalmanFilter(
        transition_matrices=1.0,
        observation_matrices=1.0
    )

    # Fit parameters
    kf = kf.em(y_k, n_iter=10)

    # Observations: NMF-separated signal
    observations = y_k
    state_means, _ = kf.filter(observations)

    y_kalman = state_means.flatten()

    sf.write(f"source_{k+1}_kalman.wav", y_kalman, sr)
    print(f"Kalman filtered result saved: source_{k+1}_kalman.wav")
'''
