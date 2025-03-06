import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


def generate_realistic_ecg(fs=500, duration=2, condition="normal", severity="mild"):
    """
    Generate a more realistic synthetic ECG waveform for normal and hypercalcemia conditions with severity levels.

    Parameters:
    - fs: Sampling frequency in Hz (default 500 Hz)
    - duration: Duration of ECG signal in seconds (default 2 sec)
    - condition: "normal" for regular ECG, "hypercalcemia" for modified ECG
    - severity: "mild", "moderate", or "severe" for hypercalcemia ECG changes.

    Returns:
    - t: Time vector
    - ecg: ECG waveform based on the selected condition and severity
    """
    t = np.linspace(0, duration, int(fs * duration))
    ecg = np.zeros_like(t)
    rr_interval = 1.0  # 1 beat per second (~60 BPM)

    for beat_start in np.arange(0, duration, rr_interval):
        beat_t = t - beat_start
        p_wave = 0.1 * np.exp(-((beat_t - 0.2) ** 2) / 0.002)
        q_wave = -0.15 * np.exp(-((beat_t - 0.3) ** 2) / 0.0005)
        r_wave = 1.0 * np.exp(-((beat_t - 0.32) ** 2) / 0.0008)
        s_wave = -0.25 * np.exp(-((beat_t - 0.34) ** 2) / 0.0006)
        t_wave = 0.2 * np.exp(-((beat_t - 0.5) ** 2) / 0.005)
        ecg += p_wave + q_wave + r_wave + s_wave + t_wave

    if condition == "hypercalcemia":
        # Severity-based QT interval shortening
        severity_factors = {"mild": 0.03, "moderate": 0.05, "severe": 0.08}
        t_shift = np.random.uniform(severity_factors[severity] - 0.01, severity_factors[severity] + 0.01)
        t_wave_index = (t % rr_interval > 0.35) & (t % rr_interval < 0.55)
        ecg[t_wave_index] = 0.15 * np.exp(-((t[t_wave_index] - 0.45 + t_shift) ** 2) / 0.005)

        # Severity-based Osborn waves (J waves)
        osborn_wave_magnitude = {"mild": 0.1, "moderate": 0.2, "severe": 0.3}
        osborn_wave = osborn_wave_magnitude[severity] * np.exp(-((t - 0.36) ** 2) / 0.0003)
        ecg += osborn_wave

        # Severity-based ST Segment Depression
        st_depression_magnitude = {"mild": 0.02, "moderate": 0.05, "severe": 0.08}
        st_segment_index = (t % rr_interval > 0.34) & (t % rr_interval < 0.45)
        ecg[st_segment_index] -= st_depression_magnitude[severity]

    return t, ecg


def generate_ecg_dataset(num_samples=1000, fs=500, duration=2, fixed_length=1000):
    """
    Generate a dataset of synthetic ECG signals for normal and hypercalcemia conditions.
    """
    data = []
    labels = []

    for _ in range(num_samples):
        condition = np.random.choice(["normal", "hypercalcemia"], p=[0.5, 0.5])
        severity = np.random.choice(["mild", "moderate", "severe"],
                                    p=[0.3, 0.4, 0.3]) if condition == "hypercalcemia" else "normal"
        t, ecg_waveform = generate_realistic_ecg(fs, duration, condition, severity)

        if len(ecg_waveform) < fixed_length:
            ecg_waveform = np.pad(ecg_waveform, (0, fixed_length - len(ecg_waveform)), mode="constant")
        elif len(ecg_waveform) > fixed_length:
            ecg_waveform = ecg_waveform[:fixed_length]

        data.append(ecg_waveform)
        labels.append(f"{condition}_{severity}")

    df = pd.DataFrame(data)
    df["label"] = labels
    return df


def low_pass_filter(ecg_signal, cutoff=40, fs=500, order=3):
    """Apply a low-pass Butterworth filter to smooth the ECG waveform."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, ecg_signal)


# Generate ECG dataset and smooth hypercalcemia ECGs
ecg_dataset = generate_ecg_dataset(num_samples=1000)
hypercalcemia_samples = ecg_dataset[ecg_dataset["label"].str.contains("hypercalcemia_severe")].sample(n=3, random_state=42)

print(ecg_dataset["label"].unique())

plt.figure(figsize=(12, 6))
for idx, row in enumerate(hypercalcemia_samples.iterrows()):
    sample_ecg = row[1][:-1].values
    t = np.linspace(0, 2, 1000)
    sample_ecg_smooth = low_pass_filter(sample_ecg, cutoff=40, fs=500, order=3)
    plt.subplot(3, 1, idx + 1)
    plt.plot(t, sample_ecg_smooth, label=f"Smoothed Hypercalcemia ECG {idx + 1}", linewidth=2, color="red")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.title(f"Hypercalcemia ECG {idx + 1} (Smoothed)")
    plt.legend()
    plt.grid()
plt.tight_layout()
plt.show()
