import numpy as np
import matplotlib.pyplot as plt

# Path to the training set
train_frames_path = "aftdb/learning-set/all_frames.npy"
train_labels_path = "aftdb/learning-set/all_labels.npy"

# Load the training set
train_frames = np.load(train_frames_path)
train_labels = np.load(train_labels_path)


# Function to visualize a few frames
def visualize_training_set(frames, labels, num_samples=5):
    """
    Visualize a few ECG signal frames with their labels.
    """
    plt.figure(figsize=(15, 10))

    for i in range(num_samples):
        # Randomly select a frame
        idx = np.random.randint(0, len(frames))
        signal = frames[idx]
        label = labels[idx]

        # Plot the frame
        plt.subplot(num_samples, 1, i + 1)
        plt.plot(signal, label=f"Signal {idx} (Label: {label})", color="blue")
        plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
        plt.legend(loc="upper right")
        plt.title(f"ECG Frame {idx} - Label: {label}")
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()


# Visualize 5 random samples from the training set
visualize_training_set(train_frames, train_labels, num_samples=5)
