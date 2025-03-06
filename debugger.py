import numpy as np
import os

# File paths
train_frames_path = "aftdb/learning-set/all_frames.npy"
train_labels_path = "aftdb/learning-set/all_labels.npy"
test_a_frames_path = "aftdb/test-set-a/test_a_frames.npy"
test_a_labels_path = "aftdb/test-set-a/test_a_labels.npy"
test_b_frames_path = "aftdb/test-set-b/test_b_frames.npy"
test_b_labels_path = "aftdb/test-set-b/test_b_labels.npy"

# Load the data
X_train = np.load(train_frames_path)
y_train = np.load(train_labels_path)
X_test_a = np.load(test_a_frames_path)
y_test_a = np.load(test_a_labels_path)
X_test_b = np.load(test_b_frames_path)
y_test_b = np.load(test_b_labels_path)

print(X_test_a[:50])