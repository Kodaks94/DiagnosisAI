import wfdb
import os
import numpy as np

# Base directory containing all records
data_dir = "aftdb/learning-set"


# Function to process each record
def process_record(record_name, data_dir):
    # Load the signal
    record_path = os.path.join(data_dir, record_name)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'qrs')

    # Normalize the signal
    signal = record.p_signal
    normalized_signal = (signal - np.mean(signal, axis=0)) / np.std(signal, axis=0)

    # Sampling frequency
    fs = record.fs

    # Segment the signal into 1-second frames
    frame_length = int(fs)  # Number of samples in 1 second
    frames = [normalized_signal[i:i + frame_length, 0] for i in range(0, len(normalized_signal), frame_length)]

    # Label the frames using QRS annotations
    labels = []
    for i in range(len(frames)):
        start = i * frame_length
        end = start + frame_length
        qrs_in_frame = [qrs for qrs in annotation.sample if start <= qrs < end]
        labels.append(1 if qrs_in_frame else 0)  # 1 = QRS present, 0 = QRS absent

    return frames, labels


# Process all files in the dataset
all_frames = []
all_labels = []

# Loop through record names (n01 to n10, s01 to s10, t01 to t10)
for prefix in ['n', 's', 't']:
    for i in range(1, 11):  # Assuming files go from 01 to 10
        record_name = f"{prefix}{i:02}"  # Format as n01, s01, etc.
        print(f"Processing {record_name}...")

        try:
            frames, labels = process_record(record_name, data_dir)
            all_frames.extend(frames)
            all_labels.extend(labels)
        except Exception as e:
            print(f"Error processing {record_name}: {e}")

# Convert to NumPy arrays and save
all_frames = np.array(all_frames)
all_labels = np.array(all_labels)

np.save(os.path.join(data_dir, "all_frames.npy"), all_frames)
np.save(os.path.join(data_dir, "all_labels.npy"), all_labels)


# Base directory for test sets
test_a_dir = "aftdb/test-set-a"
test_b_dir = "aftdb/test-set-b"

def process_test_set(test_dir, prefix, count):
    test_frames = []
    test_labels = []

    for i in range(1, count + 1):  # Process up to `count`
        record_name = f"{prefix}{i:02}"  # e.g., a01, a02, ..., b01, b20
        print(f"Processing {record_name}...")
        try:
            frames, labels = process_record(record_name, test_dir)
            test_frames.extend(frames)
            test_labels.extend(labels)
        except Exception as e:
            print(f"Error processing {record_name}: {e}")

    return test_frames, test_labels

# Process test-set-a (a01 to a30)
test_a_frames, test_a_labels = process_test_set(test_a_dir, 'a', 30)
np.save(os.path.join(test_a_dir, "test_a_frames.npy"), np.array(test_a_frames))
np.save(os.path.join(test_a_dir, "test_a_labels.npy"), np.array(test_a_labels))

# Process test-set-b (b01 to b20)
test_b_frames, test_b_labels = process_test_set(test_b_dir, 'b', 20)
np.save(os.path.join(test_b_dir, "test_b_frames.npy"), np.array(test_b_frames))
np.save(os.path.join(test_b_dir, "test_b_labels.npy"), np.array(test_b_labels))
