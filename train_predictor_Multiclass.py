import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


print("Raw Test-A labels:", np.unique(np.load("aftdb/test-set-a/test_a_classifications.npy")))
print("Raw Test-B labels:", np.unique(np.load("aftdb/test-set-b/test_b_classifications.npy")))



# Load the preprocessed training data
X_train = np.load("aftdb/learning-set/all_frames.npy")  # Shape: (num_frames, frame_length)
y_train = np.load("aftdb/learning-set/train_classifications.npy")  # Shape: (num_frames,)


# Load the test data
X_test_a = np.load("aftdb/test-set-a/test_a_frames.npy")
y_test_a = np.load("aftdb/test-set-a/test_a_classifications.npy")
X_test_b = np.load("aftdb/test-set-b/test_b_frames.npy")
y_test_b = np.load("aftdb/test-set-b/test_b_classifications.npy")

# One-hot encode the labels for multi-class classification
y_train = to_categorical(y_train, num_classes=3)
y_test_a = to_categorical(y_test_a, num_classes=3)
y_test_b = to_categorical(y_test_b, num_classes=3)

# Expand dimensions of the input for LSTM
X_train = np.expand_dims(X_train, axis=-1)  # Shape: (num_frames, frame_length, 1)
X_test_a = np.expand_dims(X_test_a, axis=-1)
X_test_b = np.expand_dims(X_test_b, axis=-1)


# Verify the data
print("Training data shape:", X_train.shape, "Labels shape:", y_train.shape)
print("Test-A data shape:", X_test_a.shape, "Labels shape:", y_test_a.shape)
print("Test-B data shape:", X_test_b.shape, "Labels shape:", y_test_b.shape)

# Hyperparameters
frame_length = X_train.shape[1]  # Length of each frame
batch_size = 64
epochs = 40
learning_rate = 0.001

# Define the LSTM model

from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights based on the training data
y_train_classes = np.argmax(y_train, axis=1)  # Decode one-hot labels
classes = np.array([0, 1, 2])  # Convert to numpy array
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_classes)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

print("Class Weights:", class_weights_dict)


model = Sequential([
    LSTM(64, input_shape=(frame_length, 1), return_sequences=True),  # LSTM with 64 units
    LSTM(64, input_shape=(frame_length, 1), return_sequences=False),
    Dropout(0.3),  # Dropout for regularization
    Dense(32, activation='relu'),  # Fully connected layer
    Dense(3, activation='softmax')  # Multi-class output (3 classes)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',  # For multi-class classification
              metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,  # Use 20% of training data for validation
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    class_weight= class_weights_dict
)

# Evaluate on test-set-a
y_pred_a = model.predict(X_test_a)
y_pred_a_classes = np.argmax(y_pred_a, axis=1)  # Convert one-hot to class labels
y_test_a_classes = np.argmax(y_test_a, axis=1)

# Evaluate on test-set-b
y_pred_b = model.predict(X_test_b)
y_pred_b_classes = np.argmax(y_pred_b, axis=1)  # Convert one-hot to class labels
y_test_b_classes = np.argmax(y_test_b, axis=1)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

# Extract training metrics from history
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Create a plot for loss and accuracy
plt.figure(figsize=(12, 6))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_loss, label='Training Loss', marker='o')
plt.plot(range(1, epochs + 1), val_loss, label='Validation Loss', marker='o')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_accuracy, label='Training Accuracy', marker='o')
plt.plot(range(1, epochs + 1), val_accuracy, label='Validation Accuracy', marker='o')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Save the figure for LinkedIn
plt.tight_layout()
plt.savefig("training_metrics_graph.jpg", dpi=300)
plt.show()

# Confusion Matrices
# Test-A
ConfusionMatrixDisplay.from_predictions(
    y_test_a_classes, y_pred_a_classes, display_labels=["Normal", "AF", "Other"]
)
plt.title("Confusion Matrix - Test Set A")
plt.savefig("confusion_matrix_test_a.jpg", dpi=300)
plt.show()

# Test-B
ConfusionMatrixDisplay.from_predictions(
    y_test_b_classes, y_pred_b_classes, display_labels=["Normal", "AF", "Other"]
)
plt.title("Confusion Matrix - Test Set B")
plt.savefig("confusion_matrix_test_b.jpg", dpi=300)
plt.show()

# Save the classification reports for reference
test_a_report = classification_report(y_test_a_classes, y_pred_a_classes, target_names=["Normal", "AF", "Other"])
test_b_report = classification_report(y_test_b_classes, y_pred_b_classes, target_names=["Normal", "AF", "Other"])

# Print and save reports
print("Test-A Classification Report:")
print(test_a_report)
print("Test-B Classification Report:")
print(test_b_report)

with open("classification_report_test_a.txt", "w") as f:
    f.write("Test-A Classification Report:\n")
    f.write(test_a_report)

with open("classification_report_test_b.txt", "w") as f:
    f.write("Test-B Classification Report:\n")
    f.write(test_b_report)

