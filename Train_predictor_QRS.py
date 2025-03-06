import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, f1_score

# Load the preprocessed training data
X_train = np.load("aftdb/learning-set/all_frames.npy")  # Shape: (num_frames, frame_length)
y_train = np.load("aftdb/learning-set/all_labels.npy")  # Shape: (num_frames,)

# Load the test data
X_test_a = np.load("aftdb/test-set-a/test_a_frames.npy")
y_test_a = np.load("aftdb/test-set-a/test_a_labels.npy")
X_test_b = np.load("aftdb/test-set-b/test_b_frames.npy")
y_test_b = np.load("aftdb/test-set-b/test_b_labels.npy")

# Expand dimensions to add a time step for LSTM (needed if frame_length is not multi-dimensional)
X_train = np.expand_dims(X_train, axis=-1)  # Shape: (num_frames, frame_length, 1)
X_test_a = np.expand_dims(X_test_a, axis=-1)
X_test_b = np.expand_dims(X_test_b, axis=-1)

# Hyperparameters
frame_length = X_train.shape[1]
batch_size = 32
epochs = 20
learning_rate = 0.001

# Define the LSTM model
model = Sequential([
    LSTM(64, input_shape=(frame_length, 1), return_sequences=False),  # LSTM with 64 units
    Dropout(0.3),  # Dropout for regularization
    Dense(32, activation='relu'),  # Fully connected layer
    Dense(1, activation='sigmoid')  # Binary classification output
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train,
                    validation_split=0.2,  # Use 20% of training data for validation
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)

# Evaluate on test-set-a
y_pred_a = (model.predict(X_test_a) > 0.5).astype(int)  # Convert probabilities to binary predictions
test_a_accuracy = accuracy_score(y_test_a, y_pred_a)
test_a_f1 = f1_score(y_test_a, y_pred_a)

print(f"Test-Set-A Accuracy: {test_a_accuracy:.2f}")
print(f"Test-Set-A F1 Score: {test_a_f1:.2f}")

# Evaluate on test-set-b
y_pred_b = (model.predict(X_test_b) > 0.5).astype(int)  # Convert probabilities to binary predictions
test_b_accuracy = accuracy_score(y_test_b, y_pred_b)
test_b_f1 = f1_score(y_test_b, y_pred_b)

print(f"Test-Set-B Accuracy: {test_b_accuracy:.2f}")
print(f"Test-Set-B F1 Score: {test_b_f1:.2f}")
import matplotlib.pyplot as plt

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig("TESTA_performance.jpg")

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig("TESTB_performance.jpg")
