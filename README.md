**ECG Classification Using BPTT-Powered LSTMs** project:
## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## **ECG Classification Using BPTT-Powered LSTMs**

### **Project Overview**
This project employs a **Bidirectional Path Through Time (BPTT)** model powered by **Long Short-Term Memory (LSTM) networks** to classify ECG signals into three categories:
- **Normal**
- **Atrial Fibrillation (AF)**
- **Other**

The model was trained on preprocessed ECG data and achieved **90% accuracy** on both the validation set and test sets **A & B**. The project involved **preprocessing raw ECG signals** and segmenting them into time-series data suitable for LSTM-based classification. However, classification performance varied due to **dataset imbalance** and overlapping features.

**Dataset Source:** [Atrial Fibrillation Database (AFTDB)](https://physionet.org/content/aftdb/1.0.0/)

---

## **1. Data Preprocessing**
### **Processing Pipeline**
- Loaded ECG signals using **wfdb**.
- Applied **zero mean, unit variance normalization** to standardize data distribution.
- Segmented signals into **1-second frames** to preserve temporal dependencies.
- Annotated frames based on **QRS detection labels** for training.

### **Generated Files**
- `all_frames.npy` - Training data  
- `all_labels.npy` - Corresponding labels  
- Test sets: **Test Set A** and **Test Set B**  

**Preprocessing Script:** `Data-preprocessing.py`

---

## **2. Model Architecture**
The classification model is a **stacked LSTM network** with the following layers:

| **Layer**  | **Details** |
|------------|------------|
| **LSTM (64 units)**  | Captures temporal dependencies (return_sequences=True) |
| **LSTM (64 units)**  | Aggregates sequential information (return_sequences=False) |
| **Dropout (0.3)**  | Prevents overfitting |
| **Dense (32 units, ReLU)** | Extracts high-level features |
| **Dense (3 units, Softmax)** | Multi-class classification output |

### **Training Configuration**
- **Loss Function:** Categorical Crossentropy  
- **Optimizer:** Adam (learning rate = 0.001)  
- **Batch Size:** 64  
- **Epochs:** 40  
- **Class Weights:** Applied to address dataset imbalance  

**Training Script:** `train_predictor_Multiclass.py`

---

## **3. Training & Performance**
The model was trained using **80%** of the dataset (with a 20% validation split). Training was monitored using **accuracy and loss metrics**.

### **Key Results**
- **Validation Accuracy:** ~90%
- **Test Set A Accuracy:** ~90%
- **Test Set B Accuracy:** ~90%

### **Performance Plots**
- Training vs Validation Loss  
- Training vs Validation Accuracy  

---

## **4. Evaluation & Confusion Matrix**
The model was evaluated on both **test sets**, and confusion matrices revealed:

### **Strengths**
✔️ **High accuracy** in detecting **Atrial Fibrillation (AF)** due to its distinct ECG patterns.

### **Weaknesses**
⚠️ Struggles with the **Normal** class due to underrepresentation and waveform similarity with other classes.  
⚠️ Misclassifications occur when **arrhythmias overlap** in feature space.

---

## **5. Key Takeaways**
✅ **90% Accuracy** across validation and test sets  
✅ Effective for **Atrial Fibrillation detection**  
⚠️ **Challenges in Normal class detection** due to dataset imbalance  
⚠️ **Validation loss fluctuations** suggest overfitting risks  

---

## **6. Future Work**
To further improve performance, the following enhancements are planned:
- **Data Augmentation:** Generate additional **normal ECG signals** to balance the dataset.
- **Transformer-based Architectures:** Explore attention-based sequence modeling.
- **Class Balancing:** Implement **SMOTE** or weighted loss functions to handle imbalance.
- **Hyperparameter Tuning:** Adjust dropout rates and LSTM cell size for better generalization.
- **Multi-Stage Classification:** First detect AF vs. non-AF, then classify further.

---

## **7. Conclusion**
This project successfully demonstrates an **LSTM-based ECG classification model**, achieving **strong performance in AF detection**. However, **improving Normal class classification** remains a challenge due to dataset imbalance. Future refinements in **data augmentation, balancing techniques, and advanced architectures** will enhance the model’s real-world applicability.

---

### 🚀 **How to Use**
1. **Preprocess Data:**
   ```sh
   python Data-preprocessing.py
   ```
2. **Train the Model:**
   ```sh
   python train_predictor_Multiclass.py
   ```
3. **Evaluate Performance:**
   - View **loss & accuracy plots**.
   - Analyze **confusion matrices** for classification insights.

---
