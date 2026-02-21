# 🧠 Deep NLP: BiLSTM Text Classification on Custom Dataset

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An end-to-end deep NLP pipeline for text classification using Bidirectional LSTM.**

*Covers text cleaning, tokenization, sequence modeling, evaluation, and custom inference.*

[![Open in Kaggle](https://img.shields.io/badge/Open%20in-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/abmusawir/deep-nlp-text-classification-on-custom-dataset)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [NLP Pipeline](#-nlp-pipeline)
- [Model Architecture](#-model-architecture)
- [Getting Started](#-getting-started)
- [Model Performance](#-model-performance)
- [Custom Inference](#-custom-inference)
- [Author](#-author)

---

## 🧠 Overview

This project builds a complete **deep learning NLP pipeline** to classify text responses from a custom dataset. Unlike traditional ML approaches, this project leverages a **Bidirectional LSTM (BiLSTM)** to capture contextual information from both directions of a text sequence — leading to richer and more accurate representations.

**Key highlights:**
- Full text preprocessing pipeline (cleaning, tokenization, padding)
- BiLSTM model trained with TensorFlow/Keras
- Evaluation with accuracy, precision, recall, F1-score & confusion matrix
- Predict labels on brand-new custom text inputs
- Full experiment available as a **Kaggle Notebook**

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.8+ |
| **Deep Learning** | TensorFlow, Keras (BiLSTM, Embedding, Dense, Dropout) |
| **Data Handling** | pandas, numpy |
| **ML Utilities** | scikit-learn (split, metrics) |
| **Visualization** | matplotlib, seaborn |
| **Environment** | Kaggle Notebook, Jupyter, VS Code |

---

## 📂 Dataset

| Field | Description |
|---|---|
| **File** | `Sheet_1.csv` |
| **`response_text`** | Raw input text response |
| **`class`** | Target label — `flagged` or `not_flagged` |
| **Split** | 80% Train / 20% Validation (stratified) |

> Unused columns like `Unnamed: ...` are automatically dropped during preprocessing.

---

## 🗂 Project Structure

```
deep-nlp-text-classification-bilstm/
│
├── 📓 deep_nlp_bilstm.ipynb       # Main notebook: preprocessing, training & evaluation
├── 📊 Sheet_1.csv                  # Custom dataset (text responses + labels)
├── 🤖 bilstm_model.h5              # Saved trained BiLSTM model
├── 🔤 tokenizer.pkl                # Saved Keras tokenizer
├── 🚀 predict.py                   # Script for inference on new text
├── 📋 requirements.txt             # Python dependencies
└── 📄 README.md                    # Project documentation
```

---

## ⚙️ NLP Pipeline

```
Raw Text (response_text)
        │
        ▼
┌──────────────────────────────┐
│      Text Preprocessing      │
│  • Lowercase                 │
│  • Remove URLs & @mentions   │
│  • Remove special characters │
│  • Strip extra whitespace    │
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│   Tokenization & Padding     │
│  • Vocab size: 20,000 words  │
│  • Convert text → sequences  │
│  • Pad to fixed max length   │
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│     BiLSTM Neural Network    │
│  • Embedding Layer           │
│  • Bidirectional LSTM        │
│  • Dense + Dropout           │
│  • Sigmoid / Softmax output  │
└──────────────────────────────┘
        │
        ▼
   Prediction + Confidence
   (flagged / not_flagged)
```

---

## 🏗 Model Architecture

```
_________________________________________________________________
Layer (type)             Output Shape          Param #
=================================================================
Embedding                (None, max_len, 128)  2,560,000
Bidirectional LSTM       (None, 256)           263,168
Dropout (0.5)            (None, 256)           0
Dense (64, ReLU)         (None, 64)            16,448
Dropout (0.3)            (None, 64)            0
Dense (1, Sigmoid)       (None, 1)             65
=================================================================
```

| Hyperparameter | Value |
|---|---|
| Vocabulary Size | 20,000 |
| Embedding Dim | 128 |
| LSTM Units | 128 (× 2 bidirectional) |
| Optimizer | Adam |
| Loss | Binary Crossentropy |
| Epochs | ~10 |
| Batch Size | 32 |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Musawir456/Deep-NLP-Text-Classifier-BiLSTM-.git
cd Deep-NLP-Text-Classifier-BiLSTM-
```

### 2. (Optional) Create a Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Notebook

```bash
jupyter notebook deep_nlp_bilstm.ipynb
```

### 5. Or Open Directly on Kaggle

[![Open in Kaggle](https://img.shields.io/badge/Open%20in-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/abmusawir/deep-nlp-text-classification-on-custom-dataset)

---

## 📈 Model Performance

| Metric | Score |
|---|---|
| **Accuracy** | ~XX% |
| **Precision** | ~XX% |
| **Recall** | ~XX% |
| **F1-Score** | ~XX% |

> 📝 *Update this table with your actual results after training.*

---

## 💡 Custom Inference

Run predictions on any new text:

```python
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model("bilstm_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 100  # same as training

def predict(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    prob = model.predict(padded)[0][0]
    label = "🚩 flagged" if prob >= 0.5 else "✅ not_flagged"
    print(f"Prediction : {label}")
    print(f"Confidence : {prob:.4f}")

predict("Your custom text goes here...")
```

---

## 👨‍💻 Author

<div align="center">

**Abdul Musawir**
*AI/ Machine Learning Engineer & Data Scientist*
📍 Lahore, Pakistan

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/musawir_4)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Musawir456)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/abmusawir)

</div>

---

<div align="center">

⭐ **If you found this project helpful, please give it a star!** ⭐

*Made with ❤️ by Abdul Musawir*

</div>
