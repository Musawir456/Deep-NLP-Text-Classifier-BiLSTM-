# Deep NLP: BiLSTM Text Classification on Custom Dataset

This project implements an end-to-end deep NLP pipeline to classify text responses from a custom dataset. It covers text cleaning, tokenization, sequence modeling with a Bidirectional LSTM (BiLSTM), evaluation, and testing on custom examples.

## Project Overview

- Load and explore a custom CSV dataset of text responses and class labels.
- Clean raw text (lowercasing, URL/user mention removal, special characters, extra spaces).
- Tokenize and pad sequences to a fixed length for neural network input.
- Train a BiLSTM-based deep learning model with TensorFlow/Keras.
- Evaluate performance using accuracy, precision, recall, F1-score and a confusion matrix.
- Predict classes for new custom text inputs.

## Dataset

- **Source:** Custom dataset (`Sheet_1.csv`) uploaded to Kaggle.
- **Key columns:**
  - `response_text`: The raw text response.
  - `class`: Label for each response (e.g., not_flagged vs flagged).
- Any extra unused columns (e.g., `Unnamed: ...`) are dropped during preprocessing.

## NLP Pipeline

1. **Text preprocessing**
   - Lowercasing, URL and mention removal.
   - Removing non-alphabetic characters and extra whitespace.
   - Storing cleaned text in a new `clean_text` column.

2. **Train–validation split**
   - 80/20 split with stratification on the class label.

3. **Tokenization & padding**
   - `Tokenizer` from Keras with a fixed vocabulary size (e.g., 20,000 words).
   - Convert text to integer sequences and pad them to a fixed max length.

4. **Model architecture (BiLSTM)**
   - Embedding layer to learn dense word vectors.
   - Bidirectional LSTM layer to capture context from both directions.
   - Dense layers with dropout for regularization.
   - Sigmoid output for binary classification (or softmax for multi-class).

5. **Training & evaluation**
   - Train for several epochs with `Adam` optimizer and `binary_crossentropy` loss.
   - Track training/validation accuracy and loss.
   - Evaluate with classification report and confusion matrix.

## Tools & Libraries

- Python, pandas, numpy
- scikit-learn (train–test split, metrics)
- TensorFlow / Keras (Embedding, LSTM, Bidirectional, Dense, Dropout)
- matplotlib, seaborn for visualizations

## Kaggle Notebook

The full experiment is available as a Kaggle notebook:

👉 https://www.kaggle.com/code/abmusawir/deep-nlp-text-classification-on-custom-dataset

## How to Run (locally)

1. Clone the repository:
   ```bash
   git clone https://github.com/<musawir456>/deep-nlp-text-classification-bilstm.git
   cd deep-nlp-text-classification-bilstm
