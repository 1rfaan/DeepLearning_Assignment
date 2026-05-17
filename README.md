# Multi-Input Deep Learning for Hotel Review Score Prediction

## Introduction
This project addresses the challenge of predicting individual customer satisfaction scores based on a large-scale, heterogeneous dataset of hotel reviews. The goal was to develop an end-to-end deep learning pipeline that moves beyond coarse sentiment classification to predict **continuous reviewer scores** (regression task). By integrating unstructured text with structured metadata, the project demonstrates how deep learning can drive business intelligence in the hospitality industry.

## Methodology
The solution utilizes a multi-input architecture to process non-homogeneous data sources:

### 1. Data Preprocessing
* **Text Cleaning:** Merged positive and negative review fields and removed placeholders like "No Negative".
* **Tokenization & Padding:** Applied sequence padding to a fixed length of **200 tokens**.
* **Numerical Scaling:** Used **Min-Max Scaling** on hotel metadata (latitude, longitude, average score) for numerical stability during gradient descent.

### 2. Model Architectures
I implemented and compared three distinct deep learning architectures to evaluate their text processing capabilities:
* **Convolutional Neural Networks (CNN):** Used 1D-convolutional filters to extract local n-gram features.
* **Long Short-Term Memory (LSTM):** Designed to capture long-range sequential dependencies and sentiment flow.
* **Gated Recurrent Units (GRU):** An optimized recurrent architecture for computational efficiency and sequential memory.

### 3. Optimization Strategy
Models were optimized using the **Hyperband** algorithm via Keras Tuner. Key tuned hyperparameters included embedding dimensions, recurrent units/filters, dropout rates, and learning rates.

## Results
The experimental results proved that recurrent architectures (specifically **GRU**) outperformed convolutional models in capturing the nuances of subjective review text.

| Model (Tuned) | MSE | RMSE | R² (Explained Variance) |
| :--- | :--- | :--- | :--- |
| **CNN** | 1.1039 | 1.0507 | 0.5856 |
| **LSTM** | 1.0641 | 1.0315 | 0.6006 |
| **GRU** | **1.0447** | **1.0221** | **0.6079** |

> **Key Finding:** The **Tuned GRU model** achieved the highest accuracy, explaining **60.8%** of the variance in reviewer scores. This suggests that the simplified gating mechanism of the GRU provides an ideal balance of learning capacity and generalization for large-scale text regression.

## Dataset Note
* **File Size:** ~45.6 MB (515,000+ rows)
* **Storage:** In accordance with professional data engineering best practices, the raw dataset is omitted from this repository due to its large size. It should be downloaded from its original source and stored locally in the `/data` directory.

## How to Run
1.  **Environment Setup:**
    ```bash
    pip install tensorflow pandas scikit-learn matplotlib keras-tuner
    ```
2.  **Dataset Placement:**
    Place the raw Booking.com hotel reviews dataset inside a folder named `data/` in your project root directory.
3.  **Execution:**
    Run the preprocessing pipeline, followed by `model_tuning.py` to initiate the Hyperband search, and use `evaluate_model.py` to verify the final performance metrics.
