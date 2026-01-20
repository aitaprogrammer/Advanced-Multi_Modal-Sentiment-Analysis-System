# Multi Modal Sentiment Analysis System

## 📌 Project Overview
This project develops a robust system to analyze sentiments in social media posts, specifically from Facebook and Twitter. The system utilizes **Ensemble Learning** techniques to combine multiple Machine Learning (ML) and Deep Learning (DL) models, aiming to classify sentiments into four distinct categories.

By leveraging the strengths of various algorithms, the system outperforms individual models, providing a tool useful for social media monitoring, brand management, and market research.

## 🛠 Tech Stack
* **Language:** Python
* **Frameworks:** TensorFlow
* **Techniques:** Ensemble Learning, Natural Language Processing (NLP)

## 📊 Dataset
* **Source:** Facebook and Twitter datasets sourced from Kaggle.
* **Structure:** Contains columns for text content and sentiment labels.
* **Splits:** Divided into training, validation, and test sets.

## ⚙️ Methodology

### 1. Data Preprocessing
To ensure data quality, the text data underwent rigorous cleaning:
* **Cleaning:** Removal of URLs, mentions, and hashtags using Regex; text converted to lowercase.
* **Feature Extraction (ML):** Used TF-IDF (Term Frequency-Inverse Document Frequency).
* **Preparation (DL):** Tokenization and padding applied for LSTM inputs.
* **Imbalance Handling:** Class imbalance was managed using calculated weights.

### 2. Model Architecture
The project implemented a diverse range of models before combining them via ensemble methods.

* **Traditional ML Models:**
    * Multinomial Naive Bayes
    * Logistic Regression
    * Support Vector Machine (SVM)
    * K-Nearest Neighbors (KNN)
    * XGBoost

* **Deep Learning Models:**
    * Convolutional Neural Networks (CNN)
    * Bidirectional LSTM

* **Ensemble Strategies:**
    * **Hard Voting:** Based on majority voting.
    * **Soft Voting:** Based on weighted probabilities.
    * **Stacking:** Uses Logistic Regression as a meta-learner to combine predictions.

## 📈 Results & Performance
The models were evaluated based on **Accuracy** and **Macro F1 Score**. The **Ensemble (Soft Voting)** approach achieved the highest performance.

| Model | Accuracy | Macro F1 |
| :--- | :--- | :--- |
| **Ensemble (Soft Voting)** | **0.8771** | **0.8738** |
| CNN + LSTM Ensemble | 0.8613 | 0.8581 |
| K-Nearest Neighbors | 0.8597 | 0.8568 |
| Stacking Ensemble | 0.8588 | 0.8565 |
| CNN | 0.8459 | 0.8422 |
| LSTM | 0.8230 | 0.8186 |
| Ensemble (Hard Voting) | 0.7365 | 0.7272 |
| Support Vector Machine | 0.7005 | 0.6920 |
| Logistic Regression | 0.6766 | 0.6693 |
| XGBoost | 0.6537 | 0.6267 |
| Multinomial Naive Bayes | 0.6367 | 0.6025 |

*Data sourced from project results table.*

### Training Convergence
* **LSTM:** Achieved ~0.9 training accuracy and ~0.8 validation accuracy.
* **CNN:** Achieved >0.9 training accuracy and >0.8 validation accuracy.
* *Note: Early stopping and dropout were used to mitigate overfitting risks.*

## ⚠️ Challenges & Solutions
* **Data Quality:** Noisy text containing slang and emojis was challenging; this was handled via regex cleaning.
* **Model Integration:** Stacking required combining LSTM and ML models which have different input formats.
* **Computational Cost:** LSTM training was resource-heavy, and parameter tuning required significant time.

## 🚀 Future Scope
* Enhance ensemble techniques further.
* Implement additional CNN architectures.
* Integrate pre-trained embeddings (e.g., GloVe or Word2Vec) to improve semantic understanding.
