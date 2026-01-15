# Social Media Sentiment Analysis System Using Ensemble Learning

## 📌 Project Overview
[cite_start]This project develops a robust system to analyze sentiments in social media posts, specifically from Facebook and Twitter[cite: 3, 24]. [cite_start]The system utilizes **Ensemble Learning** techniques to combine multiple Machine Learning (ML) and Deep Learning (DL) models, aiming to classify sentiments into four distinct categories[cite: 2, 23, 25].

[cite_start]By leveraging the strengths of various algorithms, the system outperforms individual models, providing a tool useful for social media monitoring, brand management, and market research[cite: 90, 116].

## 🛠 Tech Stack
* [cite_start]**Language:** Python [cite: 26]
* [cite_start]**Frameworks:** TensorFlow [cite: 27]
* [cite_start]**Techniques:** Ensemble Learning, Natural Language Processing (NLP) [cite: 2, 22]

## 📊 Dataset
* [cite_start]**Source:** Facebook and Twitter datasets sourced from Kaggle[cite: 31].
* [cite_start]**Structure:** Contains columns for text content and sentiment labels[cite: 32].
* [cite_start]**Splits:** Divided into training, validation, and test sets[cite: 31].

## ⚙️ Methodology

### 1. Data Preprocessing
To ensure data quality, the text data underwent rigorous cleaning:
* [cite_start]**Cleaning:** Removal of URLs, mentions, and hashtags using Regex; text converted to lowercase[cite: 34, 37].
* [cite_start]**Feature Extraction (ML):** Used TF-IDF (Term Frequency-Inverse Document Frequency)[cite: 37].
* [cite_start]**Preparation (DL):** Tokenization and padding applied for LSTM inputs[cite: 37].
* [cite_start]**Imbalance Handling:** Class imbalance was managed using calculated weights[cite: 100, 101].

### 2. Model Architecture
[cite_start]The project implemented a diverse range of models before combining them via ensemble methods[cite: 38].

* **Traditional ML Models:**
    * [cite_start]Multinomial Naive Bayes [cite: 72]
    * [cite_start]Logistic Regression [cite: 39]
    * [cite_start]Support Vector Machine (SVM) [cite: 40]
    * [cite_start]K-Nearest Neighbors (KNN) [cite: 40]
    * [cite_start]XGBoost [cite: 40]

* **Deep Learning Models:**
    * [cite_start]Convolutional Neural Networks (CNN) [cite: 41]
    * [cite_start]Bidirectional LSTM [cite: 41]

* **Ensemble Strategies:**
    * [cite_start]**Hard Voting:** Based on majority voting[cite: 42, 57].
    * [cite_start]**Soft Voting:** Based on weighted probabilities[cite: 42, 61].
    * [cite_start]**Stacking:** Uses Logistic Regression as a meta-learner to combine predictions[cite: 42, 62].

## 📈 Results & Performance
[cite_start]The models were evaluated based on **Accuracy** and **Macro F1 Score**[cite: 48]. [cite_start]The **Ensemble (Soft Voting)** approach achieved the highest performance[cite: 72, 113].

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

[cite_start]*Data sourced from project results table[cite: 72].*

### Training Convergence
* [cite_start]**LSTM:** Achieved ~0.9 training accuracy and ~0.8 validation accuracy[cite: 67, 69, 71].
* [cite_start]**CNN:** Achieved >0.9 training accuracy and >0.8 validation accuracy[cite: 83, 86, 87].
* [cite_start]*Note: Early stopping and dropout were used to mitigate overfitting risks[cite: 107].*

## ⚠️ Challenges & Solutions
* [cite_start]**Data Quality:** Noisy text containing slang and emojis was challenging; this was handled via regex cleaning[cite: 100].
* [cite_start]**Model Integration:** Stacking required combining LSTM and ML models which have different input formats[cite: 102, 103].
* [cite_start]**Computational Cost:** LSTM training was resource-heavy, and parameter tuning required significant time[cite: 105].

## 🚀 Future Scope
* [cite_start]Enhance ensemble techniques further[cite: 117].
* [cite_start]Implement additional CNN architectures[cite: 117].
* [cite_start]Integrate pre-trained embeddings (e.g., GloVe or Word2Vec) to improve semantic understanding[cite: 118].
