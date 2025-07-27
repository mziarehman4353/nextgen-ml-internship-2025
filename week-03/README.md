# Week 3 – Semi-Supervised Learning & Audio Classification

This week focuses on two diverse real-world applications of machine learning:
- **Semi-Supervised Learning** (email spam detection)
- **Supervised Audio Classification** (song genre prediction)

---

## 🔍 Task 1: Semi-Supervised Learning — Email Spam Detection

### 🎯 Objective
Explore semi-supervised learning where only a small portion of the data is labeled and the majority is unlabeled — a common real-world challenge.

### 📑 Dataset
- **Source**: SMS Spam Collection Dataset
- Loaded directly from URL:

```python
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
⚙️ Workflow
Text Preprocessing

Lowercasing, removing stopwords/punctuation

TF-IDF vectorization of messages

Simulated Setup

Labeled only 20% of the data

80% kept unlabeled for semi-supervised learning

Model Training

Used LabelSpreading and LabelPropagation from sklearn.semi_supervised

Evaluation

Accuracy, Precision, Recall, and F1-Score

✅ Outcome
Demonstrated how models can learn from sparse labeled data and still generalize well — useful in domains like medical records or email systems.

🎧 Task 2: Song Genre Classification using Audio Features
🎯 Objective
Train a supervised model to predict the genre of songs based on audio characteristics.

📑 Dataset
Source: Spotify Audio Features Dataset (from Kaggle)

Features include:

Acousticness, Danceability, Energy, Instrumentalness, Tempo, Loudness

Target: Genre

⚙️ Workflow
EDA (Exploratory Data Analysis)

Visualized relationships between features (pairplots, heatmaps, boxplots)

Preprocessing

Normalized features

Encoded genre labels

Model Training

Trained classifiers: Random Forest, Logistic Regression, Gradient Boosting, etc.

Evaluation

Metrics: Accuracy, Confusion Matrix, F1-Score

✅ Outcome
Built a working audio classification model and understood the role of engineered features in music ML applications.

👨‍💻 Submitted By
Zia Ul Rehman Zafar
Representing Universitas Muhammadiyah Surakarta (UMS) in the NextGen ML Internship 2025.