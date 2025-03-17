import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from wordcloud import WordCloud

# Getting paths
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "cleaned_sentiment_dataset.csv")
clustered_path = os.path.join(script_dir, "clustered_data.csv")
model_path = os.path.join(script_dir, "bert_sentiment_model") 

# Load data
df = pd.read_csv(csv_path)
clustered_df = pd.read_csv(clustered_path)

# Prepare data
X = df['Clean_Text']
y = df['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

logreg = LogisticRegression(C=10, class_weight="balanced")
logreg.fit(X_train_tfidf, y_train)
y_pred_logreg = logreg.predict(X_test_tfidf)
logreg_accuracy = accuracy_score(y_test, y_pred_logreg)
print(f"Logistic Regression Accuracy: {logreg_accuracy:.4f}")

# BERT Evaluation
try:
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
except Exception as e:
    print(f"Error loading BERT model: {e}")
    print("Please ensure:")
    print("1. project.py has been run successfully")
    print("2. bert_sentiment_model directory exists with model files")
    exit()

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

test_dataset = SentimentDataset(X_test.tolist(), y_test.tolist(), tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=8)

def predict_bert(model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: val.to(model.device) for key, val in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
    return predictions

y_pred_bert = predict_bert(model, test_dataloader)
bert_accuracy = accuracy_score(y_test, y_pred_bert)
print(f"BERT Accuracy: {bert_accuracy:.4f}")

# --- Model Comparison Plot ---
metrics_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'BERT'],
    'Accuracy': [logreg_accuracy, bert_accuracy]
})

plt.figure(figsize=(10, 6))
sns.barplot(
    x='Model', 
    y='Accuracy', 
    data=metrics_df, 
    hue='Model', 
    palette='Blues', 
    legend=False  
)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy Score")
plt.ylim(0, 1)
plt.show()