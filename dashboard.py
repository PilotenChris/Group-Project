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
import torch.nn as nn
from transformers import BertModel
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from wordcloud import WordCloud

# Select GPU or CPU to run the BERT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# Custom BERT Classifier model
class CustomBERTClassifier(nn.Module):
    def __init__(self, num_labels=3):
        super(CustomBERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # Custom classification with activation functions
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels),
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.pooler_output)
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


# BERT Evaluation
try:
    model = CustomBERTClassifier(num_labels=3)
    model.load_state_dict(torch.load(model_path + "/pytorch_model.bin"))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(model_path)
except Exception as e:
    print(f"Error loading BERT model: {e}")
    print("Please ensure:")
    print("1. project.py has been run successfully")
    print("2. bert_sentiment_model directory exists with model files")
    exit()


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizerSB, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizerSB
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
    model.to(device)
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs["logits"], dim=1)
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
