# Imports
import os.path
import pandas as pd
import numpy as np
import re
import string
import random
import nltk
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification
nltk.download('punkt_tab')  
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from deep_translator import GoogleTranslator

# Download necessary NLTK resources
nltk.download("stopwords")
nltk.download("vader_lexicon")
nltk.download("wordnet")
nltk.download("punkt")

# Define model path and training flag
model_path = "bert_sentiment_model"
train_model = True

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load dataset
social_media_data_df = pd.read_csv("sentimentdataset.csv")
print(social_media_data_df.head())

# Check for missing values and duplicates
null_count_df = social_media_data_df.isna().sum()
print("\nSocial Media Data DataFrame missing values count: \n", null_count_df)
print("\nSocial Media Data DataFrame duplicates: ", social_media_data_df.duplicated().sum())
print("\nSocial Media Data DataFrame columns: \n", social_media_data_df.columns)

# Drop unnecessary columns
social_media_data_df = social_media_data_df.drop(columns=["Unnamed: 0.1", "Unnamed: 0", "Sentiment", "Year", "Month",
                                                          "Day", "Hour"])

# Drop duplicates rows
social_media_data_df.drop_duplicates(inplace=True)

# Check column Platform and Country
print("\nPlatform: ", social_media_data_df.Platform.value_counts())
print("\nCountry: ", social_media_data_df.Country.value_counts())

# Strip whitespace in categorical columns
social_media_data_df.Platform = social_media_data_df.Platform.str.strip()
social_media_data_df.Country = social_media_data_df.Country.str.strip()

# Convert Timestamp to datetime and extract date-related features
social_media_data_df.Timestamp = pd.to_datetime(social_media_data_df.Timestamp)
social_media_data_df["Year"] = social_media_data_df.Timestamp.dt.year
social_media_data_df["Month"] = social_media_data_df.Timestamp.dt.month
social_media_data_df["Day"] = social_media_data_df.Timestamp.dt.day

# Check dataframe
print("\n", social_media_data_df.head())

# Initialize stop words
stop_words = set(stopwords.words("english"))


# Preprocessing function to clean up the Text data
def clean_text(text: str) -> str:
    """
    Cleans input text by:
    - Converting to lowercase
    - Removing URLs, special characters, brackets, and HTML tags
    - Removing newlines, numbers, and non-ASCII characters
    - Tokenizing and removing stop words
    :param text: Words to be cleaned
    :return: A str of cleaned up words
    """
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub("<.*?>+", "", text)
    text = re.sub("\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


# Apply text cleaning
social_media_data_df["Clean_Text"] = social_media_data_df.Text.apply(clean_text)

print("Clean text: ", social_media_data_df.head())

# EDA
social_media_analysis_df = social_media_data_df.copy()

# Sentiment Analysis using VADER
sia = SentimentIntensityAnalyzer()


# Function to classify messages using VADER
def classify_message_vader(text: str) -> float:
    score = sia.polarity_scores(text)["compound"]
    if score >= 0.05:
        return 2  # Positive
    elif score <= -0.05:
        return 0  # Negative
    else:
        return 1  # Neutral


# Apply sentiment classification
social_media_analysis_df["Sentiment"] = social_media_analysis_df.Text.apply(classify_message_vader)
print("Sentiment:", social_media_analysis_df.Sentiment)


# Back-Translation Augmentation
def back_translate(text: str, src: str = "en", target: str = "de") -> str:
    try:
        translated = GoogleTranslator(source=src, target=target).translate(text)
        back_translated = GoogleTranslator(source=target, target=src).translate(translated)
        return back_translated
    except Exception:
        return text


# Apply back-translate augmentation to Neutral Sentences
neutral_texts = social_media_analysis_df[social_media_analysis_df.Sentiment == 1]["Text"].copy()
augmented_texts = set()
augmented_clean_texts = set()

for text in neutral_texts:
    back_translated_text = back_translate(text)
    if back_translated_text not in social_media_analysis_df.Text.tolist():
        cleaned_text = clean_text(back_translated_text)
        augmented_texts.add(back_translated_text)
        augmented_clean_texts.add(cleaned_text)

# Append back-translated augmented data
augmented_df = social_media_analysis_df[social_media_analysis_df.Sentiment == 1].copy()
augmented_df = augmented_df.iloc[:len(list(augmented_texts))]

augmented_df["Clean_Text"] = list(augmented_clean_texts)
augmented_df["Text"] = list(augmented_texts)
augmented_df["User"] = np.nan

social_media_analysis_df = pd.concat([social_media_analysis_df, augmented_df], ignore_index=True)


# Synonym Replacement Augmentation
def synonym_replacement(text: str, n=1) -> str:
    words = text.split()
    new_words = words.copy()
    for _ in range(n):
        word_idx = random.randint(0, len(words) - 1)
        synonyms = wordnet.synsets(words[word_idx])
        if synonyms:
            new_word = synonyms[0].lemmas()[0].name()
            new_words[word_idx] = new_word
    return " ".join(new_words)


# Apply synonym replacement on Neutral and Negative Sentences
neutral_syn_texts = social_media_analysis_df[social_media_analysis_df.Sentiment == 1]["Text"].copy()
negative_syn_texts = social_media_analysis_df[social_media_analysis_df.Sentiment == 0]["Text"].copy()

neu_syn_aug_texts = neutral_syn_texts.apply(lambda x: synonym_replacement(x, n=2))
neg_syn_aug_texts = negative_syn_texts.apply(lambda x: synonym_replacement(x, n=2))

# Append synonym augmented data
neu_syn_aug_df = social_media_analysis_df[social_media_analysis_df.Sentiment == 1].copy()
neg_syn_aug_df = social_media_analysis_df[social_media_analysis_df.Sentiment == 0].copy()

neu_syn_aug_df["Clean_Text"] = neu_syn_aug_texts.apply(clean_text)
neu_syn_aug_df["Text"] = neu_syn_aug_texts
neu_syn_aug_df["User"] = np.nan

neg_syn_aug_df["Clean_Text"] = neg_syn_aug_texts.apply(clean_text)
neg_syn_aug_df["Text"] = neg_syn_aug_texts
neg_syn_aug_df["User"] = np.nan

social_media_analysis_df = pd.concat([social_media_analysis_df, neu_syn_aug_df], ignore_index=True)
social_media_analysis_df = pd.concat([social_media_analysis_df, neg_syn_aug_df], ignore_index=True)

# Prepare dataset for ML
X = social_media_analysis_df.Clean_Text
y = social_media_analysis_df.Sentiment

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nClass distribution in y_train:\n", y_train.value_counts())
print("\nClass distribution in y_test:\n", y_test.value_counts())

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression Model with hyperparameter tuning
param_grid = {"C": [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(LogisticRegression(class_weight="balanced"), param_grid, cv=5)
grid_search.fit(X_train_tfidf, y_train)
print("Best C parameter:", grid_search.best_params_)

# Train Logistic Regression Model
logreg = LogisticRegression(C=grid_search.best_params_["C"], class_weight="balanced")
logreg.fit(X_train_tfidf, y_train)

social_media_analysis_df["Sentiment_logreg"] = logreg.predict(vectorizer.transform(social_media_analysis_df.Clean_Text))

# Evaluate Logistic Regression Model
y_pred = logreg.predict(X_test_tfidf)
logreg_accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {logreg_accuracy:.4f}")
print("Logistic Regression Classification Report: \n", classification_report(y_test, y_pred))
print("Logistic Regression Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)


# Custom Dataset Class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizerSD, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizerSD
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length,
                                  return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


if os.path.exists(model_path):
    train_model = False

if train_model:
    # Prepare Dataset
    train_dataset = SentimentDataset(X_train.tolist(), y_train.tolist(), tokenizer)
    test_dataset = SentimentDataset(X_test.tolist(), y_test.tolist(), tokenizer)

    training_args = TrainingArguments(
        output_dir="./bert_results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    trainer.train()

    preds = trainer.predict(test_dataset)
    y_preds = np.argmax(preds.predictions, axis=1)

    print("BERT:\n", classification_report(y_test, y_preds))

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Prepare Dataset
test_dataset = SentimentDataset(X_test.tolist(), y_test.tolist(), tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=8)


# Predict using BERT Model
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


social_media_analysis_df.loc[X_test.index, "Sentiment_bert"] = predict_bert(model, test_dataloader)

# Evaluate BERT Model
bert_accuracy = accuracy_score(y_test, social_media_analysis_df.loc[X_test.index, "Sentiment_bert"])
print(f"BERT Accuracy: {bert_accuracy:.4f}")
print(f"BERT Classification Report:\n",
      classification_report(y_test, social_media_analysis_df.loc[X_test.index, "Sentiment_bert"]))
print(f"BERT Confusion Matrix:\n",
      confusion_matrix(y_test, social_media_analysis_df.loc[X_test.index, "Sentiment_bert"]))

# Remove augmented data
social_media_analysis_df = social_media_analysis_df.dropna(subset=["User"])
social_media_analysis_df.to_csv("cleaned_sentiment_dataset.csv", index=False)

# Sentiment trend analysis using BERT
plt.figure(figsize=(12, 6))
sns.lineplot(data=social_media_analysis_df, x="Timestamp", y="Sentiment_bert", marker="o", errorbar=None)
plt.xlabel("Time")
plt.ylabel("Average Sentiment Score")
plt.title("Sentiment Trends Over Time (Bert)")
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Aggregate sentiment trends by month
social_media_monthly_df = social_media_analysis_df.groupby(["Year", "Month"])["Sentiment_bert"].mean().reset_index()
social_media_monthly_df["Date"] = pd.to_datetime(social_media_monthly_df[["Year", "Month"]].assign(day=1))

# Plot sentiment trends
plt.figure(figsize=(12, 6))
sns.lineplot(data=social_media_monthly_df, x="Date", y="Sentiment_bert", marker="o")
plt.xlabel("Time")
plt.ylabel("Average Sentiment Score")
plt.title("Monthly Sentiment Trends (BERT)")
plt.xticks(rotation=45)
plt.grid()
plt.show()
