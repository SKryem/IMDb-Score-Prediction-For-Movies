import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import evaluate
import os


# Dataset for fine-tuning taken from:
# https://www.kaggle.com/datasets/garcibo/metacritic-movie-15k-review-582k-dataset
# reviewDatasetClean.csv
# The following cleaning was performed on the dataset:
# 1- Remove the following columns: "title", "user", "spoilers"
# 2- Only keep reviews in English then remove the "language" column
# 3- Only keep reviews from critics then remove the "type" column
# 4- Remove reviews that overlap with our scrapped reviews.
# After these steps are performed, the only columns left are the "review" and "grade" columns,
# with no overlapping reviews between the fine-tuning set and the set we use to derive features

def cleanFineTuningDataset(filename, output_file="./data/cleaned_reviews.csv"):
    # Check if the cleaned file already exists
    if os.path.exists(output_file):
        print(f"{output_file} already exists. Skipping the cleaning process of the Fine Tuning Dataset.")
        return

    # Load the dataset
    df = pd.read_csv(filename, lineterminator='\n')

    # Columns to remove
    columns_to_remove = ["title", "user", "spoilers"]

    # Drop the unnecessary columns
    df_cleaned = df.drop(columns=columns_to_remove)

    # Step 3: Only keep reviews in English and remove the "language" column
    df_cleaned = df_cleaned[df_cleaned["language"] == "en"]
    df_cleaned = df_cleaned.drop(columns=["language"])

    # Step 3: Only keep reviews from critics and remove the "type" column
    df_cleaned = df_cleaned[df_cleaned["type"] == "reviewer"]
    df_cleaned = df_cleaned.drop(columns=["type"])

    # Save the cleaned dataset
    df_cleaned.to_csv(output_file, index=False)
    print(f"Cleaned dataset saved as {output_file}.")


def removeOverlappingReviews(finetuning_file, original_file, output_file="./data/filtered_fine_tuning_reviews.csv"):
    # Check if the filtered file already exists
    if os.path.exists(output_file):
        print(f"{output_file} already exists. Skipping the overlapping review removal process of Fine Tuning Dataset.")
        return

    # Load the original and fine-tuning datasets
    original_reviews_df = pd.read_excel(original_file)
    fine_tuning_reviews_df = pd.read_csv(finetuning_file)

    # Step 1: Trim leading and trailing whitespaces in both datasets
    original_reviews_df['Review'] = original_reviews_df['Review'].str.strip()
    fine_tuning_reviews_df['review'] = fine_tuning_reviews_df['review'].str.strip()

    # Step 2: Remove overlapping reviews
    fine_tuning_reviews_df = fine_tuning_reviews_df[
        ~fine_tuning_reviews_df['review'].isin(original_reviews_df['Review'])
    ]

    # Save the filtered fine-tuning dataset
    fine_tuning_reviews_df.to_csv(output_file, index=False)
    print(f"Filtered fine-tuning dataset saved as {output_file}.")


def create_fine_tuning_dataset():
    cleanFineTuningDataset("./data/reviewDatasetClean.csv")
    removeOverlappingReviews("./data/cleaned_reviews.csv", "./data/imdb_reviews.csv")


# Convert grades to sentiment labels
def grade_to_sentiment(grade):
    if grade <= 35:
        return 0  # Negative
    elif grade <= 65:
        return 1  # Neutral
    else:
        return 2  # Positive


# Custom Dataset class for movie reviews
class ReviewsDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_len):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def prepare_fine_tuning_data(filename):
    create_fine_tuning_dataset()
    df = pd.read_csv(filename)

    # Convert the grades to sentiment
    df['sentiment'] = df['grade'].apply(grade_to_sentiment)

    # Drop the grade column as it's no longer needed
    df = df[['review', 'sentiment']]

    return df


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def fine_tune_BERT():
    fine_tuning_data = prepare_fine_tuning_data("./data/filtered_fine_tuning_reviews.csv")

    # Split data into training and evaluation sets
    train_data, eval_data = train_test_split(fine_tuning_data, test_size=0.2, random_state=42)

    # Tokenization
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Dataset preparation
    train_dataset = ReviewsDataset(
        reviews=train_data['review'].tolist(),
        labels=train_data['sentiment'].tolist(),
        tokenizer=tokenizer,
        max_len=512
    )
    eval_dataset = ReviewsDataset(
        reviews=eval_data['review'].tolist(),
        labels=eval_data['sentiment'].tolist(),
        tokenizer=tokenizer,
        max_len=512
    )

    # Model definition
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

    # Fine-tuning using Trainer (Hugging Face)
    training_args = TrainingArguments(
        output_dir="./fine-tuning_results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=1,
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()
