import utils
import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from BERT_fine_tuning_cuda import fine_tune_BERT


def derive_features_aux():
    if os.path.exists("./data/New_Features.csv"):
        print(f"./data/New_Features.csv already exists. Skipping the Features Generation process.")
        return
    reviews = utils.create_dict_from_data("./data/imdb_reviews.csv", 'movie_title', "Review")
    for movie_title, reviews in reviews.items():
        avg_words = utils.mean_words(reviews)
        avg_letters = utils.letters_per_word(reviews)
        numbers_freq = utils.numbers_freq(reviews)
        capitals_freq = utils.capital_letter(reviews)
        punc_freq = utils.punctuation_freq(reviews)
        data = {'movie_title': movie_title, "Avg_words": avg_words, "Avg_letters": avg_letters,
                "Numbers_freq": numbers_freq, "Capitals_freq": capitals_freq, "Punc_freq": punc_freq}
        utils.write_rows_to_csv("./data/New_Features.csv", data,
                                _headers=['movie_title', "Avg_words", "Avg_letters",
                                          "Numbers_freq", "Capitals_freq", "Punc_freq"])


# Define a function to predict sentiment
def predict_sentiment_of_review(review, model, tokenizer, max_len=512, device=torch.device('cuda')):
    encoding = tokenizer.encode_plus(
        review,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    predicted_class_id = torch.argmax(logits, dim=1).item()

    sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_mapping[predicted_class_id]


def calculate_sentiments_of_reviews(FineTuned_BERT_checkpoint_file="./fine-tuning_results/checkpoint-11084"):
    # Load the tokenizer and the fine-tuned model
    if os.path.exists("./data/movie_sentiments.csv"):
        print(f"./data/movie_sentiments.csv already exists. Skipping the Sentiment Prediction process.")
        return
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(FineTuned_BERT_checkpoint_file)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    print(device)

    # Load the IMDB reviews data
    imdb_data = pd.read_csv("./data/imdb_reviews.csv")

    # Apply the sentiment analysis to each review

    imdb_data['review_sentiment'] = imdb_data['Review'].apply(
        lambda x: predict_sentiment_of_review(x, model, tokenizer, device=device))

    # Create a new DataFrame with the required columns
    result = imdb_data[['movie_title', 'review_sentiment']]

    # Save the result to a new CSV file
    result.to_csv("./data/movie_sentiments.csv", index=False)

    print("Sentiment analysis completed and saved to ./data/movie_sentiments.csv")


def calc_avg_sentiment_of_movie():
    if os.path.exists("./data/avg_sentiment_feature.csv"):
        print(f"./data/avg_sentiment_feature.csv already exists. Skipping the Avg Sentiment Calculation process.")
        return

    sentiments = utils.create_dict_from_data("./data/movie_sentiments.csv", 'movie_title', "review_sentiment")

    for movie_title, sentiments in sentiments.items():
        avg_sent = utils.sentiment_avg(sentiments)
        data = {'movie_title': movie_title, "Avg_sentiment": avg_sent}
        utils.write_rows_to_csv("data/avg_sentiment_feature.csv", data,
                                _headers=['movie_title', "Avg_sentiment"])

    print("Average sentiment for each movie has been calculated and saved to 'avg_sentiment_feature.csv'.")


def derive_features():
    print("****************************")
    print("Deriving features")
    derive_features_aux()

    fine_tuned_BERT_checkpoint_file = utils.find_checkpoint_directory('./fine-tuning_results')  # find the fine-tuned module dir
    if fine_tuned_BERT_checkpoint_file is None:  # no fine-tuned module found
        print("No fine_tuned_BERT_checkpoint_file Found. Fine-tuning Commencing")
        fine_tune_BERT()  # this outputs a "./fine-tuning_results/checkpoint-XXXX" directory,
        # which we use to load the fine-tuned module
        fine_tuned_BERT_checkpoint_file = utils.find_checkpoint_directory('./fine-tuning_results')  # find the fine-tuned module dir
    else:
        print("Fine Tuned BERT module found. Loading...")
    # Fine-tuning the module takes alot of time, beware.
    calculate_sentiments_of_reviews(fine_tuned_BERT_checkpoint_file)
    calc_avg_sentiment_of_movie()
