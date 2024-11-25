from imdb import IMDb
from utils import get_movie_id, headers
from bs4 import BeautifulSoup
import re
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from utils import (
    mean_words, letters_per_word, numbers_freq, capital_letter, punctuation_freq, sentiment_avg,
    dataset_handle_categorial_data, dataset_drop_cols_with_names
)
import joblib
import os
import json
from sklearn.preprocessing import StandardScaler
import requests
from tmdbv3api import TMDb, Movie

# Initialize TMDb and Movie API
tmdb = TMDb()
tmdb.api_key = '3ada7c9fa8e3ca3786fc94519c754db5'
movie_api = Movie()


def initialize_bert_model():
    """Load the fine-tuned BERT model and tokenizer for sentiment analysis."""
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", clean_up_tokenization_spaces=True)
    model = BertForSequenceClassification.from_pretrained("./fine-tuning_results/checkpoint-11084")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    return model, tokenizer, device


def prepare_data_for_interactive():
    dataset = pd.read_csv("./data/filtered_expanded_movie_metadata.csv")
    dataset_drop_cols_with_names(dataset)
    original_features = list(dataset)  # before handling categorial data
    dataset = dataset_handle_categorial_data(dataset)
    median_values = load_median_values(dataset)
    # Load scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(dataset.drop(columns=["imdb_score"]))  # Fit scaler on the features
    y_train = dataset["imdb_score"]

    # Define the feature order
    feature_order = list(dataset)
    return dataset, original_features, median_values, scaler, X_train_scaled, y_train, feature_order


def get_imdb_link(movie_name):
    ia = IMDb()
    movies = ia.search_movie(movie_name)

    if movies:
        # Check if "movie_name" was found, ia.search_movie may return the link of a movie
        # with a similar name to the input, if an exact match is not found
        if movies[0]['title'].lower() == movie_name.lower():
            imdb_id = movies[0].movieID
            return f"https://www.imdb.com/title/tt{imdb_id}/"
        else:
            return None  # Return None if the title doesn't match
    else:
        return None  # Return None if no movies were found


def fetch_movie_poster(movie_id):
    """Fetches movie poster URL from IMDb."""
    ia = IMDb()
    movie = ia.get_movie(movie_id)
    if 'cover url' in movie:
        return movie['cover url']
    return None  # Return None if no poster is available


def fetch_similar_movies(movie_id):
    """Fetches similar movies from TMDb API."""
    try:
        similar = movie_api.similar(movie_id)
        if similar.results:
            return [movie.title for movie in similar.results][:3]  # Limit to top 3 similar movies

        recom = movie_api.recommendations(movie_id)
        if recom.results:
            return [movie.title for movie in recom.results][:3]
        return []
    except Exception as e:
        print("Error fetching similar movies:", e)
        return []


def convert_duration_to_minutes(duration_str):
    """Convert duration in 'XXh YYm' format to total minutes."""
    hours, minutes = 0, 0
    if 'h' in duration_str:
        hours = int(duration_str.split('h')[0].strip())
        if 'm' in duration_str:
            minutes = int(duration_str.split('h')[1].split('m')[0].strip())
    return hours * 60 + minutes


def scrape_text(soup, class_name, tag='span'):
    """Helper method to extract text from a specified tag and class."""
    element = soup.find(tag, {'class': class_name})
    return element.text.strip() if element else "Unknown"


def scrape_reviews_count(soup):
    """Scrape the number of user and critic reviews."""
    scores = soup.find_all('span', {'class': 'score'})
    user_reviews, critic_reviews = "Unknown", "Unknown"

    for score in scores:
        if score.findNext().text == "User reviews":
            user_reviews = score.text
        elif score.findNext().text == "Critic reviews":
            critic_reviews = score.text
    return user_reviews, critic_reviews


def scrape_genres(soup):
    """Scrape the movie genres."""
    genre_items = soup.find_all('a', {'class': 'ipc-chip ipc-chip--on-baseAlt'})
    return [genre_item.text for genre_item in genre_items]


def scrape_color(soup):
    """Scrape the color information (e.g., Color, Black and White)."""
    color = "Unknown"
    candidate_color_elements = soup.find_all('a',
                                             class_="ipc-metadata-list-item__list-content-item "
                                                    "ipc-metadata-list-item__list-content-item--link")

    for color_element in candidate_color_elements:
        color_text = color_element.text.strip()
        if color_text in ["Color", "Black and White"]:
            color = color_text
            break
    return color


def scrape_budget_and_gross(soup):
    """Scrape the budget and worldwide gross of the movie."""
    dollar_elements = soup.find_all('span', class_="ipc-metadata-list-item__list-content-item",
                                    string=lambda text: '$' in text)
    budget, worldwide_gross = "Unknown", "Unknown"

    if len(dollar_elements) >= 4:
        budget = dollar_elements[0].text.strip()
        worldwide_gross = dollar_elements[3].text.strip()

    # Clean the budget and gross values
    cleaned_budget = re.sub(r'\(.*?\)', '', budget).replace('$', '').replace(',', '').strip()
    cleaned_gross = re.sub(r'\(.*?\)', '', worldwide_gross).replace('$', '').replace(',', '').strip()

    return cleaned_budget, cleaned_gross


def scrape_language_and_country(soup):
    """Scrape the language and country of origin."""
    # Scrape the language
    language_label = soup.find('span', class_="ipc-metadata-list-item__label", string="Languages")
    language = language_label.find_next('a',
                                        class_="ipc-metadata-list-item__list-content-item "
                                               "ipc-metadata-list-item__list-content-item--link").text.strip() if (
        language_label) else "Unknown"

    # Scrape the country
    country_label = soup.find('span', class_="ipc-metadata-list-item__label", string="Countries of origin")
    country = country_label.find_next('a',
                                      class_="ipc-metadata-list-item__list-content-item "
                                             "ipc-metadata-list-item__list-content-item--link").text.strip() if (
        country_label) else "Unknown"

    return language, country


def convert_to_number(value):
    """
    Convert string numbers with 'K', 'M', 'B' suffixes to actual numbers.

    Examples:
        5.1K -> 5100
        1.4M -> 1400000
        2B   -> 2000000000
        150  -> 150
    """
    value = value.strip()  # Remove any leading/trailing spaces

    # Handle empty or None values
    if not value:
        return 0

    # Check if the value ends with 'K', 'M', or 'B'
    if value[-1].upper() == 'K':
        return int(float(value[:-1]) * 1_000)  # Convert 'K' (thousand) to number
    elif value[-1].upper() == 'M':
        return int(float(value[:-1]) * 1_000_000)  # Convert 'M' (million) to number
    elif value[-1].upper() == 'B':
        return int(float(value[:-1]) * 1_000_000_000)  # Convert 'B' (billion) to number
    else:
        # No suffix, return the number as is
        return int(value.replace(',', ''))  # Ensure commas are removed if present


def scrape_movie_details(movie_name):
    """
    Scrapes movie details and fills missing features with median values from the dataset.

    :param movie_name: The name of the movie to scrape
    :return: A dictionary of movie features
    """
    movie_features = {}

    # Get IMDb link
    link = get_imdb_link(movie_name)
    if link is None:
        print(f"Movie {movie_name} not found.")
        return None

    movie_id = get_movie_id(link)

    with requests.session() as session:
        main_link = f"https://www.imdb.com/title/{movie_id}"
        response = session.get(main_link, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Scrape basic features
        movie_features["movie_title"] = movie_name
        imdb_score = scrape_text(soup, "sc-d541859f-1 imUuxf")
        if imdb_score != "Unknown":
            movie_features["imdb_score"] = float(imdb_score)

        duration = soup.find('li', string=lambda text: text and 'h' in text and 'm' in text)
        if duration:
            movie_features["duration"] = convert_duration_to_minutes(duration.text)

        num_votes = scrape_text(soup, "sc-d541859f-3 dwhNqC", tag='div')
        if num_votes != "Unknown":
            movie_features["num_voted_users"] = convert_to_number(num_votes)

        content_rating = soup.find_all('a', {"class": "ipc-link ipc-link--baseAlt "
                                                      "ipc-link--inherit-color"})[-1].text
        if content_rating != "Unknown":
            movie_features["content_rating"] = content_rating

        title_year = soup.find_all('a', {"class": "ipc-link ipc-link--baseAlt ipc-link--inherit-color"})[-2].text
        if title_year != "Unknown":
            movie_features["title_year"] = title_year

        num_critic_for_reviews, num_user_for_reviews = scrape_reviews_count(soup)
        if num_critic_for_reviews != "Unknown":
            movie_features["num_critic_for_reviews"] = convert_to_number(num_critic_for_reviews)
            movie_features["Num Of Reviews"] = movie_features["num_critic_for_reviews"]

        if num_user_for_reviews != "Unknown":
            movie_features["num_user_for_reviews"] = convert_to_number(num_user_for_reviews)

        color = scrape_color(soup)
        if color != "Unknown":
            movie_features["color"] = color

        budget, gross = scrape_budget_and_gross(soup)
        if budget != "Unknown":
            movie_features["budget"] = budget
        if gross != "Unknown":
            movie_features["gross"] = gross

        language, country = scrape_language_and_country(soup)
        if language != "Unknown":
            movie_features["language"] = language
        if country != "Unknown":
            # Match the scrapped country value to the value that appears in the dataset
            if country == "United States":
                movie_features["country"] = "USA"
            elif country == "United Kingdom":
                movie_features["country"] = "UK"
            else:
                movie_features["country"] = country

        return movie_features


def scrape_reviews(movie_id):
    reviews = []
    reviews_link = f"https://www.imdb.com/title/{movie_id}/criticreviews/?ref_=tt_ov_rt"

    response = requests.get(reviews_link)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Scraping logic (based on previous example)
    first_texts_container = soup.find_all('span', {'class': "sc-d8486f96-6 bwEuBg"})
    second_texts_container = soup.find_all('a', {'class': "ipc-link ipc-link--base sc-d8486f96-6 bwEuBg"})

    for par in first_texts_container:
        reviews.append(par.next.next.text)
    for par in second_texts_container:
        reviews.append(par.next.next.next.next.text)
    del (reviews[0::2])  # due to the way the page of IMDb is formatted

    return reviews


# Predicting sentiment using fine-tuned BERT
def predict_sentiment_of_reviews(reviews, model, tokenizer, device=torch.device('cuda')):
    sentiments = []

    for review in reviews:
        encoding = tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=512,
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
        sentiments.append(sentiment_mapping[predicted_class_id])

    return sentiments


# Calculate average sentiment for a movie
def calc_avg_sentiment(sentiments):
    return sentiment_avg(sentiments)


def derive_review_features(reviews):
    """Derives average review features for a given movie."""
    avg_words = mean_words(reviews)
    avg_letters = letters_per_word(reviews)
    numbers_freq_value = numbers_freq(reviews)
    capitals_freq_value = capital_letter(reviews)
    punc_freq_value = punctuation_freq(reviews)
    return {
        "Avg_words": avg_words,
        "Avg_letters": avg_letters,
        "Numbers_freq": numbers_freq_value,
        "Capitals_freq": capitals_freq_value,
        "Punc_freq": punc_freq_value
    }


def load_median_values(dataset):
    """
    Loads the median values for the given dataset.

    :param dataset: The dataset used to calculate medians
    :return: A dictionary of feature -> median value
    """

    median_values = dataset.drop(columns=["language", "content_rating", "color", "country"],
                                 errors="ignore").median().to_dict()

    return median_values


def train_and_save_models(X_train, y_train, directory='./saved_trained_models/interactive_prog/'):
    """Train or load SVM, RandomForest, and MLP models."""
    models = {
        "SVM": SVR(),
        "RandomForest": RandomForestRegressor(),
        "MLP": MLPRegressor()
    }
    trained_models = {}

    for model_name, model in models.items():
        saved_model = load_model(model_name, directory)
        if saved_model:
            trained_models[model_name] = saved_model
        else:
            # Check and load hyperparameters, train model, and save
            file_path = os.path.join("hyper_parameters_log", f'{model_name}_expanded_dataset_best_params.json')
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    best_params = json.load(f)
                model.set_params(**best_params)
            model.fit(X_train, y_train)
            trained_models[model_name] = model
            save_model(model, model_name, directory)

    return trained_models


def predict_movie_score(movie_vector, trained_models):
    """Predict the IMDb score using all trained models."""
    predictions = {}
    for model_name, model in trained_models.items():
        prediction = model.predict(movie_vector)
        predictions[model_name] = prediction[0]
    return predictions


def load_model(model_name, directory):
    """Load a model from disk."""
    model_path = os.path.join(directory, f'{model_name}.joblib')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


def save_model(model, model_name, directory):
    """Save a model to disk."""
    os.makedirs(directory, exist_ok=True)
    model_path = os.path.join(directory, f'{model_name}.joblib')
    joblib.dump(model, model_path)


def get_movie_outer_details(movie_name):
    """Fetch movie details including IMDb ID, poster URL, similar movies, year, and genres."""
    ia = IMDb()
    movies = ia.search_movie(movie_name)

    if not movies:
        return None, None, None, None, None  # Movie not found

    # Check for exact title match or suggest closest matches
    movie = movies[0]
    if movie['title'].lower() != movie_name.lower():
        suggestions = [m['title'] for m in movies[:5]]
        return None, None, suggestions, None, None

    movie_id = movie.movieID
    year = movie.get('year', 'N/A')
    movie = ia.get_movie(movie_id)
    genres = movie['genres']
    poster_url = fetch_movie_poster(movie_id)
    similar_movies = fetch_similar_movies(movie_id)

    return movie_id, poster_url, similar_movies, year, genres


def process_movie(movie_name, movie_id, model, tokenizer, median_values, scaler, feature_order, original_features):
    """Process movie vector and scrape additional details."""
    reviews = scrape_reviews(movie_id)

    # Deriving review features
    review_features = derive_review_features(reviews)
    sentiments = predict_sentiment_of_reviews(reviews, model, tokenizer)
    review_features["Avg_sentiment"] = calc_avg_sentiment(sentiments)

    # Compile other features (color, gross, duration, etc.)
    movie_features = scrape_movie_details(movie_name)
    if not movie_features:
        return None, None  # Return if movie details were not scraped

    movie_features.update(review_features)

    movie_df = pd.DataFrame([movie_features])
    dataset_drop_cols_with_names(movie_df)
    for feature in original_features:
        if feature not in movie_df.columns:
            movie_df[feature] = median_values.get(feature, np.nan)

    movie_df = dataset_handle_categorial_data(movie_df)
    movie_df = movie_df.reindex(columns=feature_order, fill_value=0)
    scaled_features = scaler.transform(movie_df.drop(columns=["imdb_score"]))
    return scaled_features, movie_features.get("imdb_score")
