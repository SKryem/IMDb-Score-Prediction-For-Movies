import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from utils import get_datasets, dataset_handle_categorial_data, dataset_drop_cols_with_names
import json
import pandas as pd


def evaluate_model(model, X_test, y_true):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return mse, mae, r2, rmse


def log_results(model_name, dataset_name, results, log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Round the results to 3 decimal places
    logs = {
        "Model": model_name,
        "Dataset": dataset_name,
        "Mean Squared Error (MSE)": round(results[0], 3),
        "Mean Absolute Error (MAE)": round(results[1], 3),
        "R-squared (R²)": round(results[2], 3),
        "Root Mean Squared Error (RMSE)": round(results[3], 3)
    }

    log_file = f"{log_dir}/{model_name}_{dataset_name}_results.json"
    with open(log_file, "w") as f:
        json.dump(logs, f, indent=4)  # Save as JSON with pretty formatting

    print(f"Results logged to {log_file}")


def save_best_params(model_name, best_params, dataset_name):
    params_dir = "hyper_parameters_log"
    os.makedirs(params_dir, exist_ok=True)
    file_path = os.path.join(params_dir, f'{model_name}_{dataset_name}_best_params.json')

    with open(file_path, 'w') as f:
        json.dump(best_params, f)


def hyperparameter_tuning(model, param_grid, X_val, y_val, model_name, dataset_name):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_val, y_val)
    best_params = grid_search.best_params_

    # Save the best parameters
    save_best_params(model_name, best_params, dataset_name)

    return grid_search.best_estimator_


def load_best_params(model_name, dataset_name):
    file_path = os.path.join("hyper_parameters_log", f'{model_name}_{dataset_name}_best_params.json')
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None

def get_year_range(dataset):
    """Find the earliest and latest release years from the dataset."""
    column = 'title_year'
    min_year = dataset[column].min()
    max_year = dataset[column].max()
    return min_year, max_year


def split_movies_by_year_range(dataset,  interval=5, min_movies=10):
    """Split the dataset into 5-year intervals based on the release year and filter intervals with fewer than 10 movies."""
    column = 'title_year'
    min_year, max_year = get_year_range(dataset)

    # Convert the min and max years to integers
    min_year = int(min_year)
    max_year = int(max_year)

    # Create 5-year intervals
    year_ranges = [(start, start + interval - 1) for start in range(min_year, max_year + 1, interval)]

    # Filter out year ranges with fewer than 10 movies
    valid_year_ranges = []
    for start, end in year_ranges:
        filtered_data = dataset[(dataset[column] >= start) & (dataset[column] <= end)]
        if len(filtered_data) >= min_movies:
            valid_year_ranges.append((start, end))

    return valid_year_ranges


def filter_movies_by_year_range(dataset, start_year, end_year):
    """Filter movies based on a given year range."""
    column = 'title_year'
    return dataset[(dataset[column] >= start_year) & (dataset[column] <= end_year)]


def run_experiment(log_dir, original_dataset, expanded_dataset, experiment_name, target_label="imdb_score", ):
    # Define models and their parameter grids
    models = {
        "SVM": (SVR(), {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['linear', 'rbf', 'poly']
        }),
        "RandomForest": (RandomForestRegressor(random_state=42), {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }),
        "MLP": (MLPRegressor(max_iter=1000, random_state=42), {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'learning_rate': ['constant', 'adaptive'],
            'alpha': [0.0001, 0.001],
        })
    }

    datasets = [
        ("original_dataset", original_dataset),
        ("expanded_dataset", expanded_dataset)
    ]
    datasets = [(name, data) for name, data in datasets if data is not None]

    # Directory to save models (separate for each experiment)
    model_dir = os.path.join("saved_trained_models", experiment_name)
    os.makedirs(model_dir, exist_ok=True)

    # Run models
    for dataset_name, dataset in datasets:
        X = dataset.drop(columns=[target_label])
        y = dataset[target_label]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        for model_name, (model, model_param_grid) in models.items():
            model_path = os.path.join(model_dir, f"{model_name}_{dataset_name}.joblib")
            print(f"**** Model: {model_name} ****")
            # Check if the trained model exists for the current experiment
            if os.path.exists(model_path):
                print(f"Loading saved {model_name} model for {dataset_name} in experiment {experiment_name}...")
                model = joblib.load(model_path)
            else:
                print(
                    f"No saved {model_name} model for {dataset_name} found for experiment {experiment_name}. Training new model...")
                # Check if best hyperparameters are already saved
                best_params = load_best_params(model_name, dataset_name)
                if best_params:
                    model.set_params(**best_params)
                else:
                    # Perform tuning if no saved parameters are found
                    model = hyperparameter_tuning(model, model_param_grid, X_val_scaled, y_val, model_name,
                                                  dataset_name)

                # Train the model
                model.fit(X_train_scaled, y_train)

                # Save the trained model for the current experiment
                print(f"Saving trained {model_name} model for {dataset_name} in experiment {experiment_name}...")
                joblib.dump(model, model_path)

            # Evaluate
            results = evaluate_model(model, X_test_scaled, y_test)

            # Log results
            log_results(model_name, dataset_name, results, log_dir)

    print(f"Experiments completed and results saved to {log_dir}.")


def basic_experiment(log_dir):
    print("****************************")
    print("Running basic_experiment ...")
    print("Simple comparison between original and expanded datasets")
    original_dataset, expanded_dataset = get_datasets()
    run_experiment(log_dir, original_dataset, expanded_dataset, experiment_name="basic_experiment")


def feature_impact_experiment(log_dir, feature):
    print("****************************")
    print("Running feature_impact_experiment ...")
    print(f"Measuring the impact of the {feature} feature")
    # Load datasets
    _, expanded_dataset = get_datasets()

    expanded_dataset.drop(feature, axis=1, inplace=True)
    run_experiment(log_dir, None, expanded_dataset,
                   experiment_name=f"{feature}_impact_experiment")


def feature_impact_multiple_experiments(base_log_dir, features):
    for feature in features:
        # Create a unique log directory for each feature
        feature_log_dir = os.path.join(base_log_dir, f'{feature}_logs')
        os.makedirs(feature_log_dir, exist_ok=True)

        # Run the feature impact experiment
        feature_impact_experiment(feature_log_dir, feature)


def derived_features_sentiment_prediction_experiment(log_dir):
    print("****************************")
    print("Running sentiment_prediction_experiment ...")
    # Load datasets
    _, expanded_dataset = get_datasets()
    sentiment_prediction_dataset = expanded_dataset[["Avg_words", "Avg_letters", "Numbers_freq", "Capitals_freq",
                                                     "Punc_freq", "Avg_sentiment"]]
    run_experiment(log_dir, sentiment_prediction_dataset, None, experiment_name="sentiment_prediction_experiment"
                   , target_label="Avg_sentiment")


def avg_sentiment_insights_experiment(log_dir):
    print("****************************")
    print("Running avg_sentiment_insights_experiment ...")
    # Load datasets, we can't use "get_datasets" becuase it removes the "genres" feature
    expanded_dataset = pd.read_csv("./data/filtered_expanded_movie_metadata.csv")
    expanded_dataset = dataset_handle_categorial_data(expanded_dataset)

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "avg_sentiment_insights.txt")

    with open(log_file, "w") as f:
        # Split the genres and create a new DataFrame with one genre per row (each movie can have more than 1 genre)
        genre_sentiment_data = expanded_dataset[['genres', 'Avg_sentiment']].copy()
        genre_sentiment_data = genre_sentiment_data.dropna(subset=['genres'])
        genre_sentiment_data['genre'] = genre_sentiment_data['genres'].str.split('|')
        genre_sentiment_data = genre_sentiment_data.explode('genre')

        # Filter out genres with fewer than 10 movies
        genre_counts = genre_sentiment_data['genre'].value_counts()
        valid_genres = genre_counts[genre_counts > 10].index
        genre_sentiment_data = genre_sentiment_data[genre_sentiment_data['genre'].isin(valid_genres)]

        # Calculate the average sentiment for each individual genre
        avg_sentiment_by_genre = genre_sentiment_data.groupby("genre")["Avg_sentiment"].mean()
        # Write results in CSV format to be more readable for future plotting
        f.write("genre, Avg_sentiment\n")
        for genre, sentiment in avg_sentiment_by_genre.items():
            f.write(f"{genre},{sentiment:.6f}\n")

        f.write("\n\n")
        # 2. Correlation between Avg_sentiment and imdb_score
        correlation_sentiment_imdb = expanded_dataset["Avg_sentiment"].corr(expanded_dataset["imdb_score"])
        f.write(f"Correlation between Avg_sentiment and imdb_score: {correlation_sentiment_imdb:.3f}\n\n")

        # 3. Correlation between Avg_sentiment and gross
        correlation_sentiment_gross = expanded_dataset["Avg_sentiment"].corr(expanded_dataset["gross"])
        f.write(f"Correlation between Avg_sentiment and gross: {correlation_sentiment_gross:.3f}\n\n")

        # 4. Correlation between Avg_sentiment and budget
        correlation_sentiment_budget = expanded_dataset["Avg_sentiment"].corr(expanded_dataset["budget"])
        f.write(f"Correlation between Avg_sentiment and budget: {correlation_sentiment_budget:.3f}\n\n")

    print(f"Experiments completed and results saved to {log_dir}.")


def genre_experiment(log_dir, genre):
    print("****************************")
    print(f"Running genre_experiment for {genre}...")

    # Load the dataset
    expanded_dataset = pd.read_csv("./data/filtered_expanded_movie_metadata.csv")

    # Filter for the specific genre
    genre_filtered_dataset = expanded_dataset[expanded_dataset['genres'].str.contains(genre)].copy()
    dataset_drop_cols_with_names(genre_filtered_dataset)
    genre_filtered_dataset = dataset_handle_categorial_data(genre_filtered_dataset)

    if genre_filtered_dataset.shape[0] <= 10:  # make sure there are at least 10 movies of this genre
        print(f"Not enough samples for genre: {genre} (Number of samples ={genre_filtered_dataset.shape[0]} ")

        return

    experiment_name = genre + "Accuracy"
    # Run the experiment for the filtered genre dataset
    run_experiment(log_dir, None, genre_filtered_dataset, experiment_name=experiment_name)


def multiple_genre_experiments(base_log_dir):
    dataset = pd.read_csv("./data/filtered_expanded_movie_metadata.csv")

    # Extract genres and split by '|'
    all_genres = dataset['genres'].dropna().str.split('|').explode().unique()
    print("All genres:", all_genres)
    for genre in all_genres:
        genre_log_dir = os.path.join(base_log_dir, f'{genre}_logs')
        os.makedirs(genre_log_dir, exist_ok=True)

        # Run the experiment for each genre
        genre_experiment(genre_log_dir, genre)


def run_experiment_for_year_range(dataset, year_range, log_dir):
    """Run the prediction experiment for a specific year range."""
    start_year, end_year = year_range
    print(f"Running experiment for movies released between {start_year} and {end_year}...")

    # Filter the dataset for the current year range
    filtered_dataset = filter_movies_by_year_range(dataset, start_year, end_year)

    # Run the experiment (similar to feature impact experiments)
    experiment_name = str(start_year) + '_' + str(end_year) + "Accuracy"
    run_experiment(log_dir, None, filtered_dataset, experiment_name=experiment_name)


def multiple_year_range_experiment(base_log_dir):
    """Run experiments for multiple 5-year intervals."""
    _, expanded_dataset = get_datasets()
    year_ranges = split_movies_by_year_range(expanded_dataset)

    for year_range in year_ranges:
        log_dir = os.path.join(base_log_dir, f'{year_range[0]}_{year_range[1]}_logs')
        os.makedirs(log_dir, exist_ok=True)
        run_experiment_for_year_range(expanded_dataset, year_range, log_dir)
