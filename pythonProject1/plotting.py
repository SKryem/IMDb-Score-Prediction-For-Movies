import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json


def load_results(file_path):
    with open(file_path, 'r') as f:
        # Load JSON data
        results = json.load(f)
    return results


def plot_review_count_histogram():
    df = pd.read_csv('./data/imdb_reviews.csv')

    # Group by 'movie_name' and count the number of reviews per movie
    reviews_per_movie = df.groupby('Movie_name').size()

    # Plot a histogram of the number of reviews per movie
    plt.figure(figsize=(10, 6))
    sns.histplot(reviews_per_movie, bins=20, kde=False, color='blue')
    plt.title('Histogram of Number of Reviews per Movie')
    plt.xlabel('Number of Reviews')
    plt.ylabel('Number of Movies')

    plt.savefig('./plots/reviews_histogram.png', bbox_inches='tight')

    # Optionally, you can close the plot to free up memory
    plt.close()


def plot_basic_experiment_results():
    models = ["SVM", "RandomForest", "MLP"]
    datasets = ["original_dataset", "expanded_dataset"]

    mse_data = []
    r2_data = []

    # Load the results for each model and dataset
    for model in models:
        for dataset in datasets:
            file_path = f'./experiments_results/basic_experiment_logs/{model}_{dataset}_results.json'
            if os.path.exists(file_path):
                results = load_results(file_path)
                mse_data.append({"Model": model, "MSE": results["Mean Squared Error (MSE)"], "Dataset": dataset})
                r2_data.append({"Model": model, "R²": results["R-squared (R²)"], "Dataset": dataset})

    # Convert to DataFrames for plotting
    mse_df = pd.DataFrame(mse_data)
    r2_df = pd.DataFrame(r2_data)

    # Plot MSE with models on the x-axis and datasets represented by colors
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y="MSE", hue="Dataset", data=mse_df, palette="Set2")
    plt.title('MSE for Original and Expanded Datasets (Basic Experiment)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.xlabel('Model')
    plt.xticks(rotation=0)
    plt.legend(title="Dataset")
    plt.tight_layout()
    plt.savefig('./plots/basic_experiment_mse.png')
    plt.show()

    # Plot R² with models on the x-axis and datasets represented by colors
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y="R²", hue="Dataset", data=r2_df, palette="Set2")
    plt.title('R² for Original and Expanded Datasets (Basic Experiment)')
    plt.ylabel('R-squared (R²)')
    plt.xlabel('Model')
    plt.xticks(rotation=0)
    plt.legend(title="Dataset")
    plt.tight_layout()
    plt.savefig('./plots/basic_experiment_r2.png')
    plt.show()


# file_path = f'./basic_experiment_logs/{model}_expanded_dataset_results.json'
def plot_feature_impact_experiment_results():
    feature_names = ["Avg_words", "Avg_letters", "Numbers_freq", "Capitals_freq", "Punc_freq", "Avg_sentiment"]
    models = ["SVM", "RandomForest", "MLP"]

    mse_data = []

    for model in models:
        # Load MSE for the expanded dataset
        file_path = f'./experiments_results/basic_experiment_logs/{model}_expanded_dataset_results.json'
        if os.path.exists(file_path):
            results = load_results(file_path)
            mse_data.append({"Model": model, "MSE": results["Mean Squared Error (MSE)"], "Feature": "None"})

        # Load MSE for datasets with features removed
        for feature in feature_names:
            file_path = f'./experiments_results/multiple_features_logs/{feature}_logs/{model}_expanded_dataset_results.json'
            if os.path.exists(file_path):
                results = load_results(file_path)
                mse_data.append({"Model": model, "MSE": results["Mean Squared Error (MSE)"], "Feature": feature})

    # Convert to DataFrame for plotting
    mse_df = pd.DataFrame(mse_data)

    # Plot MSE for each model, with hue as the removed feature
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y="MSE", hue="Feature", data=mse_df, palette="Set2")
    plt.title('MSE Impact of Removing Features for Each Model')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.xlabel('Model')
    plt.xticks(rotation=0)
    plt.legend(title="Removed Feature")
    plt.tight_layout()

    # Save the plot as an image
    plt.savefig('./plots/feature_impact_mse_all_models.png')
    plt.show()


def plot_sentiment_prediction_results():
    models = ["SVM", "RandomForest", "MLP"]

    mse_data = {}
    r2_data = {}

    for model in models:
        file_path = f'./experiments_results/sentiment_prediction_logs/{model}_original_dataset_results.json'
        results = load_results(file_path)
        mse_data[model] = results["Mean Squared Error (MSE)"]
        r2_data[model] = results["R-squared (R²)"]

    # Create DataFrames for plotting
    mse_df = pd.DataFrame(mse_data, index=["MSE"])
    r2_df = pd.DataFrame(r2_data, index=["R²"])

    # Plot MSE
    plt.figure(figsize=(10, 6))
    mse_df.plot(kind='bar')
    plt.title('MSE for Sentiment Prediction (All Models)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.xlabel('Models')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('./plots/sentiment_prediction_mse.png')
    plt.show()

    # Plot R²
    plt.figure(figsize=(10, 6))
    r2_df.plot(kind='bar')
    plt.title('R² for Sentiment Prediction (All Models)')
    plt.ylabel('R-squared (R²)')
    plt.xlabel('Models')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('./plots/sentiment_prediction_r2.png')
    plt.show()


def plot_avg_sentiment_by_genre():
    # Read the CSV-like results file and load into a DataFrame

    avg_sentiment_data = pd.read_csv("./experiments_results/insights_logs/avg_sentiment_insights.txt", header=0,
                                     skipfooter=7,
                                     engine='python')  # Use skipfooter to avoid reading correlation lines

    # Strip any extra spaces from column names (if any)
    avg_sentiment_data.columns = avg_sentiment_data.columns.str.strip()

    # Plot the Average Sentiment by Genre
    plt.figure(figsize=(12, 8))
    sns.barplot(x='genre', y='Avg_sentiment', data=avg_sentiment_data, hue='genre', dodge=False, palette="Set3")

    plt.title('Average Sentiment by Genre', fontsize=16)
    plt.xlabel('Genre', fontsize=14)
    plt.ylabel('Average Sentiment', fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=12)  # Rotate x-axis labels for better readability
    plt.tight_layout()

    # Save the plot as an image
    plot_file = os.path.join("./plots/", "avg_sentiment_by_genre.png")
    plt.savefig(plot_file)
    plt.show()


def plot_genre_experiment_results(base_log_dir='./experiments_results/multiple_genres_logs'):
    """Plot the R² and MSE metrics for all genres and models."""
    r2_data = []
    mse_data = []
    models = ["SVM", "RandomForest", "MLP"]
    dataset = pd.read_csv("./data/filtered_expanded_movie_metadata.csv")

    # Extract genres and split by '|'
    all_genres = dataset['genres'].dropna().str.split('|').explode().unique()
    # Loop over each genre and model to load and store the results
    for genre in all_genres:
        for model in models:
            file_path = f'{base_log_dir}/{genre}_logs/{model}_expanded_dataset_results.json'
            if os.path.exists(file_path):
                # Use the existing load_results function
                results = load_results(file_path)
                r2_data.append({
                    "Genre": genre,
                    "Model": model,
                    "R²": results["R-squared (R²)"]
                })
                mse_data.append({
                    "Genre": genre,
                    "Model": model,
                    "MSE": results["Mean Squared Error (MSE)"]
                })
            else:
                print(f"File {file_path} not found.")

    # Convert to DataFrames for plotting
    r2_df = pd.DataFrame(r2_data)
    mse_df = pd.DataFrame(mse_data)

    # Plot R² for each genre and model
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Genre", y="R²", hue="Model", data=r2_df, palette="Set2")
    plt.title('R² for Different Genres and Models')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('./plots/r2_multiple_genres.png')
    plt.show()

    # Plot MSE for each genre and model
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Genre", y="MSE", hue="Model", data=mse_df, palette="Set3")
    plt.title('MSE for Different Genres and Models')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('./plots/mse_multiple_genres.png')
    plt.show()


def plot_year_range_experiment_results(log_dir="./experiments_results/multiple_year_range_experiment/"):
    """
    Plots the MSE and R-squared (R²) metrics for movies in different year ranges from the experiment results.

    :param log_dir: The directory where the result JSON files are stored.
    """
    # Initialize lists to store MSE and R² results for each model and year range
    mse_data = []
    r2_data = []

    # Assume models are the same as previous experiments
    models = ["SVM", "RandomForest", "MLP"]

    # Get all year range directories
    year_range_dirs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    # Load the results for each model and year range
    for interval_dir in year_range_dirs:
        year_range = interval_dir.split('_logs')[0]
        for model in models:
            file_path = os.path.join(log_dir, f'{interval_dir}/{model}_expanded_dataset_results.json')
            if os.path.exists(file_path):
                results = load_results(file_path)  # Assuming load_results loads JSON results
                mse_data.append({"Model": model, "Year Range": year_range, "MSE": results["Mean Squared Error (MSE)"]})
                r2_data.append({"Model": model, "Year Range": year_range, "R²": results["R-squared (R²)"]})

    # Convert the data to DataFrames for plotting
    mse_df = pd.DataFrame(mse_data)
    r2_df = pd.DataFrame(r2_data)

    mse_df_without_outlier = mse_df[mse_df["Year Range"] != "1957_1961"]
    r2_df_without_outlier = r2_df[r2_df["Year Range"] != "1957_1961"]

    # Plot with 1957-1961 included
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Year Range", y="MSE", hue="Model", data=mse_df, palette="Set2")
    plt.title('MSE for Year Range Experiment (With 1957-1961)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('./plots/multiple_year_range_mse_with_outlier.png')
    plt.show()

    # Plot without 1957-1961 interval
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Year Range", y="MSE", hue="Model", data=mse_df_without_outlier, palette="Set2")
    plt.title('MSE for Year Range Experiment (Without 1957-1961)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('./plots/multiple_year_range_mse_without_outlier.png')
    plt.show()

    # Plot R² with 1957-1961 included
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Year Range", y="R²", hue="Model", data=r2_df, palette="Set2")
    plt.title('R² for Year Range Experiment (With 1957-1961)')
    plt.ylabel('R-squared (R²)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('./plots/multiple_year_range_r2_with_outlier.png')
    plt.show()

    # Plot R² without 1957-1961 interval
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Year Range", y="R²", hue="Model", data=r2_df_without_outlier, palette="Set2")
    plt.title('R² for Year Range Experiment (Without 1957-1961)')
    plt.ylabel('R-squared (R²)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('./plots/multiple_year_range_r2_without_outlier.png')
    plt.show()

def main():
    os.makedirs('./plots3', exist_ok=True)
    # plot_review_count_histogram()
    # plot_basic_experiment_results()
    # plot_feature_impact_experiment_results()
    # plot_sentiment_prediction_results()
    # plot_avg_sentiment_by_genre()
    #
    # plot_genre_experiment_results()
    plot_year_range_experiment_results()


main()
