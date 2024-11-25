import tkinter as tk
from tkinter import messagebox
from interactive_utils import (
    initialize_bert_model, process_movie, train_and_save_models, predict_movie_score, prepare_data_for_interactive,
    get_movie_outer_details
)
import requests
from PIL import Image, ImageTk
from io import BytesIO


def on_predict(movie_entry, result_label, poster_label, suggestions_textbox, movie_details_label, BERT_model,
               tokenizer, median_values, scaler, feature_order, original_features, trained_models):
    movie_name = movie_entry.get().strip()
    if not movie_name:
        messagebox.showwarning("Input Error", "Please enter a movie name.")
        return

    # Clear previous suggestions and movie details
    suggestions_textbox.config(state=tk.NORMAL)
    suggestions_textbox.delete("1.0", tk.END)
    movie_details_label.config(text="")

    # Updated to unpack additional details from get_movie_outer_details
    movie_id, poster_url, similar_movies, year, genres = get_movie_outer_details(movie_name)

    if movie_id is None:
        if similar_movies:
            suggestions_text = f"No exact match for '{movie_name}'. Did you mean:\n" + "\n".join(similar_movies)
        else:
            suggestions_text = "Movie not found. Please try again with a different name."
        result_label.config(text="")
        poster_label.config(image="")  # Clear the poster if not found
        suggestions_textbox.insert(tk.END, suggestions_text)
        suggestions_textbox.config(state=tk.DISABLED)
        return

    # Display Movie Details
    movie_details = f"Movie: {movie_name} | Year: {year} | Genres: {', '.join(genres)}"
    movie_details_label.config(text=movie_details)

    movie_vector, true_score = process_movie(movie_name, movie_id, BERT_model, tokenizer, median_values, scaler,
                                             feature_order, original_features)
    if movie_vector is not None:
        predictions = predict_movie_score(movie_vector, trained_models)
        result_text = f"Predicted IMDb scores for '{movie_name}':\n" + "\n".join(
            f"{model}: {score:.2f}" for model, score in predictions.items())
        if true_score:
            result_text += f"\nActual IMDb Score: {true_score}"
        result_label.config(text=result_text, fg="green")
    else:
        result_label.config(text="Movie data unavailable. Please try again later.", fg="red")

    # Display Poster
    if poster_url:
        response = requests.get(poster_url)
        poster_img = Image.open(BytesIO(response.content))
        poster_img.thumbnail((250, 375))  # Larger poster size
        poster_img = ImageTk.PhotoImage(poster_img)
        poster_label.config(image=poster_img)
        poster_label.image = poster_img  # Keep reference

    # Display Suggestions for Similar Movies
    if similar_movies:
        suggestions_text = "Similar Movies:\n" + "\n".join(similar_movies)
        suggestions_textbox.insert(tk.END, suggestions_text)
        suggestions_textbox.config(state=tk.DISABLED)
    else:
        suggestions_textbox.insert(tk.END, "No similar movies available.")
        suggestions_textbox.config(state=tk.DISABLED)

def main():
    BERT_model, tokenizer, device = initialize_bert_model()
    dataset, original_features, median_values, scaler, X_train_scaled, y_train, feature_order = prepare_data_for_interactive()
    trained_models = train_and_save_models(X_train_scaled, y_train, "./saved_trained_models/interactive_prog/")

    # Setup GUI
    root = tk.Tk()
    root.title("IMDb Score Predictor")

    root.configure(bg="lightgray")
    tk.Label(root, text="Enter Movie Name:", font=("Helvetica", 14, "bold"), bg="lightgray").pack(pady=(10, 5))

    movie_entry = tk.Entry(root, font=("Helvetica", 12))
    movie_entry.pack(pady=(0, 10))

    # Updated movie details label
    movie_details_label = tk.Label(root, font=("Helvetica", 12, "italic"), bg="lightgray")
    movie_details_label.pack(pady=(5, 10))

    result_label = tk.Label(root, text="", font=("Helvetica", 12), wraplength=400, justify="left", bg="lightgray")
    result_label.pack(pady=(5, 10))

    poster_label = tk.Label(root, bg="lightgray")
    poster_label.pack(pady=(10, 10))

    # Updated similar movies to be a Text widget, for selectable text
    suggestions_textbox = tk.Text(root, font=("Helvetica", 12, "italic"), wrap="word", height=10, bg="lightgray")
    suggestions_textbox.pack(pady=(10, 20))

    tk.Button(
        root, text="Predict Score", font=("Helvetica", 12, "bold"), bg="blue", fg="white",
        command=lambda: on_predict(
            movie_entry, result_label, poster_label, suggestions_textbox, movie_details_label, BERT_model, tokenizer,
            median_values, scaler, feature_order, original_features, trained_models)
    ).pack(pady=(10, 20))

    root.geometry("500x750")
    root.mainloop()

if __name__ == "__main__":
    main()
