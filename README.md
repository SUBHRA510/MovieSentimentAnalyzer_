## Movie Review Sentiment Analysis 🎬 | NLP + Tkinter + Sklearn

## 1. Problem Statement
Movie production houses and streaming platforms like Netflix, Stan, and Disney+ AU receive thousands of user reviews daily. Manually reading each review to gauge audience sentiment is time-consuming and not scalable. 

**Goal**: Build an automated system that classifies a movie review as `Positive` or `Negative` in real-time, so studios can quickly measure public reaction after a release.

## 2. Dataset
- **Source**: `movie_reviews.csv` - Custom dataset of movie reviews
- **Size**: N rows with 2 columns: `review` (text) and `sentiment` (label)
- **Location**: Stored locally at `C:/Users/Subhra/OneDrive/Desktop/python opencv/movie_reviews.csv`
- **Note**: For production, this would connect to IMDB/Twitter API or AWS S3.

## 3. Tech Stack & Methodology
| **Component** | **Technology Used** | **Why** |
| --- | --- | --- |
| **Data Handling** | `Pandas` | Industry standard for CSV processing |
| **ML Pipeline** | `Scikit-learn Pipeline` | Combines vectorization + model for clean code & deployment |
| **Text Vectorization** | `CountVectorizer + TfidfTransformer` | Converts text to numerical features. TF-IDF weights rare words higher |
| **Model** | `Multinomial Naive Bayes` | Fast, works well for text classification, baseline for NLP tasks |
| **GUI** | `Tkinter` | Lightweight Python GUI. No web server needed for demo |
| **Validation** | `train_test_split` | 80-20 split with `random_state=42` for reproducibility |

**ML Pipeline Flow**:
`Raw Text → CountVectorizer → TF-IDF → Naive Bayes → Prediction`

## 4. Key Features
1. **End-to-End GUI**: User can paste any review and get instant sentiment prediction. No coding needed.
2. **Error Handling**: Warns if input is empty using `messagebox`. Checks if file exists with `os.path.exists`.
3. **Reproducible**: `random_state=42` ensures same train/test split every run.
4. **Modular**: Sklearn `Pipeline` makes it easy to swap `MultinomialNB` with `LogisticRegression` or `SVM` later.

## 5. How to Run
```bash

pip install pandas scikit-learn

python sentiment_analyzer.pyimport os

file_path = r"C:\Users\Subhra\OneDrive\Desktop\python opencv\movie_reviews.csv"
print(os.path.exists(file_path))  # Should print: True


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from tkinter import Tk, Label, Text, Button, END, messagebox

file_path = r"C:/Users/Subhra/OneDrive/Desktop/python opencv/movie_reviews.csv"
data = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='warn')

X = data['review']
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

model.fit(X_train, y_train)

def predict_sentiment():
    review = review_text.get("1.0", END).strip()
    if not review:
        messagebox.showwarning("Input Error", "Please enter a movie review.")
        return
    prediction = model.predict([review])[0]
    result_label.config(text=f"Predicted Sentiment: {prediction}")


root = Tk()
root.title("Movie Review Sentiment Analysis")
root.geometry("600x400")


Label(root, text="Enter your movie review:", font=("Helvetica", 14)).pack(pady=10)
review_text = Text(root, height=8, width=70, font=("Helvetica", 12))
review_text.pack(pady=10)

predict_button = Button(root, text="Analyze Sentiment", font=("Helvetica", 12, "bold"), command=predict_sentiment)
predict_button.pack(pady=10)


result_label = Label(root, text="", font=("Helvetica", 14, "bold"), fg="blue")
result_label.pack(pady=10)

root.mainloop()
