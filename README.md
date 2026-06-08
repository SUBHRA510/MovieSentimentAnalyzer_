import os

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
