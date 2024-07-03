import tkinter as tk
from tkinter import messagebox
import joblib  
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.tokenize import word_tokenize



model = joblib.load('svm_model.pkl')  
vec = joblib.load('tfidf_vectorizer.pkl')  
def preprocess_text(text):
    tk = TweetTokenizer()
    stemmer = SnowballStemmer('english')
    sw = set(stopwords.words('english'))
    tokens = tk.tokenize(text)
    text = ' '.join(tokens)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    text = ' '.join([w for w in word_tokenize(text) if len(w) >= 3])
    text = ' '.join([stemmer.stem(i.lower()) for i in tk.tokenize(text) if i.lower() not in sw])
    return text
def classify_email():
    email_text = email_entry.get("1.0",'end-1c')  #widget
    if email_text.strip() == "":
        messagebox.showwarning("Warning", "Please enter an email text.")
        return
    processed_text = preprocess_text(email_text)

    transformed_text = vec.transform([processed_text])

    prediction = model.predict(transformed_text)

    result = 'Spam' if prediction[0] == 1 else 'Ham'

    result_label.config(text=f"Classification Result: {result}") #update

root = tk.Tk() # window
root.title("Email Classifier")

email_label = tk.Label(root, text="Enter Email Text:") #label widget
email_label.pack()

email_entry = tk.Text(root, height=10, width=50)
email_entry.pack()

classify_button = tk.Button(root, text="Classify Email", command=classify_email)
classify_button.pack(pady=10)

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()