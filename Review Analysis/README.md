```python
# There are 1000 reviews for restaurants and films in a collection in the csv file on
# Brightspace. All those reviews are labeled with its category (either restaurant review or movie
# review). Developing classifiers that could automatically determine whether a future
# text body is a restaurant review or a movie review.

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')
from nltk.stem import WordNetLemmatizer
import string

# Step 1: Load and Prepare the Dataset
dataset = pd.read_excel('file.xlsx')

restaurant_reviews = dataset[dataset['label'] == 'restaurant'].iloc[:400]
movie_reviews = dataset[dataset['label'] == 'movie'].iloc[:400]

train_dataset = pd.concat([restaurant_reviews, movie_reviews], ignore_index=True)
test_dataset = dataset.iloc[800:]

train_reviews = train_dataset['review'].tolist()
train_labels = train_dataset['label'].tolist()

test_reviews = test_dataset['review'].tolist()
test_labels = test_dataset['label'].tolist()

# Step 2: Transform Reviews into TF-IDF Matrix
stopwords_set = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    filtered_words = [word for word in lemmatized_words if word.lower() not in stopwords_set and word not in string.punctuation]
    return ' '.join(filtered_words)

preprocessed_train_reviews = [preprocess_text(review) for review in train_reviews]
preprocessed_test_reviews = [preprocess_text(review) for review in test_reviews]

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5)
train_features = vectorizer.fit_transform(preprocessed_train_reviews)
test_features = vectorizer.transform(preprocessed_test_reviews)

# Step 3: Train and Evaluate Models
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=50),
    'Support Vector Machine': SVC(),
    'Artificial Neural Network': MLPClassifier(hidden_layer_sizes=(4,), random_state=42)
}

results = {}

for model_name, model in models.items():
    model.fit(train_features, train_labels)
    predictions = model.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    results[model_name] = accuracy

# Print accuracy results
for model_name, accuracy in results.items():
    print(f'{model_name}: {accuracy}')

# Determine the best performing model
best_model = max(results, key=results.get)
print(f'\nBest performing model: {best_model}')

```

    Naive Bayes: 0.99
    Logistic Regression: 0.99
    Random Forest: 1.0
    Support Vector Machine: 0.99
    Artificial Neural Network: 0.995
    
    Best performing model: Random Forest

