"""
This is the Main Code to RUN for all the classification Algorithms implemented.
This code is used to classify posts based on movie reviews from Reddit dataset into 3 different classes
The classes are: positive, neutral and negative.
"""

import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import argparse
import os
from emoji import demojize
import contractions
import praw

# ------ Data Preprocessing -------
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()

slang_dict = {
    "u": "you",
    "r": "are",
    "2moro": "tomorrow",
    "gr8": "great",
    "lol": "laugh out loud",
    "btw": "by the way",
    "thx": "thanks",
    "omg": "oh my god",
    "idk": "I don't know",
    "bff": "best friends forever",
    "brb": "be right back",
    "gtg": "got to go",
    "imo": "in my opinion",
    "tbh": "to be honest",
    "wtf": "what the f***",
    "omw": "on my way",
    "irl": "in real life",
    "fyi": "for your information",
    "jk": "just kidding",
    "np": "no problem",
    "oml": "oh my lord",
    "smh": "shake my head",
    "tbt": "throwback Thursday",
    "yolo": "you only live once"
}

contractions_dict = {
    "isn't": "is not",
    "don't": "do not",
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not"
}

# Variables Initialization
sentiment_dict = {}

"""
Function to generate Lexicon of sentiments with a Polarity, from a text file AFINN-111.txt
:parameter Link/PATH of file to get lexicons from
:returns affin_list
"""
affin_dict = {}

def generateAffinityList(datasetLink):
    try:
        with open(datasetLink) as f:
            affin_list = f.readlines()
    except Exception as e:
        print(f"ERROR: Opening File {datasetLink}: {e}")
        exit(0)
    return affin_list

"""
This function is used to create a Dictionary of words according to the polarities
Every word from the AFFIN-111 Lexicon is categorized
We have taken 4 Categories:
Very Positive Words, Positive Words, Negative Words, Very Negative Words
:parameter affin_list
"""
vNegative, Negative, Positive, vPositive = [], [], [], []
def createDictionaryFromPolarity(affin_list):
    global affin_dict
    for word in affin_list:
        word_split = word.split("\t")
        word_text = word_split[0].lower()
        word_score = int(word_split[1].strip())
        affin_dict[word_text] = word_score

    for word_text, word_score in affin_dict.items():
        if word_score in [-4, -5]:
            vNegative.append(word_text)
        elif word_score in [-3, -2, -1]:
            Negative.append(word_text)
        elif word_score in [1, 2, 3]:
            Positive.append(word_text)
        elif word_score in [4, 5]:
            vPositive.append(word_text)


def preprocessing(dataSet):
    processed_data = []
    for text in dataSet:
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", 'URL', text, flags=re.MULTILINE)  # Replace URLs with 'URL'
        text = re.sub(r'#(\w+)', r'\1', text)  # Remove hashtags but keep the words
        text = demojize(text, delimiters=(" ", " "))  # Convert emojis to text
        text = contractions.fix(text)  # Expand contractions
        text = ' '.join(contractions_dict.get(word, word) for word in text.split())
        text = re.sub(r'[^\w\s]', '', text)  # Remove all punctuations
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in stopwords.words('english')]
        tokens = [slang_dict[token] if token in slang_dict else token for token in tokens]
        tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Use only lemmatization
        cleaned_text = " ".join(tokens)
        if cleaned_text:  # Only add non-empty texts
            processed_data.append(cleaned_text)

    print(f"Number of posts after preprocessing: {len(processed_data)}")
    return processed_data

"""
This function is used to classify the Data using
Support Vector Machine Algorithm
:parameter train_X, train_Y, test_X
:returns yHat
"""
def classify_svm(train_X, train_Y, test_X):
    print("Classifying using Support Vector Machine ...")
    clf = SVC(class_weight='balanced')
    clf.fit(train_X, train_Y)
    yHat = clf.predict(test_X)
    return yHat

"""
This function is used to evaluate the performance of the classifier
It is used to calculate the Precision, Recall, F-M1easure and Accuracy
using the confusion matrix
:parameter conf_mat Confusion Matrix
"""

def evaluate_classifier(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred, labels=[1, 0, -1])
    print("Confusion Matrix:\n", conf_mat)
    
    # Calculate precision, recall, and fscore for the whole dataset
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

    # Print precision, recall, and fscore
    print(f"Overall Precision: {precision}")
    print(f"Overall Recall: {recall}")
    print(f"Overall F1-Measure: {fscore}")

    # Calculate and print accuracy
    accuracy = np.trace(conf_mat) / np.sum(conf_mat)
    print("Accuracy: ", accuracy)

def label_using_affin(text, affin_dict):
    score = 0
    for word in text.split():
        score += affin_dict.get(word, 0)
    if score > 0:
        return 1
    elif score < 0:
        return -1
    else:
        return 0

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentimental Analysis of Movie Reviews")
    parser.add_argument("Algorithm", help="Classification Algorithm to be used (all gnb svm maxEnt rf)", metavar='algo')
    parser.add_argument("Crossvalidation", help="Using Cross validation (yes/no)", metavar='CV')
    args = parser.parse_args()

    # Fetch the current working dir
    os.chdir('../')
    dirPath = os.getcwd()

    # STEP 1: Generate Affinity List
    print("Please wait while we Classify your data ...")
    affin_list = generateAffinityList(dirPath+"/Data/Affin_Data.txt")

    # STEP 2: Create Dictionary based on Polarities from the Lexicons
    createDictionaryFromPolarity(affin_list)

    # Fetch Reddit titles
    reddit = praw.Reddit(
        client_id="dpggItpTIl7DA_6EaKKC5g",
        client_secret="DXJBIePIaXLU2Gpd-EpyaKoWc2AEeg",
        user_agent="Movie-sentiment-analysis by u/FixNo6557",
    )
    subreddit = reddit.subreddit("Spiderman")
    titles = []
    for post in subreddit.top(limit=200):
        titles.append(post.selftext)
    for post in subreddit.new(limit=200):
        titles.append(post.selftext)
    print(f"Raw: {len(titles)} posts fetched")

    print("Processing Reddit titles ...")
    processed_titles = preprocessing(titles)
    print(f"Processed: {len(processed_titles)} posts")

    # Label the posts using Affin dictionary
    labels = [label_using_affin(title, affin_dict) for title in processed_titles]

    # Generate feature vectors
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(processed_titles)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    print(f"Training set size: {len(y_train)}, Testing set size: {len(y_test)}")

    print("Training the Classifier according to the data provided ...")
    print("Classifying the Test Data ...")
    print("Evaluation Results will be displayed Shortly ...")

    if args.Crossvalidation.lower() == "no":
        yHat = classify_svm(X_train, y_train, X_test)
        evaluate_classifier(y_test, yHat)

    if args.Crossvalidation.lower() == "yes":
        cv_kFold = KFold(n_splits=10, shuffle=True, random_state=5)
        i = 0
        print("Starting "+str(cv_kFold.n_splits)+" Crossvalidation")
        all_y_true = []
        all_y_pred = []
        for train_idx, test_idx in cv_kFold.split(X):
            X_train_cv, X_test_cv = np.array([X[ele].toarray()[0] for ele in train_idx]), np.array([X[ele].toarray()[0] for ele in test_idx])
            Y_train_cv, Y_test_cv = np.array([labels[ele] for ele in train_idx]), np.array([labels[ele] for ele in test_idx])

            i += 1
            print("Fold: ", i)
            yHat = classify_svm(X_train_cv, Y_train_cv, X_test_cv)
            all_y_true.extend(Y_test_cv)
            all_y_pred.extend(yHat)
        
        evaluate_classifier(all_y_true, all_y_pred)