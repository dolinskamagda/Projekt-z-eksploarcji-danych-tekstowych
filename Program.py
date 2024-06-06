import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords

base_file = pd.read_csv(r"C:\Users\06mid\OneDrive\Pulpit\projekt eksploracja danych\Musical_instruments_reviews.csv")
base_file['reviewText'] = base_file['reviewText'].astype('string')
base_file['summary'] = base_file['summary'].astype('string')


base_file.info()
# print(base_file.head())

base_file = base_file.dropna()

def calculate_word_count(df, column):
    df['word_count'] = df[column].apply(lambda x: len(str(x).split(" ")))
    print(df[[column, 'word_count']].head())
    print(f'Mean: {df.word_count.mean()}')

def calculate_char_count(df, column):
    df['char_count'] = df[column].str.len()
    print(df[[column, 'char_count']].head())
    print(f'Mean: {df.char_count.mean()}')

def calculate_sentence_count(df, column):
    df['sentence'] = df[column].apply(lambda x: len([word for word in x.split() if word.endswith('.')]))
    print(df[[column, 'sentence']].iloc[:20])
    print(f'Mean: {df.sentence.mean()}')

def calculate_exclam_count(df, column):
    df['exclam'] = df[column].apply(lambda x: len([word for word in x.split() if word.endswith('!')]))
    print(df[[column, 'exclam']].iloc[:20])
    print(f'Mean: {df.exclam.mean()}')

# calculate_word_count(base_file, 'reviewText')
# calculate_char_count(base_file, 'summary')
# calculate_sentence_count(base_file, 'reviewText')
# calculate_exclam_count(base_file, 'reviewText')
#
#
#
nltk.download('stopwords')
stop = stopwords.words('english')

def display_stopwords_example():
    print(stop[:10])

def calculate_stopwords_count(df, column):
    df['stopwords'] = df[column].apply(lambda x: len([word for word in x.split() if word in stop]))
    print(df[[column, 'stopwords']].head())
    print(f'Mean: {df.stopwords.mean()}')

def display_reviewer_counts(df):
    review_counts = df['reviewerName'].value_counts()
    print(review_counts)

def plot_ratings_distribution(df, column):
    bins = [1, 2, 3, 4, 5, 6]
    df[column].hist(bins=bins, width=0.95, edgecolor='black', align='left')
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.xticks([1, 2, 3, 4, 5])
    plt.show()

# display_stopwords_example()
# calculate_stopwords_count(base_file, 'reviewText')
# display_reviewer_counts(base_file)
# plot_ratings_distribution(base_file, 'overall')
# #
def transform_helpful(row):
    positive, total = map(int, row.strip('[]').split(', '))
    return pd.Series([positive, total])

def apply_transform_helpful(df):
    df[['helpful_yes', 'helpful_total']] = df['helpful'].apply(transform_helpful)
    print(df[['helpful_yes', 'helpful_total']].describe())

def calculate_helpful_ratio(df):
    df['helpful_ratio'] = df['helpful_yes'] / df['helpful_total']
    print(df[['reviewText', 'helpful_ratio']])

def plot_helpful_ratio_distribution(df):
    df['helpful_ratio'] *= 100
    n, bins, _ = plt.hist(df['helpful_ratio'], bins=10, edgecolor='black', alpha=0.0, histtype='step')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    plt.plot(bin_centers, n, color='red', linestyle='-', marker='o')
    plt.title('Rozkład wskaźnika pomocności')
    plt.xlabel('Wskaźnik pomocności (%)')
    plt.ylabel('Częstotliwość')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# apply_transform_helpful(base_file)
# calculate_helpful_ratio(base_file)
# plot_helpful_ratio_distribution(base_file)

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

class TextDataAnalyzer:

    def __init__(self, data):
        self.data = data
        self.stemmer = PorterStemmer()
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        self.tfidf_matrix = None
        self.tfidf_features = None
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self):
        self.data['reviewText_processed'] = self.data['reviewText'].apply(
            lambda x: " ".join(word.lower() for word in x.split() if word.lower() not in self.stop_words))
        self.data['reviewText_processed'] = self.data['reviewText_processed'].apply(lambda x: " ".join(x.split()))


    def preprocess_and_analyze_frequency(self):
        self.preprocess_text()
        print("Normalized Text:")
        print(self.data['reviewText_processed'].head())

        freq = pd.Series(' '.join(self.data['reviewText_processed']).split()).value_counts()
        freq = freq[freq > 500]
        return freq[:10]

    def stem_text(self):
        self.data['StemmedText'] = self.data['reviewText'].apply(lambda x: ' '.join(self.stemmer.stem(word) for word in word_tokenize(x)))
        return self.data[['reviewText', 'StemmedText']]

    def lemmatize_text(self):
        self.data['LemmatizedText'] = self.data['reviewText'].apply(lambda x: ' '.join(token.lemma_ for token in self.nlp(x)))
        return self.data[['reviewText', 'LemmatizedText']]

    def vectorize_text(self, max_features=1000):
        tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
        self.tfidf_matrix = tfidf_vectorizer.fit_transform(self.data['reviewText_processed'])
        self.tfidf_features = tfidf_vectorizer.get_feature_names_out()
        dense_tfidf_matrix = self.tfidf_matrix.todense()
        tfidf_mean_values = np.mean(dense_tfidf_matrix, axis=0).tolist()[0]
        tfidf_mean_df = pd.DataFrame(list(zip(self.tfidf_features, tfidf_mean_values)), columns=['word', 'tfidf_mean'])
        tfidf_mean_df = tfidf_mean_df.sort_values(by='tfidf_mean', ascending=False)

        print("Top words by TF-IDF mean value:")
        print(tfidf_mean_df.head(10))

        print("\nTF-IDF Matrix (first 5 rows):")
        df_tfidf_matrix = pd.DataFrame(dense_tfidf_matrix, columns=self.tfidf_features)
        print(df_tfidf_matrix.head(25))


    def plot_distributions(self):
        self.data['word_count'] = self.data['reviewText'].apply(lambda x: len(x.split()))
        self.data['word_count'].hist(bins=[0, 10, 20, 30, 40, 50], edgecolor='black')
        plt.title('Distribution of Word Counts')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        plt.show()

    def analyze_sentiment(self):
        self.preprocess_text()
        self.data['sentiment_score'] = self.data['reviewText'].apply(lambda x: TextBlob(x).sentiment.polarity)
        return self.data[['reviewText', 'sentiment_score']].head(50)

    def analyze_sentiment_processed(self):
        self.preprocess_text()
        self.data['sentiment_score_processed'] = self.data['reviewText_processed'].apply(
            lambda x: TextBlob(x).sentiment.polarity)
        return self.data[['reviewText_processed', 'sentiment_score_processed']].head(50)

analyzer = TextDataAnalyzer(base_file)
#
# print("Proccessed Text Results:")
print(analyzer.preprocess_and_analyze_frequency())
# print("\nWord Count Distribution:")
# print(analyzer.plot_distributions())
# print("Vectorized Text Results:")
# print(analyzer.vectorize_text())
# print("Stemmed Text Results:")
# print(analyzer.stem_text())
# print("\nLemmatized Text Results:")
# print(analyzer.lemmatize_text())
print("Sentiment Analysis Results:")
print(analyzer.analyze_sentiment())
print("\nSentiment Analysis on Processed Text:")
print(analyzer.analyze_sentiment_processed())

import unittest
