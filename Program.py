import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
import plotly.graph_objs as go
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

base_file = pd.read_csv(r"C:\Users\06mid\OneDrive\Pulpit\projekt eksploracja danych\Musical_instruments_reviews.csv")
base_file['reviewText'] = base_file['reviewText'].astype('string')


# #Wstępna analiza danych
#
base_file.info()
# print(base_file.head())

base_file = base_file.dropna()
#
#
# base_file['word_count'] = base_file['reviewText'].apply(lambda x: len(str(x).split(" ")))
# print(base_file[['reviewText','word_count']].head())
# print(f'Mean: {base_file.word_count.mean()}')
#
# base_file['char_count'] = base_file['summary'].str.len()
# print(base_file[['summary','char_count']].head())
# print(f'Mean: {base_file.char_count.mean()}')
#
# base_file['sentence'] = base_file['reviewText'].apply(lambda x: len([x for x in x.split() if x.endswith('.')]))
# print(base_file[['reviewText','sentence']].iloc[0:].head(20))
# print(f'Mean: {base_file.sentence.mean()}')
#
# base_file['exclam'] = base_file['reviewText'].apply(lambda x: len([x for x in x.split() if x.endswith('!')]))
# print(base_file[['reviewText','exclam']].iloc[0:].head(20))
# print(f'Mean: {base_file.exclam.mean()}')
#
# import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')
# stop = stopwords.words('english')
# #
# print(stop[:10])
# #
# base_file['stopwords'] = base_file['reviewText'].apply(lambda x: len([x for x in x.split() if x in stop]))
# print(base_file[['reviewText','stopwords']].head())
# print(f'Mean: {base_file.stopwords.mean()}')
#
# review_counts = base_file['reviewerName'].value_counts()
# print(review_counts)
#
# bins = [1, 2, 3, 4, 5, 6]
# base_file['overall'].hist(bins=bins, width=0.95, edgecolor='black', align='left')
# plt.title('Distribution of Ratings')
# plt.xlabel('Rating')
# plt.ylabel('Frequency')
# plt.xticks([1, 2, 3, 4, 5])
# # plt.show()
# #
# #
# def transform_helpful(row):
#     # Usunięcie nawiasów kwadratowych i podział stringa na dwa elementy
#     positive, total = map(int, row.strip('[]').split(', '))
#     return pd.Series([positive, total])
# #
# # Stosowanie funkcji transformującej
# base_file[['helpful_yes', 'helpful_total']] = base_file['helpful'].apply(transform_helpful)
# #
# # Wyświetlenie podstawowych statystyk
# print(base_file[['helpful_yes', 'helpful_total']].describe())
# #
# # Obliczenie stosunku pozytywnych reakcji do całkowitej liczby reakcji
# base_file['helpful_ratio'] = base_file['helpful_yes'] / base_file['helpful_total']
# print(base_file[['reviewText', 'helpful_ratio']])
# #
# base_file['helpful_ratio'] *= 100
#
# base_file['helpful_ratio'].hist(bins=10, edgecolor='black')
# plt.title('Distribution of Helpfulness Ratio')
# plt.xlabel('Helpfulness Ratio')
# plt.ylabel('Frequency')
# plt.show()

# # #Normalizacja tekstu
# base_file['reviewText'] = base_file['reviewText'].apply(lambda x: " ".join(x.lower() for x in x.split()))
# print(base_file['reviewText'].head())
# # #
# # Usuwanie nadmiarowych spacji bezpośrednio w DataFrame
# base_file['reviewText'] = base_file['reviewText'].apply(lambda x: " ".join(x.split()))
#
# print(base_file['reviewText'].head(10))
#
# stop = stopwords.words('english')
# base_file['reviewText'] = base_file['reviewText'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
# print(base_file['reviewText'].head())
#
# freq = pd.Series(' '.join(base_file['reviewText']).split()).value_counts()
# freq = freq[freq > 500]
# print(freq[:10])
# # #
#
# from nltk.stem import PorterStemmer
# from nltk.tokenize import word_tokenize
# import spacy
#
# # Pobranie niezbędnych zasobów
# nltk.download('punkt')
#
# stemmer = PorterStemmer()
# nlp = spacy.load('en_core_web_sm')
#
# def stem_text(text):
#     words = word_tokenize(text)
#     return ' '.join(stemmer.stem(word) for word in words)
#
# def lemmatize_text(text):
#     doc = nlp(text)
#     return ' '.join(token.lemma_ for token in doc)
#
# base_file['StemmedText'] = base_file['reviewText'].apply(stem_text)
# base_file['LemmatizedText'] = base_file['reviewText'].apply(lemmatize_text)
#
# Wyświetlenie pierwszych kilku wierszy dla podglądu przetworzonych danych
# print(base_file[['reviewText', 'StemmedText']].head())
# print(base_file[['reviewText', 'LemmatizedText']].head())
#
# # Wektoryzacja tekstu
# from sklearn.feature_extraction.text import TfidfVectorizer
#
#
# tfidf_vectorizer = TfidfVectorizer(max_features=1000)
# tfidf_matrix = tfidf_vectorizer.fit_transform(base_file['reviewText'])
# terms = tfidf_vectorizer.get_feature_names_out()
# tfidf_df = pd.DataFrame(data=tfidf_matrix.toarray(), columns=terms)
# print(tfidf_df.head(30))

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import matplotlib.pyplot as plt

class TextDataAnalyzer:
    def __init__(self, data):
        self.data = data
        self.stemmer = PorterStemmer()
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])  # Załadowanie tylko niezbędnych komponentów
        self.tfidf_matrix = None
        self.tfidf_features = None
        nltk.download('punkt')  # Pobieranie niezbędnych zasobów NLTK

    def preprocess_text(self):
        # Normalizacja tekstu
        self.data['reviewText'] = self.data['reviewText'].apply(lambda x: " ".join(x.lower() for x in x.split()))
        # Usuwanie nadmiarowych spacji
        self.data['reviewText'] = self.data['reviewText'].apply(lambda x: " ".join(x.split()))
        # Usuwanie stopwords
        stop = stopwords.words('english')
        self.data['reviewText'] = self.data['reviewText'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    def stem_text(self):
        # Stosowanie stemmingu na tekście
        self.data['StemmedText'] = self.data['reviewText'].apply(lambda x: ' '.join(self.stemmer.stem(word) for word in word_tokenize(x)))
        print(self.data[['reviewText', 'StemmedText']].head())

    def lemmatize_text(self):
        # Stosowanie lematyzacji na tekście
        self.data['LemmatizedText'] = self.data['reviewText'].apply(lambda x: ' '.join(token.lemma_ for token in self.nlp(x)))
        print(self.data[['reviewText', 'LemmatizedText']].head())

    def vectorize_text(self):
        # Wektoryzacja tekstu
        tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        self.tfidf_matrix = tfidf_vectorizer.fit_transform(self.data['reviewText'])
        self.tfidf_features = tfidf_vectorizer.get_feature_names_out()

    def analyze_frequency(self):
        # Analiza częstotliwości słów
        freq = pd.Series(' '.join(self.data['reviewText']).split()).value_counts()
        freq = freq[freq > 500]
        print(freq[:10])

    def plot_distributions(self):
        # Rysowanie dystrybucji
        self.data['word_count'] = self.data['reviewText'].apply(lambda x: len(x.split()))
        self.data['word_count'].hist(bins=[0, 10, 20, 30, 40, 50], edgecolor='black')
        plt.title('Distribution of Word Counts')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        plt.show()

# Użycie klasy
if __name__ == '__main__':
    data =  base_file # Ustal odpowiednią ścieżkę do plik
    analyzer = TextDataAnalyzer(data)
    # analyzer.preprocess_text()
    # analyzer.stem_text()
    # analyzer.lemmatize_text()
    # analyzer.analyze_frequency()
    analyzer.vectorize_text()
    analyzer.plot_distributions()

