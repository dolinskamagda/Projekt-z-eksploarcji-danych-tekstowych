import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
import plotly.graph_objs as go
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

base_file = pd.read_csv(r"C:\Users\06mid\OneDrive\Pulpit\projekt eksploracja danych\Musical_instruments_reviews.csv")
# print(base_file)

# base_file.info()
# print(base_file.head())

base_file = base_file.dropna()

base_file['reviewText'] = base_file['reviewText'].astype('string')


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

# import pandas as pd
# from textblob import TextBlob
#
# # Słownik mapujący tagi części mowy z TextBlob na kategorie
# pos_family = {
#     'noun': ['NN', 'NNS', 'NNP', 'NNPS'],
#     'pron': ['PRP', 'PRP$', 'WP', 'WP$'],
#     'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
#     'adj': ['JJ', 'JJR', 'JJS'],
#     'adv': ['RB', 'RBR', 'RBS', 'WRB']
# }
#
# # Funkcja do zliczania określonej części mowy w tekście
# def count_pos(text, part_of_speech):
#     count = 0
#     if text:
#         blob = TextBlob(str(text))
#         count = sum(1 for word, tag in blob.tags if tag in pos_family[part_of_speech])
#     return count
#
# # Zastosowanie funkcji do DataFrame
# for part in pos_family:
#     base_file[f'{part}_count'] = base_file['reviewText'].apply(lambda x: count_pos(x, part))
#
# # Wydrukuj wyniki
# print(base_file.head(20)[['reviewText'] + [f'{part}_count' for part in pos_family]].dropna())

# review_counts = base_file['reviewerName'].value_counts()
# print(review_counts)

# bins = [1, 2, 3, 4, 5, 6]
# base_file['overall'].hist(bins=bins, width=0.95, edgecolor='black', align='left')
# plt.title('Distribution of Ratings')
# plt.xlabel('Rating')
# plt.ylabel('Frequency')
# plt.xticks([1, 2, 3, 4, 5])
# plt.show()


def transform_helpful(row):
    # Usunięcie nawiasów kwadratowych i podział stringa na dwa elementy
    positive, total = map(int, row.strip('[]').split(', '))
    return pd.Series([positive, total])
#
# Stosowanie funkcji transformującej
base_file[['helpful_yes', 'helpful_total']] = base_file['helpful'].apply(transform_helpful)
#
# Wyświetlenie podstawowych statystyk
# print(base_file[['helpful_yes', 'helpful_total']].describe())
#
# Obliczenie stosunku pozytywnych reakcji do całkowitej liczby reakcji
base_file['helpful_ratio'] = base_file['helpful_yes'] / base_file['helpful_total']
# print(base_file[['reviewText', 'helpful_ratio']])

# base_file['helpful_ratio'] *= 100

# base_file['helpful_ratio'].hist(bins=10, edgecolor='black')
# plt.title('Distribution of Helpfulness Ratio')
# plt.xlabel('Helpfulness Ratio')
# plt.ylabel('Frequency')
# plt.show()

