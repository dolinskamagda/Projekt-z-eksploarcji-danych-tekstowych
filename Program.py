import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords

base_file = pd.read_csv(r"C:\Users\06mid\OneDrive\Pulpit\projekt eksploracja danych\Musical_instruments_reviews.csv")
base_file['reviewText'] = base_file['reviewText'].astype('string')
base_file['summary'] = base_file['summary'].astype('string')
base_file = base_file.dropna()

# base_file.info()
# print(base_file.head())



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
    # Definiowanie przedziałów
    bins = [0, 5, 10, 20, 30, 66]
    labels = ['0-5', '5-10', '10-20', '20-30', '30-66']
    review_counts_bins = pd.cut(review_counts, bins=bins, labels=labels, right=False)

    # Zliczanie użytkowników w każdym przedziale
    review_counts_grouped = review_counts_bins.value_counts().sort_index()

    new_order = ['10-20', '0-5', '5-10', '30-66', '20-30']
    review_counts_grouped = review_counts_grouped.reindex(new_order)

    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(review_counts_grouped, autopct='%1.1f%%', startangle=140, pctdistance=0.85,
                                      textprops=dict(color="w"))

    # Ustawienia dla tekstu procentowego na zewnątrz
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(12)  # Zmniejszenie rozmiaru czcionki

    # Przesunięcie wartości procentowych na zewnątrz z liniami prowadzącymi
    for i, a in enumerate(autotexts):
        x, y = a.get_position()
        if labels[i] in ['20-30']:
            a.set_position((1.1 * x, 1.1 * y))  # Większe przesunięcie na zewnątrz dla '30-40' i '40-66'
        else:
            a.set_position((1.4 * x, 1.4 * y))  # Mniejsze przesunięcie dla pozostałych

    # Dodanie legendy
    ax.legend(wedges, labels, title="Comment Ranges", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Number of Users per Comment Range')
    plt.ylabel('')
    plt.show()


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
plot_ratings_distribution(base_file, 'overall')
# #
def transform_helpful(row):
    positive, total = map(int, row.strip('[]').split(', '))
    return pd.Series([positive, total])

def apply_transform_helpful(df):
    df[['helpful_yes', 'helpful_total']] = df['helpful'].apply(transform_helpful)
    # print(df[['helpful_yes', 'helpful_total']].describe())

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


from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
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
        self.stop_words = list(stopwords.words('english'))

    def preprocess_text(self):
        self.data['reviewText_processed'] = self.data['reviewText'].apply(
            lambda x: " ".join(word.lower() for word in x.split() if word.lower() not in self.stop_words))
        self.data['reviewText_processed'] = self.data['reviewText_processed'].apply(lambda x: " ".join(x.split()))
        self.data['reviewText_processed'] = self.data['reviewText_processed'].str.replace('[^\w\s]', '', regex=True)

    def preprocess_and_analyze_frequency(self):
        self.preprocess_text()
        print("Normalized Text:")
        print(self.data['reviewText_processed'].head())

        freq = pd.Series(' '.join(self.data['reviewText_processed']).split()).value_counts()
        freq = freq[freq > 500]
        return freq[:10]

    def stem_text(self):
        self.data['StemmedText'] = self.data['reviewText'].apply(
            lambda x: ' '.join(self.stemmer.stem(word) for word in word_tokenize(x)))
        result = self.data[['reviewText', 'StemmedText']]
        return result

    def lemmatize_text(self):
        self.data['LemmatizedText'] = self.data['reviewText'].apply(
            lambda x: ' '.join(token.lemma_ for token in self.nlp(x)))
        result = self.data[['reviewText', 'LemmatizedText']]
        return result

    def vectorize_text(self, max_features=1000):
        tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
        self.tfidf_matrix = tfidf_vectorizer.fit_transform(self.data['reviewText_processed'])
        self.tfidf_features = tfidf_vectorizer.get_feature_names_out()
        dense_tfidf_matrix = self.tfidf_matrix.todense()
        tfidf_mean_values = np.mean(dense_tfidf_matrix, axis=0).tolist()[0]
        tfidf_mean_df = pd.DataFrame(list(zip(self.tfidf_features, tfidf_mean_values)), columns=['word', 'tfidf_mean'])
        tfidf_mean_df = tfidf_mean_df.sort_values(by='tfidf_mean', ascending=False)
        print("\nTF-IDF Matrix (first 5 rows):")
        df_tfidf_matrix = pd.DataFrame(dense_tfidf_matrix, columns=self.tfidf_features)
        print(df_tfidf_matrix.head(25))
        print("Top words by TF-IDF mean value:")

        return tfidf_mean_df.head(10)

    def plot_distributions(self):
        self.data['word_count'] = self.data['reviewText'].apply(lambda x: len(x.split()))
        self.data['word_count'].hist(bins=[0, 10, 20, 30, 40, 50], edgecolor='black')
        plt.title('Distribution of Word Counts')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        plt.show()

    def compare_frequency_and_tfidf(self):
        freq = self.preprocess_and_analyze_frequency()
        tfidf = self.vectorize_text()

        comparison_df = pd.DataFrame({
            'Most Frequent Words': freq.index,
            'Frequency': freq.values,
            'Top TF-IDF Words': tfidf['word'].values,
            'TF-IDF Score': tfidf['tfidf_mean'].values
        })

        print(comparison_df)
        fig, ax = plt.subplots(figsize=(14, 7))
        bar1 = ax.bar(comparison_df['Most Frequent Words'], comparison_df['Frequency'], width=0.4, align='center',
                      label='Frequency')
        bar2 = ax.bar(comparison_df['Top TF-IDF Words'], comparison_df['TF-IDF Score'], width=0.4, align='edge',
                      label='TF-IDF Score')

        for bar in bar1:
            yval = bar.get_height()
        for bar in bar2:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, round(yval, 4), ha='left', va='bottom')

        ax.set_xlabel('Words')
        ax.set_ylabel('Counts / TF-IDF Score')
        ax.set_title('Comparison of Most Frequent Words and Top TF-IDF Words')
        ax.legend()
        plt.xticks(rotation=45)
        plt.show()

    def visualize_top_words_wordcloud(self, max_features=1000, ngram_range=(1, 2)):

        bow = CountVectorizer(max_features=max_features, lowercase=True, ngram_range=ngram_range, analyzer="word",
                              stop_words=self.stop_words)
        bow.fit(self.data['reviewText_processed'])
        word_counts = bow.transform(self.data['reviewText_processed'])
        word_counts_sum = word_counts.sum(axis=0)
        word_freq_df = pd.DataFrame(word_counts_sum, columns=bow.get_feature_names_out()).T
        word_freq_df.columns = ['Frequency']
        top_words_df = word_freq_df.sort_values(by='Frequency', ascending=False)
        wordcloud = WordCloud(width=1600, height=800, max_words=max_features,
                              background_color='white').generate_from_frequencies(top_words_df['Frequency'])
        plt.figure(figsize=(20, 20))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Top {max_features} Most Frequent Words and Bigrams WordCloud')
        plt.show()


analyzer = TextDataAnalyzer(base_file)
analyzer.preprocess_text()
# print(analyzer.preprocess_and_analyze_frequency())
# print("\nWord Count Distribution:")
# print(analyzer.plot_distributions())
# print("Vectorized Text Results:")
# print(analyzer.vectorize_text())
# analyzer.compare_frequency_and_tfidf()
# analyzer.visualize_top_words_wordcloud()
# print("Stemmed Text Results:")
# print(analyzer.stem_text())
# print("\nLemmatized Text Results:")
# print(analyzer.lemmatize_text())

class SentimentAnalyzer:

    def __init__(self, data):
        self.data = data

    def analyze_sentiment(self):
        self.data['sentiment_score'] = self.data['reviewText'].apply(lambda x: TextBlob(x).sentiment.polarity)
        return self.data[['reviewText', 'sentiment_score']]

    def analyze_sentiment_processed(self):
        self.data['sentiment_score_processed'] = self.data['reviewText_processed'].apply(
            lambda x: TextBlob(x).sentiment.polarity)
        return self.data[['reviewText_processed', 'sentiment_score_processed']]

    def assign_sentiment_category(self):
        self.data['sentiment_category'] = self.data['sentiment_score'].apply(
            lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
        self.data['sentiment_category_processed'] = self.data['sentiment_score_processed'].apply(
            lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))

    def plot_sentiment_comparison(self):
        sentiment_counts = self.data['sentiment_category'].value_counts().reindex(
            ['positive', 'neutral', 'negative']).fillna(0)
        sentiment_counts_processed = self.data['sentiment_category_processed'].value_counts().reindex(
            ['positive', 'neutral', 'negative']).fillna(0)

        fig, ax = plt.subplots(figsize=(10, 7))
        index = ['positive', 'neutral', 'negative']
        bar_width = 0.35
        bar_x = np.arange(len(index))

        bar1 = ax.bar(bar_x - bar_width / 2, sentiment_counts, bar_width, label='Original')
        bar2 = ax.bar(bar_x + bar_width / 2, sentiment_counts_processed, bar_width, label='Processed')

        for rect in bar1 + bar2:
            height = rect.get_height()
            ax.annotate('{}'.format(int(height)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Counts')
        ax.set_title('Sentiment Comparison')
        ax.set_xticks(bar_x)
        ax.set_xticklabels(index)
        ax.legend()

        plt.show()


# sentiment_analyzer = SentimentAnalyzer(base_file)
# sentiment_analyzer.analyze_sentiment()
# sentiment_analyzer.analyze_sentiment_processed()
# sentiment_analyzer.assign_sentiment_category()
# sentiment_analyzer.plot_sentiment_comparison()



# import unittest
#
# class TestTextDataAnalyzer(unittest.TestCase):
#
#     def setUp(self):
#         self.df = base_file.copy()
#         self.analyzer = TextDataAnalyzer(self.df)
#
#     def test_preprocess_and_analyze_frequency(self):
#         freq = self.analyzer.preprocess_and_analyze_frequency()
#         self.assertGreaterEqual(len(freq), 3)
#
#     def test_stem_text(self):
#         stemmed_df = self.analyzer.stem_text()
#         self.assertEqual(len(stemmed_df), len(self.df))
#
#     def test_vectorize_text(self):
#         self.analyzer.vectorize_text()
#         self.assertIsNotNone(self.analyzer.tfidf_matrix)
#
#     def test_plot_distributions(self):
#         try:
#             self.analyzer.plot_distributions()
#         except Exception as e:
#             self.fail(f"plot_distributions() raised an exception: {e}")
#
# if __name__ == '__main__':
#     unittest.main()

