import re
import gzip

import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd

REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def get_df(path, limit=-1):
    i = 0
    df = {}
    for d in parse(path):
        if limit != -1 and i > limit:
            return pd.DataFrame.from_dict(df, orient='index')

        df[i] = d
        i += 1

    return pd.DataFrame.from_dict(df, orient='index')

def filter_and_combine(df_reviews, df_meta):
    df_reviews = df_reviews.drop(columns=['unixReviewTime'])
    df_meta = df_meta.drop(columns=['imUrl', 'brand', 'salesRank'])
    combined = df_meta.join(df_reviews.set_index('asin'), on ='asin')

    # remove any reviews that are null
    combined_filtered = combined.dropna(subset=['reviewText'])

    # duplicate column to preserve original
    combined_filtered['reviewTextProcessed'] = combined_filtered['reviewText']
    
    return combined_filtered

def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    
    return reviews

def remove_stop_words(corpus):
    english_stop_words = stopwords.words('english')
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
            ' '.join([word for word in review.split() 
                      if word not in english_stop_words])
        )

    return removed_stop_words

def get_stemmed_text(corpus):
    return [' '.join([PorterStemmer().stem(word) for word in review.split()]) for review in corpus]

def vectorizer(reviews, limit=-1):
    if limit != -1:
        reviews = reviews[:limit]

    ngram_vectorizer_count = CountVectorizer(binary=True, ngram_range=(1,2))
    ngram_vectorizer_count.fit(reviews)
    X_count = ngram_vectorizer_count.transform([reviews[0]])
    ##print(X_count)

    ngram_vectorizer_tfidf = TfidfVectorizer()
    ngram_vectorizer_tfidf.fit(reviews)
    X_tfidf = ngram_vectorizer_tfidf.transform([reviews[0]])
    print(X_tfidf)

if __name__ == '__main__':

    df_reviews = get_df('input/reviews_Electronics.json.gz', limit=20000)
    df_meta = get_df('input/meta_Electronics.json.gz')

    combined = filter_and_combine(df_reviews, df_meta)
    reviews_clean = preprocess_reviews(combined['reviewTextProcessed'].tolist())
    no_stop_words = remove_stop_words(reviews_clean)
    stemmed_reviews = get_stemmed_text(no_stop_words)

    vectorizer(stemmed_reviews, limit=5000)




