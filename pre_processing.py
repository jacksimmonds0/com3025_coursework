import re
import gzip

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd

class PreProcessing:

    REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    def __init__(self, limit_reviews=-1, limit_meta=-1):
        self.df_reviews = self.get_df('input/reviews_Electronics.json.gz', limit=limit_reviews)
        self.df_meta = self.get_df('input/meta_Electronics.json.gz', limit=limit_meta)

    def parse(self, path):
        g = gzip.open(path, 'rb')
        for l in g:
            yield eval(l)

    def get_df(self, path, limit=-1):
        i = 0
        df = {}
        for d in self.parse(path):
            if limit != -1 and i > limit:
                return pd.DataFrame.from_dict(df, orient='index')

            df[i] = d
            i += 1

        return pd.DataFrame.from_dict(df, orient='index')

    def filter_and_combine(self, df_reviews, df_meta):
        df_reviews = df_reviews.drop(columns=['unixReviewTime'])
        df_meta = df_meta.drop(columns=['imUrl', 'brand', 'salesRank'])
        combined = df_meta.join(df_reviews.set_index('asin'), on ='asin')

        # remove any reviews that are null
        combined_filtered = combined.dropna(subset=['reviewText'])

        # duplicate column to preserve original
        combined_filtered['reviewTextProcessed'] = combined_filtered['reviewText']
        
        return combined_filtered

    def preprocess_reviews(self, reviews):
        reviews = [PreProcessing.REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
        reviews = [PreProcessing.REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
        
        return reviews

    def remove_stop_words(self, corpus):
        english_stop_words = stopwords.words('english')
        removed_stop_words = []
        for review in corpus:
            removed_stop_words.append(
                ' '.join([word for word in review.split() 
                        if word not in english_stop_words])
            )

        return removed_stop_words

    def get_stemmed_text(self, corpus):
        return [' '.join([PorterStemmer().stem(word) for word in review.split()]) for review in corpus]

    def change_categories_column(self, df):
        categories = df['categories'].tolist()
        single_categories = []

        for cats in categories:
            try:
                single_categories.append(cats[0][2])
            except IndexError:
                single_categories.append('Electronics')

        # replace the categories column with a single category for each item
        # arbitrarily taking the 2nd category
        df['categories'] = single_categories
        
        return df


    def get_df_reviews(self):
        return self.df_reviews

    def get_df_meta(self):
        return self.df_meta

