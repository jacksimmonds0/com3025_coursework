from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class FeatureVector:

    def __init__(self, reviews):
        self.reviews = reviews

    def ngram_vectorizer(self, limit=-1):
        if limit != -1:
            reviews = self.reviews[:limit]
        else:
            reviews = self.reviews

        ngram_vectorizer_count = CountVectorizer(binary=True, ngram_range=(1,2))
        ngram_vectorizer_count.fit(reviews)
        
        return ngram_vectorizer_count.transform(reviews)


    def tfidf_vectorizer(self, limit=-1):
        if limit != -1:
            reviews = self.reviews[:limit]
        else:
            reviews = self.reviews

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectorizer.fit(reviews)
        
        return tfidf_vectorizer.transform(reviews)
        