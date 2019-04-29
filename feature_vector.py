from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import gensim
import numpy as np
import math

class FeatureVector:

    def __init__(self, reviews, limit=-1):
        self.reviews = reviews

        if limit != -1:
            self.reviews = self.reviews[:limit]
        else:
            self.reviews = self.reviews


    def ngram_vectorizer(self, limit=-1):

        ngram_vectorizer_count = CountVectorizer(binary=True, ngram_range=(1,2))
        ngram_vectorizer_count.fit(self.reviews)
        
        return ngram_vectorizer_count.transform(self.reviews)


    def tfidf_vectorizer(self, limit=-1):

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectorizer.fit(self.reviews)
        
        return tfidf_vectorizer.transform(self.reviews)


    def word2vec_gensim(self):
        list_of_sent = self.get_list_of_sent(self.reviews)
        w2v_model = gensim.models.Word2Vec(list_of_sent, min_count=5, size=50, workers=4)

        return w2v_model


    def get_list_of_sent(self, reviews):
        list_of_sent = []

        for sent in reviews:
            filtered_sentence = []
            for w in sent.split():        
                if(w.isalpha()):    
                    filtered_sentence.append(w.lower())

            list_of_sent.append(filtered_sentence)

        return list_of_sent


    def avg_review_word_vector(self, list_of_sent, w2v_model, y_list):
        sent_vectors = []
        count = 0
        for sent in list_of_sent:
            sent_vec = np.zeros(50)
            count_words = len(sent)

            if count_words == 0:
                y_list.pop(count)
                count += 1
                continue

            for word in sent:
                try:
                    vec = w2v_model.wv[word]
                    sent_vec += vec
                except:
                    pass

            sent_vec /= count_words
            sent_vectors.append(sent_vec)          
            count += 1

        return np.array(sent_vectors), np.array(y_list)
        
        