from pre_processing import PreProcessing
from feature_vector import FeatureVector


if __name__ == '__main__':
    
    # step 1 - pre processing the training data
    # convert to combined pandas dataframe
    # remving stopwords and stemming the review text
    pre_processing = PreProcessing(limit_reviews=20000)

    df_reviews = pre_processing.get_df_reviews()
    df_meta = pre_processing.get_df_meta()

    combined = pre_processing.filter_and_combine(df_reviews, df_meta)
    reviews_clean = pre_processing.preprocess_reviews(combined['reviewTextProcessed'].tolist())
    no_stop_words = pre_processing.remove_stop_words(reviews_clean)
    stemmed_reviews = pre_processing.get_stemmed_text(no_stop_words)


    # step 2 - converting to feature vectors
    # using either ngrams or tf-idf
    feature_vector = FeatureVector(stemmed_reviews)
    X_tfidf = feature_vector.tfidf_vectorizer(limit=5000)
    print(X_tfidf[0])


    # step 3 - implement algorithms
    # either PCA or t-SNE??