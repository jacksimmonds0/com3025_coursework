import pickle
import argparse
import os
from pre_processing import PreProcessing
from feature_vector import FeatureVector


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="the processed review list filename")
    args = parser.parse_args()

    # retrieve processed list from file
    print("Attempting to use preprocessed reviews from file:" + args.input_file)
    if not os.path.exists(args.input_file):
        raise Exception("File does not exist - please specify a valid file.")
    else:
        with open (args.input_file, 'rb') as fp:
            stemmed_reviews = pickle.load(fp)

    # step 2 - converting to feature vectors
    # using either ngrams or tf-idf
    feature_vector = FeatureVector(stemmed_reviews)
    X_tfidf = feature_vector.tfidf_vectorizer(limit=5000)
    print(X_tfidf[0])


    # step 3 - implement algorithms
    # either PCA or t-SNE??