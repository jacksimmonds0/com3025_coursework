import pickle
import argparse
import os
from feature_vector import FeatureVector


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="the processed review list filename")
    args = parser.parse_args()

    # retrieve processed list from file
    print("Attempting to use preprocessed review list from file: " + args.input_file)
    if not os.path.exists(args.input_file):
        raise Exception("File does not exist - please specify a valid file.")
    else:
        combined = pd.read_csv(args.input_file, sep='\t')
        combined = combined.dropna()

    # step 2 - converting to feature vectors
    # using either ngrams or tf-idf
    feature_vector = FeatureVector(combined['reviewTextProcessed'].tolist())
    X = feature_vector.tfidf_vectorizer().toarray()
    y = np.array(combined['overall'].tolist())

    # step 3 - implement algorithms
    # either PCA or t-SNE??