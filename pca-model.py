import pickle
import argparse
import os
import pandas as pd
import numpy as np
from feature_vector import FeatureVector

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

    pca = SparsePCA(n_components=2)
    X_r = pca.fit(X).transform(X)

    plt.figure()
    colors = ['red', 'navy', 'turquoise', 'darkorange', 'green']
    lw = 2

    for color, target_name in zip(colors, list(set(y))):
        plt.scatter(X_r[y == target_name, 0], X_r[y == target_name, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)

    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of Amazon dataset')

    plt.show()