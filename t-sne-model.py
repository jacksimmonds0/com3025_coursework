import pickle
import argparse
import os
import pandas as pd
import numpy as np
from feature_vector import FeatureVector
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
    # using either ngrams or tf-idf or own word vectors (using gensim and word2vec)
    feature_vector = FeatureVector(combined['reviewTextProcessed'].tolist())

    # train the word vectors on the entire training set corpus
    w2v_model = feature_vector.word2vec_gensim()

    # only plot the top 5 categories
    common_cats = combined.categories.value_counts().nlargest(5).to_frame('count').rename_axis('categories').reset_index()
    combined = pd.merge(combined, common_cats, how='inner', on=['categories'])

    list_of_sent = feature_vector.get_list_of_sent(combined['reviewTextProcessed'].tolist())
    X, y = feature_vector.avg_review_word_vector(list_of_sent, w2v_model, combined['categories'].tolist())

    # taking a subset of 2000 reviews, of the same percentage for each category to plot
    # difficult to visualise with >2000 reviews plotted
    perc = 1 - (2000 / len(X))
    splits = StratifiedShuffleSplit(n_splits=1, test_size=perc, random_state=0)
    for train_index, test_index in splits.split(X, y):
        X, X_test = X[train_index], X[test_index]
        y, y_test = y[train_index], y[test_index]


    # feat_cols = X
    # df['feat_cols'] = X
    df = pd.DataFrame()
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))
    df['y'] = df.y.astype("category").cat.codes
    rndperm = np.random.permutation(df.shape[0])


    # step 3 - implement algorithms
    # either PCA or t-SNE?
    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(X)

    df['tsne-one'] = tsne_results[:,0]
    df['tsne-two'] = tsne_results[:,1] 
    df['tsne-three'] = tsne_results[:,2]
    ax = plt.figure(figsize=(16,10)).gca(projection='3d')

    ax.scatter(
        xs=df.loc[rndperm,:]["tsne-one"], 
        ys=df.loc[rndperm,:]["tsne-two"], 
        zs=df.loc[rndperm,:]["tsne-three"], 
        c=df.loc[rndperm,:]["y"], 
        cmap='gist_rainbow'
    )


    ax.set_xlabel('tsne-one')
    ax.set_ylabel('tsne-two')
    ax.set_zlabel('tsne-three')
    plt.title('t-SNE of Amazon Reviews Dataset')

    plt.show()