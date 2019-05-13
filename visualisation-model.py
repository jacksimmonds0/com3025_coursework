import pickle
import argparse
import os
import pandas as pd
import numpy as np
from feature_vector import FeatureVector
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="the processed review list filename")
    parser.add_argument("type", help="the type of model to visualise: PCA or t-SNE")
    parser.add_argument("dimensions", help="the number of dimensions to use: 2D or 3D")
    args = parser.parse_args()

    # retrieve processed list from file
    print("Attempting to use preprocessed review list from file: " + args.input_file)
    print("Using the model " + args.type + " in " + args.dimensions)
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
    #common_cats = combined.categories.value_counts().nlargest(5).to_frame('count').rename_axis('categories').reset_index()
    common_cats = pd.DataFrame(['Digital Cameras', 'Routers', 'Radios', 'Video', 'Blank Media'], columns=['categories'])
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


    # step 3 - implement algorithms
    # either PCA or t-SNE
    if args.dimensions == '2D':
        
        if args.type == 'PCA':
            pca = PCA(n_components=2)
            X_r = pca.fit(X).transform(X)
        elif args.type == 't-SNE':
            pca = PCA(n_components=2)
            X_r = pca.fit(X).transform(X)

        plt.figure()
        colors = ['red', 'navy', 'turquoise', 'darkorange', 'green']
        lw = 2

        for color, target_name in zip(colors, list(set(y))):
            plt.scatter(X_r[y == target_name, 0], X_r[y == target_name, 1], color=color, alpha=.8, lw=lw,
                        label=target_name)

        plt.legend(loc='best', shadow=False, scatterpoints=1)

    elif arg.dimensions == '3D':
        
        if args.type == 'PCA':
            pca = PCA(n_components=3)
            X_r = pca.fit(X).transform(X)
        elif args.type == 't-SNE':
            tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
            tsne_results = tsne.fit_transform(X)

        # change categories to integers for colouring the points
        df = pd.DataFrame()
        df['y'] = y
        df['label'] = df['y'].apply(lambda i: str(i))
        df['y'] = df.y.astype("category").cat.codes
        rndperm = np.random.permutation(df.shape[0])

        df['pca-one'] = X_r[:,0]
        df['pca-two'] = X_r[:,1] 
        df['pca-three'] = X_r[:,2]
        ax = plt.figure(figsize=(16,10)).gca(projection='3d')

        ax.scatter(
            xs=df.loc[rndperm,:]["pca-one"], 
            ys=df.loc[rndperm,:]["pca-two"], 
            zs=df.loc[rndperm,:]["pca-three"], 
            c=df.loc[rndperm,:]["y"], 
            cmap='gist_rainbow'
        )

        ax.set_xlabel('pca-one')
        ax.set_ylabel('pca-two')
        ax.set_zlabel('pca-three')
        plt.legend(loc='best', shadow=False, scatterpoints=1)


    plt.title(args.type + ' of Amazon Reviews Dataset')
    plt.show()