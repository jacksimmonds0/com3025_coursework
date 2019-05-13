import pickle
import argparse
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras import backend as K
from sklearn.metrics import roc_curve, auc
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

#mac hack
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("output_file", help="the filename for the output model")
    parser.add_argument("input_file", help="the processed dataframe (reviews and sentiment) filename")
    args = parser.parse_args()

    # retrieve preprocessed list from file
    print("Attempting to use preprocessed dataframe (reviws and sentiment) from file:" + args.input_file)
    if not os.path.exists(args.input_file):
        raise Exception("File does not exist - please specify a valid file.")
    else:
        with open (args.input_file, 'rb') as fp:
            reviews_and_sentiment = pickle.load(fp)


    #more work to be done, comments needed
    X, y = (reviews_and_sentiment['reviewTextProcessed'].values, reviews_and_sentiment['sentiment'].values)

    tk = Tokenizer(lower = True)
    tk.fit_on_texts(X)
    X_seq = tk.texts_to_sequences(X)
    X_pad = pad_sequences(X_seq, maxlen=100, padding='post')

    X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size = 0.25, random_state = 4)

    batch_size = 64
    X_train1 = X_train[batch_size:]
    y_train1 = y_train[batch_size:]
    X_valid = X_train[:batch_size]
    y_valid = y_train[:batch_size]

    vocabulary_size = len(tk.word_counts.keys())+1
    max_words = 100
    embedding_size = 32
    model = Sequential()
    model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(200))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train1, y_train1, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=4)

    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    y_probs = model.predict(X_test)
    #skplt.metrics.plot_roc(y_test, y_probs)
    #plt.show()

    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

    #pickle and save the model file
    with open(args.output_file, 'wb') as fp:
        pickle.dump(model, fp)
