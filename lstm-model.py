import pickle
import argparse
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

#mac hack
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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

    X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size = 0.25, random_state = 1)

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
    model.add(LSTM(200))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train1, y_train1, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=10)

    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Test accuracy:", scores[1])
