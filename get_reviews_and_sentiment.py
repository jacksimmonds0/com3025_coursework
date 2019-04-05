import pickle
import argparse
import pandas as pd
from pre_processing import PreProcessing
from feature_vector import FeatureVector
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
import os

#mac hack
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("review_limit", help="the number of reviews to be processed")
    args = parser.parse_args()
    
    try:
        review_limit = int(args.review_limit)
    except ValueError:
        raise Exception("Review limit must be a number")

    if review_limit < 100:
        raise Exception("Review limit must be over 100")
    # step 1 - pre processing the training data
    # convert to combined pandas dataframe
    # remving stopwords and stemming the review text
    pre_processing = PreProcessing(limit_reviews=review_limit)

    df_reviews = pre_processing.get_df_reviews()
    df_meta = pre_processing.get_df_meta()

    combined = pre_processing.filter_and_combine(df_reviews, df_meta)

    reviews_and_sentiment = combined[['reviewTextProcessed', 'overall']]

    # convert string values to numerical values
    reviews_and_sentiment['overall'] = pd.to_numeric(reviews_and_sentiment['overall'])

    # convert the rating value to 1 or 0
    # if the average rating is 1, 2, 3 then 0 (negative sentiment)
    # if the average rating is 4 or 5 then 1 (positive sentiment)
    reviews_and_sentiment['sentiment'] = reviews_and_sentiment['overall'].apply(lambda x: 1 if x>3 else 0)
    reviews_and_sentiment['sentiment'] = [1 if x > 3 else 0 for x in reviews_and_sentiment['overall']]

    print(reviews_and_sentiment)

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