import pickle
import argparse
import pandas as pd
from pre_processing import PreProcessing

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("output_file", help="the processed review list filename")
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

    # convert string rating values to numerical values
    reviews_and_sentiment['overall'] = pd.to_numeric(reviews_and_sentiment['overall'])

    # convert the rating value to 1 or 0 (sentiment value)
    # if the average rating is 1, 2, 3 then 0 (negative sentiment)
    # if the average rating is 4 or 5 then 1 (positive sentiment)
    reviews_and_sentiment['sentiment'] = reviews_and_sentiment['overall'].apply(lambda x: 1 if x>3 else 0)
    reviews_and_sentiment['sentiment'] = [1 if x > 3 else 0 for x in reviews_and_sentiment['overall']]

    print(reviews_and_sentiment)

    #pickle and save the preprocessed dataframe to file
    with open(args.output_file, 'wb') as fp:
        pickle.dump(reviews_and_sentiment, fp)