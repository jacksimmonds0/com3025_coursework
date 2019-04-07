import pickle
import argparse
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
    reviews_clean = pre_processing.preprocess_reviews(combined['reviewTextProcessed'].tolist())
    no_stop_words = pre_processing.remove_stop_words(reviews_clean)
    stemmed_reviews = pre_processing.get_stemmed_text(no_stop_words)


    #pickle the list of preprocessed reviews to file
    with open(args.output_file, 'wb') as fp:
        pickle.dump(stemmed_reviews, fp)