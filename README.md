# COM3025 Deep Learning and Advanced AI Coursework
## Required files
In order to be able to compile the code, two external files will need to be acquired from the following links:
http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics.json.gz
http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz

These two files must be imported into the input folder.

## Running the code
In order to run the code it is preferable to be in a virtual environment i.e.
```
source activate Labs_Env
```

Functionality is split into separate files. Preprocessing can be completed separately to training of the model to save time. To run the preprocessing, run the following:
```
python get_processed_reviews.py 'output_file' 'review_limit'
```
Where the quoted strings are required parameters.

To train the model, run the following:
```
python app.py 'input_file'
```
Where the 'input_file' is the filename name generated from get_processed_reviews.py.

You can also access Jupyter Notebook through the command:
```
jupyter notebook
```