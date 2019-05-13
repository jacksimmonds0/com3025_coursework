
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

Functionality is split into separate files. Preprocessing can be completed separately to training of the model to save time. 

### LSTM Sentiment Analysis
To run the preprocessing, run the following:
```
python lstm-preprocessing.py 'output_file' 'review_limit'
```
Where the quoted strings are required parameters.

To train the model, run the following:
```
python lstm-model.py 'input_file'
```
Where the `'input_file'` is the filename name generated from `lstm-preprocessing.py`.

### Visualisation
To run the preprocessing, run the following:
```
python visualisation-preprocessing.py 'output_file' 'review_limit'
```
To run the model, run the following:
```
python visualisation-model.py 'input_file' PCA 2D
python visualisation-model.py 'input_file' t-SNE 3D
```
This can run either model (PCA or t-SNE) to plot on either 2 or 3 dimensions 

