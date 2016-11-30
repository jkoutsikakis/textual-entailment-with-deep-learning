# Textual Entailment With Deep Learning
1. create the following directories: raw, raw/dataset, raw/embeddings
2. download the GloVe embeddings (http://nlp.stanford.edu/data/glove.840B.300d.zip), unzip it and put the file [glove.840B.300d.txt] inside the raw/embeddings directory
3. download the SNLI corpus (http://nlp.stanford.edu/projects/snli/snli_1.0.zip), unzip it and put the files [snli_1.0_train.jsonl, snli_1.0_dev.jsonl, snli_1.0_test.jsonl] inside the raw/dataset directory
4. run python preprocess_data.py

* for the centroids model:
  1. run once "python generate_centroids.py" which can be found inside models/centroids/
  2. run "python generate_centroids.py" which can be found inside models/centroids/
  3. the results will be created inside models/centroids/results
* for the GRU model:
  1. run "python gru.py" which can be found inside models/gru/
  2. the results will be created inside models/gru/results
* for the attention model:
  1. run "python attention_model.py" which can be found inside models/attention/
  2. the results will be created inside models/attention/results
