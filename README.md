# Code for the paper XXXX submitted at CIKM'22.

The Santander products recommendation dataset used in our experiments can be downloaded from [here](https://www.kaggle.com/c/santander-product-recommendation/data?select=train_ver2.csv.zip).
In particular, what we need is the file `train_ver2.csv` which we can put in the folder `data`.

The whole dataset contains hundreds of thousands of users data along a timespan of 17 months. However, we can reduce the dimension of the dataset so to use only a **subsample** of the total users.
```
python subsample_data.py --input_file "data/train_ver2.csv" --sample_size 20000 --min_data_points 17
```
`sample_size` is the number of users we want to subsample from the full dataset (input `None` if you want to use the full data), and `min_data_points` is used to filter users having less than `min_data_points` records (ignore it if you don't want to filter).
This process will generate the file `data/train_reduced.csv` which we can simply rename as `data/train.csv`.

## Train our Transformer model

The code relative to our model is stored in the directory `model`.
The file `transformer.py` implements the transformer while the file `transformer_model.py` handles the data preprocessing, training and testing.

To begin with, let's preprocess the data.
```
python model/transformer_model.py --save_data --no_load_data --no_train
```
This command creates a file `data.npz` stored in the folder `data` so that next time the preprocessed data can be immediately loaded.

Next, let's train the model for 100 epochs with a warm-up learning rate for the first 10 epochs.
```
python model/transformer_model.py --save_weights --epochs 100 --warmup_epochs 10
```

If we want only to test it, we can simply use the following.
```
python model/transformer_model.py --load_weights --no_train
```

To evaluate the model on the full set of metrics you can run the file `evaluation.py`
```
python model/evaluation.py
```
Default evaluation is on acquisition, but you can evaluate the model on the items ownership task adding the `--ownership` argument.
```
python model/evaluation.py --ownership
```

## Results on items ownership
| Model                      | Prec1  | Prec5 | Prec10 | Rec1 | Rec5 | Rec10 | MRR20 | NDCG20 |
|----------------------------|--------|-------|-------|-------|------|-------|-------|--------|
| Our model                  | **0.9920**| **0.3537**| **0.1875** | **0.7560** | **0.9836** | **0.9990** | **0.9956**| **0.9961**|
| Amazon Personalize         | 0.9622| 0.2825| 0.1664 | 0.7397| 0.8865|0.9571 | 0.9435| 0.9435|

## Results on items acquisition
| Model                      | Prec1  | Prec5 | Prec10 | Rec1 | Rec5 | Rec10 | MRR20 | NDCG20 |
|----------------------------|--------|-------|-------|-------|------|-------|-------|--------|
| Our model                  | **0.9891**| **0.4022**| **0.2157** |**0.6975**|**0.9764**|0.9979| **0.9937**| **0.9941**|
| Xgboost                    | 0.6887| 0.2449| 0.1285 | 0.6062|0.9440|0.9866| 0.8054| 0.8556|
| Amazon Personalize         | 0.4653| 0.1491| 0.0935 |0.4183|0.6396|0.7869| 0.5788| 0.6505|

## Clustering

You can run the clustering algorithm with the command:
```
python analyze_embedding.py
```