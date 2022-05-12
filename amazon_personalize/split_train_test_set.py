import pandas as pd
import os
import numpy as np
import argparse


if __name__ == '__main__':
    """
    input_file: csv file containing the user-item interactions of the test users
    it splits the user-item interactions test data into 2 csv files:
    - the first one contains the interactions associated to the N-1 timestamps (used by AP to predict recommendations)
    - the second one contains the interactions associated to last timestamp (used compute metrics)
    USAGE: python split_train_test_set.py --input_file data/train.csv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    args = parser.parse_known_args()[0]
    input_file = args.input_file

    df = pd.read_csv(input_file)
    print("dataset rows:", df.shape[0])
    print("dataset users:", len(np.unique(df['ncodpers'])))

    def remove_last(x):
        last_timestamp = x["fecha_dato"][-1:].values[0]
        x = x.loc[x['fecha_dato'] != last_timestamp]
        return x

    def keep_last(x):
        last_timestamp = x["fecha_dato"][-1:].values[0]
        x = x.loc[x['fecha_dato'] == last_timestamp]
        return x

    df_test = df.groupby('ncodpers').apply(remove_last)
    df_test = df_test.reset_index(drop=True)
    print("train dataset rows:", df_test.shape[0])

    df_pred = df.groupby('ncodpers').apply(keep_last)
    df_pred = df_pred.reset_index(drop=True)
    print("test dataset rows:", df_pred.shape[0])

    df_test.to_csv(os.path.join("data", "train.csv"), index=False)
    df_pred.to_csv(os.path.join("data", "test.csv"), index=False)
    print("process done")
