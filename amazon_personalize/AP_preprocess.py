import pandas as pd
import time
import datetime
import argparse
from config import *


def median_income(df):
    df.loc[df.renta.isnull(), 'renta'] = df.renta.median(skipna=True)
    return df


def make_users_data(df, output_user_data):
    # make some data cleaning
    df_user_fields = ["ncodpers", "age", "ind_nuevo", "antiguedad", "renta", "segmento"]
    df_user = df[df_user_fields]
    # keep unique ids
    df_user = df_user.drop_duplicates(subset="ncodpers", keep="first")
    df_user["ncodpers"] = df_user["ncodpers"].astype(int)
    # entries whose "ind_nuevo" field is missing are all new customers, so replace accordingly
    df_user.loc[df_user["ind_nuevo"].isnull(), "ind_nuevo"] = 1
    df_user["ind_nuevo"] = df_user["ind_nuevo"].astype(int)
    # entries whose "antiguedad" field is missing all new customers, so give them the minimum seniority
    df_user.antiguedad = pd.to_numeric(df_user.antiguedad, errors="coerce")
    df_user.loc[df_user.antiguedad.isnull(), "antiguedad"] = df_user.antiguedad.min()
    df_user.loc[df_user.antiguedad < 0, "antiguedad"] = 0
    df_user["antiguedad"] = df_user["antiguedad"].astype(int)
    # fix customers age
    df_user["age"] = pd.to_numeric(df_user["age"], errors="coerce")
    df_user["age"].fillna(df_user["age"].mean(), inplace=True)
    df_user["age"] = df_user["age"].astype(int)
    # add "UNKNOWN" if field "segmento" is missing
    df_user.loc[df_user["segmento"].isnull(), "segmento"] = "UNKNOWN"
    # add "USER_ID" required field
    df_user["USER_ID"] = df_user["ncodpers"]
    df_user.drop(columns=['ncodpers'], inplace=True)
    df_user = df_user[['USER_ID'] + df_user_fields[1:]]
    print(df_user)
    df_user.to_csv(output_user_data, index=False)


def make_interactions_data(df, output_interactions_data):
    df_interactions_vector = df[["fecha_dato", "ncodpers"] + target_cols]
    df_interactions_index = pd.DataFrame(columns=["TIMESTAMP", "USER_ID", "ITEM_ID"])

    def get_user_product(row):
        if not hasattr(get_user_product, "count"):
            get_user_product.count = 0
            get_user_product.count_rows = 0
        for i, c in enumerate(df_interactions_vector.columns[2:]):
            if row[c] == 1:
                pass
                timestamp = int(time.mktime(datetime.datetime.strptime(row["fecha_dato"], "%Y-%m-%d").timetuple()))
                df_interactions_index.loc[get_user_product.count] = [timestamp, row["ncodpers"], i]
                get_user_product.count += 1
        get_user_product.count_rows += 1
        print(round(get_user_product.count_rows/df_interactions_vector.shape[0], 3) * 100, "%")

    df_interactions_vector.apply(get_user_product, axis=1)
    df_interactions_index["ITEM_ID"] = df_interactions_index["ITEM_ID"].astype(int)
    df_interactions_index["USER_ID"] = df_interactions_index["USER_ID"].astype(int)
    df_interactions_index["TIMESTAMP"] = df_interactions_index["TIMESTAMP"].astype(int)
    print(df_interactions_index)
    df_interactions_index.to_csv(output_interactions_data, index=False)


if __name__ == '__main__':
    """
    input_file: csv file containing the original Santander product recommendation data.
    - can be downloaded from https://www.kaggle.com/c/santander-product-recommendation/data.
    It generates 2 files as required by Amazon Personalize to train a recosys model.
    - users_data.csv: csv file containing users' features, all missing values will be filled, and are kept only the
      5 most important features.
    - interactions_data.csv: csv file containing users interactions according to the format wanted by AP which is
      defined here: https://docs.aws.amazon.com/personalize/latest/dg/data-prep-formatting.html.
    USAGE: python amazon_personalize/AP_preprocess.py --input_file "data/train.csv" --output_user_data "data/users_data.csv" --output_interactions_data "data/interactions_data.csv"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_user_data', type=str, required=True)
    parser.add_argument('--output_interactions_data', type=str, required=True)
    args = parser.parse_known_args()[0]

    df = pd.read_csv(args.input_file)

    # preprocess
    # provide median income by province
    df = df.groupby('nomprov').apply(median_income)
    df.loc[df.renta.isnull(), "renta"] = df.renta.median(skipna=True)

    # make datasets
    print("Making users data...")
    make_users_data(df, args.output_user_data)
    print("Making interactions data...")
    make_interactions_data(df, args.output_interactions_data)
    print("process done")
