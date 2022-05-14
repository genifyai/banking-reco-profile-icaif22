import numpy as np
import pandas as pd
import argparse


if __name__ == '__main__':
    """
    input_file: csv file containing the users data
    it generates a json file which indicates AP about which users to generate recommendations,
    the json file format is specified at https://docs.aws.amazon.com/personalize/latest/dg/recommendations-batch.html
    USAGE: python amazon_personalize/make_input_json.py --input_file "data/users_data.csv"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    args = parser.parse_known_args()[0]
    input_file = args.input_file

    df = pd.read_csv(input_file)
    users = np.unique(df[['USER_ID']], axis=0)
    print("total users:", len(users))

    with open('amazon_personalize/input_aws.json', 'w+') as f:
        for i in users:
            f.write('{"userId": "' + str(i[0]) + '"}' + '\n')

    print("process done")
