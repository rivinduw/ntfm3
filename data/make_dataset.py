"""
Formats the csvs to feed in to tf dataset.
Determines number of time-steps ahead to predict
"""
import numpy as np
import pandas as pd
import json

def make_dataset(dataset_file="data/SH1N30s2.csv",steps = 1,train_size=57600):

    print("building dataset")
    data = pd.read_csv(dataset_file,index_col=0)

    data_in_train = data.iloc[:57600-steps,:]
    data_out_train = data.iloc[steps:57600,:]

    data_in_test = data.iloc[57600:-steps,:]
    data_out_test = data.iloc[57600+steps:,:]

    print("writing training data to file")
    data_in_train.to_csv(f"data/train/SH1N30s2-in-{steps}.csv")
    data_out_train.to_csv(f"data/train/SH1N30s2-out-{steps}.csv")

    print("writing test data to file")
    data_in_test.to_csv(f"data/test/SH1N30s2-in-{steps}.csv")
    data_out_test.to_csv(f"data/test/SH1N30s2-out-{steps}.csv")

    print("writing params to file")
    data_params = {}
    data_params["train_size"] = len(data_in_train)-steps*32
    data_params["dev_size"] = len(data_in_test)-steps*32
    data_params["test_size"] = len(data_in_test)-steps*32

    with open("data/dataset_params.json","w") as f:
        f.write(json.dumps(data_params))

if __name__ == '__main__':
    make_dataset()
