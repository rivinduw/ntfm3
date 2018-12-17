"""
Formats the csvs to feed in to tf dataset.
Determines number of time-steps ahead to predict
"""
import numpy as np
import pandas as pd
import json

# from scipy.stats import mstats

def make_dataset(dataset_file="data/SH1N30s2.csv",steps = 1,train_size=57600):

    print("building dataset")
    data = pd.read_csv(dataset_file,index_col=0)

    data.iloc[:,::2] *= (120.0/1.0)


    data = data.astype(float)#unnessary
    # data.iloc[:,1::2] *= 1.0001 #unnessary?

    lane_num = 3.0
    Vols = data.iloc[:,0::2].values
    Occs = data.iloc[:,1::2].values
    avgSpeeds = np.clip(Vols/(Occs*lane_num+1e-6),0.,120.0)
    avgDensity = Vols/(lane_num*avgSpeeds+1e-6)

    # print(avgSpeeds)
    # data = pd.DataFrame(np.zeros((data.shape[0],data.shape[1])))
    data.iloc[:,1::2] = avgDensity
    data.iloc[:,0::2] = avgDensity*avgSpeeds

    # data.iloc[:,::2] = avgSpeeds*avgDensity
    data.replace(to_replace=[None], value=1.0, inplace=True)
    data.replace(to_replace=[0], value=1.0, inplace=True)

    data = data.iloc[:,17*2:22*2].fillna(value=1.0)

    data_in_train = data.iloc[:57600-steps,:]
    data_out_train = data.iloc[steps:57600,:]

    data_in_test = data.iloc[57600:-steps,:]
    data_out_test = data.iloc[57600+steps:,:]

    print("writing training data to file")
    data_in_train.to_csv(f"data/train/SH1N30s2-in-{steps}s.csv")
    data_out_train.to_csv(f"data/train/SH1N30s2-out-{steps}s.csv")

    print("writing test data to file")
    data_in_test.to_csv(f"data/test/SH1N30s2-in-{steps}s.csv")
    data_out_test.to_csv(f"data/test/SH1N30s2-out-{steps}s.csv")

    print("writing params to file")
    data_params = {}
    data_params["train_size"] = len(data_in_train)-steps*32*120
    data_params["dev_size"] = len(data_in_test)-steps*32*120
    data_params["test_size"] = len(data_in_test)-steps*32*120

    with open("data/dataset_params.json","w") as f:
        f.write(json.dumps(data_params))

if __name__ == '__main__':
    make_dataset()
