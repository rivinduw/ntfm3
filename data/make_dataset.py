"""
Formats the csvs to feed in to tf dataset.
Determines number of time-steps ahead to predict
"""
import numpy as np
import pandas as pd
import glob
import os
import json

# from scipy.stats import mstats

def make_dataset(datadir = '/home/rwee015/Documents/Data/DataFromMikeSept2015/extract/',steps = 1,train_frac=0.6):

    print("building dataset")

    onRoad   = ['671', '672','673','674','675', '677','709','937','938','939','940','941'] #'676','936',
    onRamps  = [  '0',   '0','696',  '0','699',  '0','1121',  '0',  '0','951',  '0','953']
    offRamps = [  '0','1087',  '0',  '0',  '0','670',   '0','704','950',  '0','952',  '0']

    roadSegs = list(filter(lambda a: a != '0', set(onRoad+onRamps+offRamps))) #combine and remove 0s
    if not os.path.exists("data/processedFiles/"):
        os.makedirs("data/processedFiles/")

    allData=pd.DataFrame()
    files = glob.glob(datadir+'*')
    for file in files:
        data = pd.read_csv(file,parse_dates=[5])

        someSegs = data.loc[data['carriagewaySegmentId'].isin(roadSegs)]

        print("found",len(someSegs.groupby(['carriagewaySegmentId']).mean()),"of",len(roadSegs),"segments")

        unstackedSegs = someSegs.groupby(['lastReadingTime','carriagewaySegmentId']).mean().groupby([pd.Grouper(freq='10S', level=0),"carriagewaySegmentId"]).mean().unstack()

        unstackedSegs = unstackedSegs.resample('10S').mean()#.ffill()#

        getCols = ['totalVolume','averageOccupancy','averageSpeed','r_in','r_out']
        shortCols = ['q','o','s','r','f']
        unstackedSegs['totalVolume',0] = np.zeros(unstackedSegs.shape[0]) #empty columns if no on/off ramp

        allSegs = pd.DataFrame()
        for i in range(len(onRoad)):
            currentSeg = int(onRoad[i])
            onRamp = int(onRamps[i])
            offRamp = int(offRamps[i])
            oneSeg = pd.concat([unstackedSegs[getCols[0]][currentSeg],unstackedSegs[getCols[1]][currentSeg],unstackedSegs[getCols[2]][currentSeg],unstackedSegs[getCols[0]][onRamp],unstackedSegs[getCols[0]][offRamp]],axis=1)
            oneSeg.columns = ['{:02d}'.format(i)+'_'+str(currentSeg)+'_'+col for col in shortCols]
            allSegs = pd.concat([allSegs,oneSeg],axis=1)

        allSegs.to_csv('data/processedFiles/'+file.split("/")[-1]+'.csv')
        print('data/processedFiles/'+file.split("/")[-1]+'.csv','done')
        allData = pd.concat([allData,allSegs],axis=0)
        del allSegs

    means = someSegs.groupby(['carriagewaySegmentId']).mean()
    means['distance'] = (1000*means['segmentTime'] * means['averageSpeed']/60)
    seg_lens = means['distance'][[int(s) for s in onRoad]]
    pd.DataFrame(seg_lens).T.to_csv('data/seg_lens.csv',index=False)

    del data
    del someSegs
    del unstackedSegs

    allDataIn = allData.fillna(0.0)#method='pad',inplace=False)
    allData.fillna(0.0,inplace=True)

    train_size = int(float(train_frac)*len(allData))
    print("train size:",str(train_size))

    max_vals = allData.iloc[:train_size,:].max(axis=0)
    mean_vals = allData.iloc[:train_size,:][allData.iloc[:train_size,:]>0.0001].median(axis=0).fillna(0.0)
    pd.DataFrame(max_vals).T.to_csv('data/max_vals.csv',index=False)

    data_in_train = allDataIn.iloc[:train_size-steps,:]
    data_out_train = allData.iloc[steps:train_size,:]

    data_in_test = allDataIn.iloc[train_size:-steps,:]
    data_out_test = allData.iloc[train_size+steps:,:]

    print("writing training data to file")
    if not os.path.exists("data/train/"):
        os.makedirs("data/train/")
        os.makedirs("data/test/")
    data_in_train.to_csv("data/train/data-in.csv")
    data_out_train.to_csv("data/train/data-out.csv")

    print("writing test data to file")
    data_in_test.to_csv("data/test/data-in.csv")
    data_out_test.to_csv("data/test/data-out.csv")

    print("writing params to file")
    data_params = {}
    data_params["train_size"] = len(data_in_train)-steps*32*120
    data_params["dev_size"] = len(data_in_test)-steps*32*120
    data_params["test_size"] = len(data_in_test)-steps*32*120

    data_params["max_vals"] = list(max_vals)
    data_params["mean_vals"] = list(mean_vals)
    data_params["num_cols"] = len(data_params["max_vals"])#data_in_train.shape[1]
    data_params["seg_lens"] = list(seg_lens)
    data_params["num_segs"] = len(data_params["seg_lens"])

    with open("data/dataset_params.json","w") as f:
        f.write(json.dumps(data_params))

if __name__ == '__main__':
    make_dataset()
