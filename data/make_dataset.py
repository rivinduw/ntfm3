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

    onRoad   = [ '672','673', '1','674', '1','675', '677', '1', '1', '1', '937', '1','938', '1','939','940','941'] #'676','936','671',709 removed
    onRamps  = [   '0','696', '0',  '0','699', '0',   '0', '0','1121', '0',   '0', '0',  '0', '951','0',  '0','953']
    offRamps = ['1087', '0',  '0',  '0', '0',  '0', '670', '0',   '0', '704', '0', '950','0', '0',  '0','952',  '0']

    roadSegs = list(filter(lambda a: a != '0', set(onRoad+onRamps+offRamps))) #combine and remove 0s
    if not os.path.exists("data/processedFiles/"):
        os.makedirs("data/processedFiles/")

    allData=pd.DataFrame()
    files = sorted(glob.glob(datadir+'*'))
    for file in files:
        data = pd.read_csv(file,parse_dates=[5])

        someSegs = data.loc[data['carriagewaySegmentId'].isin(roadSegs)]#[::10]

        print("found",len(someSegs.groupby(['carriagewaySegmentId']).mean()),"of",len(roadSegs),"segments")

        unstackedSegs = someSegs.groupby(['lastReadingTime','carriagewaySegmentId']).mean().groupby([pd.Grouper(freq='10S', level=0),"carriagewaySegmentId"]).mean().unstack()
        unstackedSegs['totalVolume',0] = np.zeros(unstackedSegs.shape[0]) #empty columns if no on/off ramp
        unstackedSegs = unstackedSegs+1e-3

        # unstackedSegs = unstackedSegs[::180]

        unstackedSegs = unstackedSegs.resample('10S').mean()#.ffill()#

        getCols = ['totalVolume','averageOccupancy','averageSpeed','r_in','r_out']
        shortCols = ['q','o','s','r','f']
        unstackedSegs['totalVolume',1] = np.zeros(unstackedSegs.shape[0]) #empty columns if missing data
        unstackedSegs['averageOccupancy',1] = np.zeros(unstackedSegs.shape[0])
        unstackedSegs['averageSpeed',1] = np.zeros(unstackedSegs.shape[0])

        allSegs = pd.DataFrame()
        for i in range(len(onRoad)):
            currentSeg = int(onRoad[i])
            onRamp = int(onRamps[i])
            offRamp = int(offRamps[i])
            oneSeg = pd.concat([unstackedSegs[getCols[0]][currentSeg],unstackedSegs[getCols[1]][currentSeg],unstackedSegs[getCols[2]][currentSeg],unstackedSegs[getCols[0]][onRamp],unstackedSegs[getCols[0]][offRamp]],axis=1)
            # oneSeg = oneSeg+1e-3
            oneSeg.columns = ['{:02d}'.format(i)+'_'+str(currentSeg)+'_'+col for col in shortCols]
            allSegs = pd.concat([allSegs,oneSeg],axis=1)

        allSegs.to_csv('data/processedFiles/'+file.split("/")[-1]+'.csv')
        print('data/processedFiles/'+file.split("/")[-1]+'.csv','done')
        allData = pd.concat([allData,allSegs],axis=0)
        del allSegs


    #because of the zero/missing segments, the lengths are not accurate when calculated
    # means = someSegs.groupby(['carriagewaySegmentId']).mean()
    # means['distance'] = (1000*means['segmentTime'] * means['averageSpeed']/60)
    # seg_lens = means['distance'][[int(s) for s in onRoad]]
    # pd.DataFrame(seg_lens).T.to_csv('data/seg_lens.csv',index=False)

    seg_lens = [ '568.0','411.0', '559.0','559.0', '396.5','396.5', '523.0', '419.0', '419.0', '468.0', '468.0', '431.0','431.0', '573.5','573.5','561.0','531.0']
    #should be string. If doing again, just remove quotes from above
    seg_lens = [float(i) for i in seg_lens]
    pd.DataFrame(seg_lens).T.to_csv('data/seg_lens.csv',index=False)

    del data
    del someSegs
    del unstackedSegs
    # allData = allData + 1e-6
    allDataIn = allData.fillna(0.0)#[:int(len(allData)*0.8)]#method='pad',inplace=False)
    allData = allDataIn#.fillna(0.0,inplace=True)

    train_size = int(float(train_frac)*len(allData))
    print("train size:",str(train_size))

    max_vals = allData.iloc[:train_size,:][allData.iloc[:train_size,:]>0.0001].quantile(.95,axis=0).fillna(0.0)+1.0
    max_vals = [float(i)+100.0 if (i==1.0 or i==1.0) else float(i) for i in max_vals]
    mean_vals = allData.iloc[:train_size,:][allData.iloc[:train_size,:]>0.0001].median(axis=0).fillna(0.0)
    # mean_vals = [float(i)+50.0 if (i==1.0 or i==1.001 or i==0.0 or i==0.001) else float(i) for i in mean_vals]
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
#[71.0, 1.0, 100.0999, 0.0, 0.0, 69.0, 20.916666666666668, 100.0020329698233, 0.0, 11.0, 81.66666666666667, 19.555555555555557, 100.06612978997968, 16.0, 0.0, 77.2, 18.666666666666668, 99.91924252521684, 0.0, 0.0, 72.0, 30.33333333333333, 99.97123370844159, 19.0, 0.0, 73.0, 19.83333333333333, 100.04138198956488, 0.0, 20.0, 71.6, 10.2, 99.99851829823957, 18.0, 0.0, 42.5, 22.666666666666664, 99.99985052450269, 0.0, 25.5, 37.0, 19.666666666666668, 99.99959238824756, 0.0, 21.0, 31.5, 33.0, 99.999999
# [34.0, 1.0, 100.059, 0.0, 0.0, 39.0, 3.666666666666667, 96.92822972967716, 0.0, 3.0, 46.33333333333334, 3.333333333333333, 98.73443427463017, 4.0, 0.0, 45.2, 4.466666666666667, 92.02368842936494, 0.0, 0.0, 41.5, 5.833333333333332, 86.88589556793916, 6.0, 0.0, 46.0, 4.5, 97.90352083365634, 0.0, 8.0, 45.4, 4.466666666666667, 92.58574675121665, 5.0, 0.0, 21.0, 5.166666666666667, 93.15151588059288, 0.0, 12.5, 17.5, 4.0, 96.68751085510283, 0.0, 7.0, 16.0, 3.333333333333333, 99.3935655037603, 3.0,
#[18.0, 1.0, 100.0308, 0.0, 0.0, 21.5, 2.0, 93.4590299732678, 0.0, 2.0, 25.33333333333333, 1.8888888888888888, 95.72116071602804, 2.0, 0.0, 24.8, 2.5333333333333328, 88.11907421922245, 0.0, 0.0, 20.5, 3.0, 82.4758546851543, 4.0, 0.0, 24.5, 2.333333333333333, 93.69524174311081, 0.0, 5.0, 24.6, 2.5333333333333337, 89.12147578787625, 3.0, 0.0, 11.5, 2.833333333333333, 87.1372953267659, 0.0, 6.5, 9.5, 2.166666666666667, 92.10341314413057, 0.0, 4.0, 9.0, 1.8333333333333333, 96.241265920

if __name__ == '__main__':
    make_dataset()
