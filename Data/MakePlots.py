import numpy as np
import pylab
import pandas as pd
from blaze import mean
from pandas import DataFrame
import matplotlib.pyplot as plt
from os.path import join
from os import listdir
import matplotlib.image as mpimg

def numToCases(arr):
    return [F'Case-{x:04d}' for x in arr]

def caseToNumber(arr):
    return [int(x.split('-')[1]) for x in arr if x.find('Case') != -1]

def plotHists(hists, title):
    return -1
    # colors = ['b','g']
    # for idx_col, data in enumerate(hists):
    #     plt.hist(data[cur_area].values, bins=np.arange(.1,1,.01),color=colors[idx_col])
    # plt.title(title)
    # plt.show()

def removeBlanks(line):
    return line.replace('   ',' ').replace('  ',' ').replace('\n','')

def appendValues(arr, values):
    for x in values:
        arr.append(int(x))
    return arr

def readTrainVal(folder):
    mydict = {}
    for ii, cur_model in enumerate(['Siemens','GE','Combined']):
        file_name = join(folder,F'Splits_{cur_model}.txt')
        name = F'{cur_model}'
        f = open(file_name, "r")
        status = 'start'
        train = []
        val = []
        for line in f:
            line = removeBlanks(line)
            # print(line)
            if (status == 'start') and (line.find('Train') != -1):
                values = line.split('[')[1].replace(']','').split(' ')[1:]
                train = appendValues(train,values)
                status = 'train'
                continue
            if status == 'train':
                if line.find(']') != -1: # Last line of train examples
                    values = line.split(']')[0].split(' ')[1:]
                    train = appendValues(train,values)
                    status = 'donetrain'
                    continue
                else: # Still reading train exampes
                    values = line.replace('\n','').split(' ')[1:]
                    train = appendValues(train,values)
                    continue
            # ------------ Reading validation examples --------
            if (status == 'donetrain') and (line.find('Validation') != -1):
                if line.find(']') == -1:
                    status = 'val'
                    values = line.split('[')[1].replace(']','').split(' ')[1:]
                else:
                    status = 'doneval'
                    values = line.split('[')[1].split(']')[0].split(' ')[1:]
                val = appendValues(val,values)
                continue
            if status == 'val':
                if line.find(']') != -1: # Last line of train examples
                    values = line.split(']')[0].replace('\n','').split(' ')[1:]
                    val = appendValues(val,values)
                    status = 'doneval'
                    continue
                else: # Still reading train exampes
                    values = line.replace('\n','').split(' ')[1:]
                    val = appendValues(val,values)
                    continue

        mydict[F'{cur_model}_train'] = train
        mydict[F'{cur_model}_validation'] = val
    return mydict

def getMeanCase(ser, val):
    diff = ser - val
    abs_diff = diff.abs()
    sort_diff = abs_diff.sort_values()
    return sort_diff.index[0]

def plotCurCase(case, allFiles, folder_name, idx):
    file_name = [x for x in allFiles if x.find(F'{case}') != -1]
    fig = plt.figure(figsize=(24,8))
    if len(file_name) > 0:
        plt.subplot(1,3,idx)
        plt.imshow(mpimg.imread(join(folder_name,file_name[0])))
        plt.axis('off')
        pylab.savefig(F'Test{idx}.png',bbox_inches='tight')

def plotImages(folder_name, df, cur_area):
    allFiles = listdir(folder_name)

    minval = df[cur_area].min()
    meanval = df[cur_area].mean()
    maxval = df[cur_area].max()

    mincase = df.index[df[cur_area] == minval]
    plotCurCase(mincase, allFiles, folder_name,1)
    meancase = getMeanCase(df[cur_area], meanval)
    plotCurCase(meancase, allFiles, folder_name,3)
    maxcase = df.index[df[cur_area] == maxval]
    plotCurCase(maxcase, allFiles, folder_name,2)
    print(F'Min:{mincase}:{minval:0.3f} Mean:{meancase}:{meanval:0.3f} Max:{maxcase}:{maxval:0.3f}')
    plt.show()

if __name__ == "__main__":

    # removeOutliers = {'Siemens': [212,50,218], 'GE': [392,364,394,354,357] }
    # cur_folder = 'Prostate'
    cur_folder = 'PZ'
    cur_ctr = cur_folder.lower()
    splits = readTrainVal(cur_folder)
    # print(splits)
    
    img_folder = '/media/osz1/DATA_Old/ALL/IMAGES/SEGMENTATION/Prostate/PaperRUN/GE/Prostate/'
    output_run_folder = '/media/osz1/DATA/Dropbox/UMIAMI/WorkUM/ProstateSegCNN/OUTPUT/Prostate/PaperRUN/'

    for ii, cur_model in enumerate(['Siemens','GE','Combined']):
        for mm, cur_area in enumerate(['ROI','Original']):
            for ll, cur_dataset in enumerate(['GE','Siemens']):
                file_name = F'{cur_dataset}_{cur_model}.csv'
                df = pd.read_csv(join(cur_folder,file_name), index_col=0)
                df.dropna(axis=0,how='any', inplace=True)
                # If we are plotting the model on the original dataset, then we need to filter the train vs validation
                if cur_model == cur_dataset:
                    hists = []
                    avg = []
                    std = []
                    for kk, cur_subset in enumerate(['train','validation']):
                        indexes = numToCases(splits[F'{cur_model}_{cur_subset}'])
                        # print(F'Filtering the cases for model {cur_model} and dataset:{cur_dataset}_{cur_subset}: tot_case: {len(indexes)}')
                        hists.append(df.filter(items=indexes, axis=0))
                        avg.append(F'{hists[-1][cur_area].mean():0.3f}')
                        std.append(F'{hists[-1][cur_area].std():0.3f}')

                    folder_name  = '/media/osz1/DATA_Old/ALL/IMAGES/SEGMENTATION/Prostate/PaperRUN/GE/Prostate/GE_yes_DA_DONE/'
                    plotImages(folder_name, hists[1], cur_area)
                    title = F'AVG train,val:{avg} train,val:{std}    Model:{cur_model}  \nDataset:{cur_dataset}   Ctr:{cur_ctr}   Area:{cur_area}'
                    print(F'AVG:{avg[1]} STD:{std[1]}  {cur_model}-{cur_dataset} {cur_area}')
                    plotHists(hists,title)

                else:
                    filt_df = df.filter(regex='Case*', axis=0) # We don't want the average (computed manually)
                    avg = filt_df[cur_area].mean()
                    std= filt_df[cur_area].std()
                    title = F'AVG {avg:0.3f} STD {std:0.3f}   Model:{cur_model}  Dataset:{cur_dataset} \n Ctr:{cur_ctr} Subset:ALL  Area: {cur_area}'
                    print(F'AVG:{avg:0.3f} STD:{std:0.3f}  {cur_model}-{cur_dataset} {cur_area}')
                    plotHists([filt_df],title)

                    # plt.bar(caseToNumber(filt_df.index.values), filt_df[cur_area].values)




