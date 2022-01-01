from typing import Deque
import pandas as pd
import numpy as np
from trees import bagging, randomForests
from statistics import mean
from statistics import stdev
import timeit
import math
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

#sample the data based on the given random_state and t_frac
def sampleData(data,random_state_value,t_frac):
    return data.sample(frac=t_frac,random_state=random_state_value)

def performCV():
    trainData= pd.read_csv('trainingSet.csv')

    trainData = sampleData(trainData,18,1)

    halfTrainData = sampleData(trainData,32,0.5)
    #get the ten fold data
    tenFoldDataSet = np.array_split(halfTrainData,10)

    noOfTrees = [10,20,40,50]

    baggingMeanAccuracies = []
    baggingSE = []
    baggingTenFoldAccuracyList = []

    randomForestMeanAccuracies = []
    randomForestsSE = []
    randomForestTenFoldAccuracyList = []

    for tree in noOfTrees:
        print("Cross Validation running for Number of trees", tree)
        for index in range(10):
            temp_tenFoldDataSet=tenFoldDataSet.copy()

            setIndex = temp_tenFoldDataSet[index]
            del temp_tenFoldDataSet[index]

            setC = pd.concat(temp_tenFoldDataSet)

        
            training_accuracy,test_accuracy = bagging(setC,setIndex,8,tree)
            baggingTenFoldAccuracyList.append(test_accuracy)

            training_accuracy,test_accuracy = randomForests(setC,setIndex,8,tree)
            randomForestTenFoldAccuracyList.append(test_accuracy)
    
    
        baggingMeanAccuracies.append(mean(baggingTenFoldAccuracyList))
        randomForestMeanAccuracies.append(mean(randomForestTenFoldAccuracyList))

    
        baggingSE.append(stdev(baggingTenFoldAccuracyList)/math.sqrt(10))
        randomForestsSE.append(stdev(randomForestTenFoldAccuracyList)/math.sqrt(10))

        p=ttest_rel(baggingTenFoldAccuracyList,randomForestTenFoldAccuracyList)

        print("t-test statistics for BT and RF for no. of trees as ",tree , p)

    
        baggingTenFoldAccuracyList.clear()
        randomForestTenFoldAccuracyList.clear()

    plt.errorbar( noOfTrees, baggingMeanAccuracies, yerr= baggingSE ,label='BT')
    plt.errorbar( noOfTrees, randomForestMeanAccuracies, yerr= randomForestsSE ,label='RF')

    plt.xlabel('Number of Trees')
    plt.ylabel('Testing Accuracy')
    plt.legend()
    plt.title('Test Accuracy of Bagging and RF and their standard errors')

    p=ttest_rel(baggingMeanAccuracies,randomForestMeanAccuracies)

    plt.show()

if __name__ == "__main__":
    performCV()