from datetime import time
import pandas as pd
import numpy as np
from adaboost import adaboost
from statistics import mean,stdev
import math
import matplotlib.pyplot as plt
import timeit
import multiprocessing as mp


#sample the data based on the given random_state and t_frac
def sampleData(data,random_state_value,t_frac):
    return data.sample(frac=t_frac,random_state=random_state_value)

def performCV():
    trainData= pd.read_csv('trainingSet.csv')
    trainData = sampleData(trainData,18,1)

    tenFoldDataSet = np.array_split(trainData,10)

    noOfEstimators = [5, 10, 20, 40, 80, 100]
    
    decisionTreeMeanAccuracies = []
    decisionTreeSE = []
    decisionTreeTenFoldAccuracyList = []

    trainTestSets = []

    pool = mp.Pool()

    for index in range(10):
        trainTest = []
        temp_tenFoldDataSet=tenFoldDataSet.copy()

        setIndex = temp_tenFoldDataSet[index]
        del temp_tenFoldDataSet[index]

        setC = pd.concat(temp_tenFoldDataSet)

        trainTest.append(setIndex)
        trainTest.append(setC)

        trainTestSets.append(trainTest)
    
    for estimators in noOfEstimators:
        print("Cross Validation running for Number of estimators", estimators)
        listOfResults = [pool.apply_async(adaboost, args=(trainTestSets[i][1],trainTestSets[i][0],estimators,0.5)) for i in range(10)]

        for result in listOfResults:
            decisionTreeTenFoldAccuracyList.append(result.get()[1])
            
        decisionTreeMeanAccuracies.append(mean(decisionTreeTenFoldAccuracyList))
        
        decisionTreeSE.append(stdev(decisionTreeTenFoldAccuracyList)/math.sqrt(10))
        
        decisionTreeTenFoldAccuracyList.clear()
    
    pool.close()

    plt.errorbar( noOfEstimators, decisionTreeMeanAccuracies, yerr= decisionTreeSE)
    
    plt.xlabel('No of Estimators')
    plt.ylabel('Testing Accuracy')
    plt.title('Test Accuracy of Adaboost and the standard errors')

    plt.show()

if __name__ == "__main__":
    performCV()