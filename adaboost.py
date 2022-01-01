from os import replace
from random import weibullvariate
import random
import pandas as pd
from scipy.sparse import data
import timeit
import numpy as np
from math import log
import statistics
from sklearn.ensemble import AdaBoostClassifier

class Node:
    def __init__(self,attribute,type):
        self.attribute = attribute
        self.nodeType = type
        self.children = {}

def get_accuracy(predicted,actual):
    return round(float(sum(actual==predicted))/float(len(actual)),2)

def calculateGini(trainingSet,size):
    unique, counts = np.unique(trainingSet[:,49], return_counts=True)

    sumOfIndividualValues=0
    for i in range(len(unique)):
        sumOfIndividualValues += (counts[i]/size)**2

    gain = 1-sumOfIndividualValues
    return gain
    
def bestAttribute(trainingSet,attributes): 
    totalSize = trainingSet.shape[0]
    gainOfSet = calculateGini(trainingSet,totalSize)
    best_attribute = None
    max_gain = 0
    for column in attributes:
        gini=0
        for value in np.unique(trainingSet[:,column]):
            subSetOfValue = trainingSet[trainingSet[:,column] == value]
            subSetSize = len(subSetOfValue)
            countByTotal = subSetSize/totalSize
            gini += countByTotal * calculateGini(subSetOfValue,subSetSize)
        gain = gainOfSet-gini
        if gain > max_gain:
            best_attribute = column
            max_gain = gain

    return best_attribute

def buildTree(trainingSet,attributes,depth,maxDepth,minExamples):
    if depth==maxDepth or len(attributes)==0 or len(trainingSet)<minExamples:
        maxLabel = statistics.mode(sorted(trainingSet[:,49]))
        return Node(maxLabel,'leaf')
    
    if(len(np.unique(trainingSet[:,49]))==1):
        return Node(np.unique((trainingSet[:,49]))[0],'leaf')
    
    bestAttributeNow = bestAttribute(trainingSet,attributes)
    
    if bestAttributeNow == None:
        maxLabel = statistics.mode(sorted(trainingSet[:,49]))
        return Node(maxLabel,'leaf')
    
    root = Node(bestAttributeNow,'child')
    attributes.remove(bestAttributeNow)
    bestAttributeUniqueList = sorted(np.unique(trainingSet[:,bestAttributeNow]))
    for value in bestAttributeUniqueList:
        root.children[value] = buildTree(trainingSet[trainingSet[:,bestAttributeNow] == value],attributes,depth+1,maxDepth,minExamples)
    attributes.append(bestAttributeNow)

    return root

def predict(root,row):
    if root.nodeType=='leaf':
        return root.attribute
    
    val = row[root.attribute]
    
    return predict(root.children[val],row)

def split_features_outcome(data):
    features = data.drop(['decision'],axis=1)
    decision = data['decision']
    return features,decision

def decisionTree(trainingSetRound,trainingSet,depth):
    features,decision = split_features_outcome(trainingSetRound)
    attributes = list(range(len(features.columns)))

    start = timeit.default_timer()
    trainingSetArray = trainingSetRound.to_numpy()
    root = buildTree(trainingSetArray,attributes,0,depth,50)
    stop = timeit.default_timer()
    features,decision = split_features_outcome(trainingSet)
    y_pred=[]
    for index,row in features.iterrows():
        y_pred.append(predict(root,row))
    
    return y_pred,root

def adaboost(trainingSet,testSet,noOfEstimators,learningRate):
    trainingSet['decision']=trainingSet['decision'].replace(0,-1)

    trainingSetArray = trainingSet.to_numpy()

    probSet = np.empty((trainingSetArray.shape[0],noOfEstimators+1))
    predSet = np.empty((trainingSetArray.shape[0],noOfEstimators))
    misClassSet = np.empty((trainingSetArray.shape[0],noOfEstimators))

    #initially the weight for all the data point is same which is 1/N, N is the number of data points
    probSet[:,0] = 1/trainingSet.shape[0]
    
    alpha = []

    listOfRoots = []

    for i in range(noOfEstimators):
        #select the sample based on the weight provided for each of the data point
        tempTrainingSet = trainingSet.sample(frac=1,replace=True,weights=probSet[:,i])
        #train a DT with depth as 1 and get the predictions for the training set
        y_pred,root = decisionTree(tempTrainingSet,trainingSet,1)
        predSet[:,i] = y_pred
        listOfRoots.append(root)

        #identify the misclassified data points
        misClassSet[:,i] = np.where(trainingSetArray[:,49]!=predSet[:,i],1,0)

        #sum of the probability of the misclassified data points
        error = sum(misClassSet[:,i]*probSet[:,i])
    
        alpha.append(learningRate*log((1-error)/error))

        #give more weight to misclassified and less weight to correctly identified
        new_weight = probSet[:,i] * np.exp(-1 * alpha[i]*trainingSetArray[:,49]*predSet[:,i])
        z=np.sum(new_weight)
        normalized_weight = new_weight/z
        probSet[:,i+1] = normalized_weight
    
    weighed_pred_value=0
    
    for i in range(noOfEstimators):
        weighed_pred_value += alpha[i]*predSet[:,i]
    #make final prediction
    final_pred = np.sign(weighed_pred_value)

    trainingAccuracy = get_accuracy(final_pred,trainingSet['decision'])

    #calculate test accuracy
    features,decision = split_features_outcome(testSet)
    decision = decision.replace(0,-1)
    y_pred = []
    for index,row in features.iterrows():
        modelPredictions = []
        for root in listOfRoots:
            modelPredictions.append(predict(root,row))
        final_pred=0
        for i in range(noOfEstimators):
            final_pred += alpha[i] * modelPredictions[i]
        y_pred.append(np.sign(final_pred))
    
    testAccuracy = get_accuracy(y_pred,decision)

    return trainingAccuracy,testAccuracy

if __name__ == "__main__":
    
    data_train=pd.read_csv("trainingSet.csv")
    data_test=pd.read_csv("testSet.csv")

    noOfEstimators = 20
    learningRate = 0.5

    trainingAccuracy,testAccuracy = adaboost(data_train,data_test,noOfEstimators,learningRate)
    print('Training Accuracy for Adaboost' ,trainingAccuracy)
    print('Testing Accuracy for Adaboost',testAccuracy)