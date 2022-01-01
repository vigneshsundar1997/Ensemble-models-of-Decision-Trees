from os import replace
import pandas as pd
import numpy as np
import timeit
import statistics
import math
import random
import multiprocessing as mp
from sys import argv

def get_accuracy(predicted,actual):
    #round off to two decimals
    return round(float(sum(actual==predicted))/float(len(actual)),2)

#Create a class to store the nodes. If the nodeType is leaf, then the attribute stores the label, else it stores the attribute
class Node:
    def __init__(self,attribute,type):
        self.attribute = attribute
        self.nodeType = type
        self.children = {}

#Split the features and decision of the set
def split_features_outcome(data):
    features = data.drop(['decision'],axis=1)
    decision = data['decision']
    return features,decision

#Function to calculate the gini index for the given attribute
def calculateGini(trainingSet,size):
    #get the counts of 0 and 1 labels for the attribute set
    unique, counts = np.unique(trainingSet[:,49], return_counts=True)

    #get sum of p(x)^2
    sumOfIndividualValues=0
    for i in range(len(unique)):
        sumOfIndividualValues += (counts[i]/size)**2

    #calculate the gini Index
    gain = 1-sumOfIndividualValues
    return gain
    
def bestAttribute(trainingSet,attributes):
    
    totalSize = trainingSet.shape[0]
    #calculate the gini index of the overall set
    gainOfSet = calculateGini(trainingSet,totalSize)
    best_attribute = None
    max_gain = 0
    #calculate gain for each attribute and the return the attribute with maximum gain
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

#function used to generate the tree
def buildTree(trainingSet,attributes,depth,maxDepth,minExamples):
    #If the depth has reached maxDepth for the tree or the number of attributes remaining is zero or the number of examples left is less than minExamples, get the maximum label of the set and return a leaf node
    if depth==maxDepth or len(attributes)==0 or len(trainingSet)<minExamples:
        maxLabel = statistics.mode(sorted(trainingSet[:,49]))
        return Node(maxLabel,'leaf')
    
    #If the label is the same, return a leaf node with that label
    if(len(np.unique(trainingSet[:,49]))==1):
        return Node(np.unique((trainingSet[:,49]))[0],'leaf')
    
    #Find the best attribute
    bestAttributeNow = bestAttribute(trainingSet,attributes)
    
    #if the gain of all attributes is 0, then return a leaf node with max label
    if bestAttributeNow == None:
        maxLabel = statistics.mode(sorted(trainingSet[:,49]))
        return Node(maxLabel,'leaf')
    
    root = Node(bestAttributeNow,'internal')
    #remove the attribute for the recursion
    attributes.remove(bestAttributeNow)
    bestAttributeUniqueList = sorted(np.unique(trainingSet[:,bestAttributeNow]))
    #form the children
    for value in bestAttributeUniqueList:
        root.children[value] = buildTree(trainingSet[trainingSet[:,bestAttributeNow] == value],attributes,depth+1,maxDepth,minExamples)
    attributes.append(bestAttributeNow)
    return root

def buildRandomTree(trainingSet,attributes,depth,maxDepth,minExamples,sampleAttributeSize):
    #If the depth has reached maxDepth for the tree or the number of attributes remaining is zero or the number of examples left is less than minExamples, get the maximum label of the set and return a leaf node
    if depth==maxDepth or len(attributes)==0 or len(trainingSet)<minExamples:
        maxLabel = statistics.mode(sorted(trainingSet[:,49]))
        return Node(maxLabel,'leaf')

    #If the label is the same, return a leaf node with that label
    if(len(np.unique(trainingSet[:,49]))==1):
        return Node(np.unique((trainingSet[:,49]))[0],'leaf')
    
    #select the best attribute from a random set of attributes of size sqrt(p) where p is the no. of attributes of the given set 
    sampledAttributes = random.sample(attributes,sampleAttributeSize)
    #Find the best attribute
    bestAttributeNow = bestAttribute(trainingSet,sampledAttributes)
    #if the gain of all attributes is 0, then return a leaf node with max label
    if bestAttributeNow == None:
        maxLabel = statistics.mode(sorted(trainingSet[:,49]))
        return Node(maxLabel,'leaf')
    
    root = Node(bestAttributeNow,'internal')
    #remove the attribute for the recursion
    attributes.remove(bestAttributeNow)
    bestAttributeUniqueList = sorted(np.unique(trainingSet[:,bestAttributeNow]))
    #form the children
    for value in bestAttributeUniqueList:
        root.children[value] = buildRandomTree(trainingSet[trainingSet[:,bestAttributeNow] == value],attributes,depth+1,maxDepth,minExamples,sampleAttributeSize)
    attributes.append(bestAttributeNow)
    return root

def predict(root,row):
    #if we reach a leaf node, return the attribute value which stores the label
    if root.nodeType=='leaf':
        return root.attribute
    
    #else get the value for the attribute
    val = row[root.attribute]

    #recurse the children
    return predict(root.children[val],row)

def bagging(trainingSet,testSet,depth,noOfTrees):
    features,decision = split_features_outcome(trainingSet)
    attributes = list(range(len(features.columns)))
    listOfRoots = []

    features_array = features.to_numpy()

    pool = mp.Pool()

    trainingSetList = []

    #generate the given noOfTrees trees and store all the roots
    for i in range(0,noOfTrees):
        baggedTrainingSet=trainingSet.sample(frac=1,replace=True)
        baggedTrainingSetArray = baggedTrainingSet.to_numpy()
        trainingSetList.append(baggedTrainingSetArray)
    
    #multiprocessing
    listOfRootsObject = [pool.apply_async(buildTree, args=(x,list(attributes),0,depth,50)) for x in trainingSetList]
    

    listOfRoots=[]
    for root in listOfRootsObject:
        listOfRoots.append(root.get())
    
    pool.close()

    y_pred = []
    
    #predict the values, and take the majority. If both the labels are of same count, return 0 label
    for i in range(len(features_array)):
       predictedValues=[]
       for root in listOfRoots:
          predictedValues.append(predict(root,features_array[i]))
       y_pred.append(statistics.mode(sorted(predictedValues)))   

    training_accuracy = get_accuracy(np.array(y_pred),decision)

    #calculate for testing
    features,decision = split_features_outcome(testSet)
    features_array = features.to_numpy()
    y_pred=[]
    for i in range(len(features_array)):
        predictedValues=[]
        for root in listOfRoots:
           predictedValues.append(predict(root,features_array[i]))
        y_pred.append(statistics.mode(sorted(predictedValues)))
    test_accuracy = get_accuracy(np.array(y_pred),decision)    
    return training_accuracy,test_accuracy

def randomForests(trainingSet,testSet,depth,noOfTrees):
    features,decision = split_features_outcome(trainingSet)
    attributes = list(range(len(features.columns)))

    features_array = features.to_numpy()
    #get sqrt(noOfAttributes), the number of attributes randomnly selected to get the bestAttribute
    p = int(math.sqrt(len(attributes)))
    
    pool = mp.Pool()

    trainingSetList = []

    for i in range(0,noOfTrees):
        baggedTrainingSet=trainingSet.sample(frac=1,replace=True)
        baggedTrainingSetArray = baggedTrainingSet.to_numpy()
        trainingSetList.append(baggedTrainingSetArray)
    
    listOfRootsObject = [pool.apply_async(buildRandomTree, args=(x,list(attributes),0,depth,50,p)) for x in trainingSetList]
    listOfRoots=[]
    for root in listOfRootsObject:
        listOfRoots.append(root.get())
    pool.close()

    #predict the values, and take the majority. If both the labels are of same count, return 0 label
    y_pred = []
    for i in range(len(features_array)):
        predictedValues=[]
        for root in listOfRoots:
            predictedValues.append(predict(root,features_array[i]))
        y_pred.append(statistics.mode(sorted(predictedValues)))

    training_accuracy = get_accuracy(np.array(y_pred),decision)

    features,decision = split_features_outcome(testSet)
    features_array = features.to_numpy()
    y_pred = []
    for i in range(len(features_array)):
        predictedValues=[]
        for root in listOfRoots:
            predictedValues.append(predict(root,features_array[i]))
        y_pred.append(statistics.mode(sorted(predictedValues)))
    
    test_accuracy = get_accuracy(np.array(y_pred),decision)    
    return training_accuracy,test_accuracy


if __name__ == "__main__":

    trainingDataFileName = argv[1]
    testDataFileName = argv[2]
    modelIdx = argv[3]
    
    data_train=pd.read_csv(trainingDataFileName)
    data_test=pd.read_csv(testDataFileName)

    if modelIdx == str(1):
        training_accuracy,test_accuracy=bagging(data_train,data_test,8,30)
        print('Training Accuracy BT:', training_accuracy)
        print('Testing Accuracy BT:' ,test_accuracy)
    if modelIdx == str(2):
        training_accuracy,test_accuracy=randomForests(data_train,data_test,8,30)
        print('Training Accuracy RF:', training_accuracy)
        print('Testing Accuracy RF:' ,test_accuracy)