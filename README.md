# Ensemble-models-of-Decision-Trees
Implementation of the Ensemble models of Decision Tree Classifier : Random Forests, Bagging, Adaboost. It also contains the analysis of the hyper parameters.

The project folder contains 5 python files: 
1. preprocess-assg4.py
2. ensemble.py
3. cv_numtrees.py
4. adaboost.py
5. cv_adaboost.py

###############

1. preprocess.py

This script contains the preprocessing steps like removing the columns, normalization, label encoding, discretization and split the dataset. It makes use of dating-full.csv as the input. It outputs trainingSet.csv and testSet.csv.

Execution : python3 preprocess.py

2. ensemble.py

This script contains the training and testing of the models for Bagging Tree and Random Forests. It takes in three arguments, the training file name, the test file and the model to be run.

modelIdx = 1 for BT (Bagging Tree)
modelIdx = 2 for RF (Random Forests)

Execution : python3 ensemble.py trainingFileName testFileName modelIdx

eg: 
Run command for BT model
python3 ensemble.py trainingSet.csv testSet.csv 1

Run command for RF model
python3 ensemble.py trainingSet.csv testSet.csv 2


3. cv_numtrees.py

This script performs the ten fold validation for the two models Bagging Tree and Random Forests based on the number of trees used. It outputs the p value of the t-test statistics between BT and RF model for each tree size. It also outputs a graph indicating the test accuracies of the two models and their standard errors for different no. of trees.

Execution : python3 cv_numtrees.py

4. adaboost.py

This script contains the training and testing of the models for Adaboost algorithm. It outputs the training and testing accuracy of the dataset. This contains the number of estimators as 20 after learning from cv.

Execution : python3 adaboost.py

5. cv_adaboost.py

This script performs the ten fold validation for the adaboost model based on the number of esimators used. It also outputs a graph indicating the test accuracies of the model and their standard errors for different no. of estimators.

Execution : python3 cv_adaboost.py
