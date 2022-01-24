import numpy as np
import sys
import scipy.integrate
np.set_printoptions(threshold=sys.maxsize)
# Standardize the data attributes for the Iris dataset.
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pywt
import pywt.data
import csv
import os
from pylab import *
from sklearn import *
from sklearn.metrics import *
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import gmean
def obtain_train_Data() :
    temp_list_to_save_SVM = []
    y=[]
    for line in  open("Filter_banks_feature_vectors\\coif4Coeffs_Avrage_features_for_Training_Data.txt" , 'r'):
        list1 =line.rstrip().split()
        if len(list1) == 2 :
            y.append( list1[1])
            continue
        temp_list_to_save_SVM.append([float(i) for i in list1])


    return temp_list_to_save_SVM , y

def Random_Forest_train ():
    x, y = obtain_train_Data()
    cross_fold_percentage = 0
    counter = 0

    x1 = []     #Variable to load Train sample
    y1 = []     #Variable to load Train class
    x2 = []     #Variable to load Train sample
    y2 = []     #Variable to load Train sample
    sp = 0      #totall of specificity in 5 fold cross
    sen = 0     #Total of sensitivity in 5 fold cross
    Gm = 0      #Total of Geometric mean in 5 fold cross

    clf = RandomForestClassifier(max_depth=64, random_state= 0  , n_estimators=210 , bootstrap=False,n_jobs=-1)

    kf = KFold(n_splits=5,shuffle= True)

    for train_indices, test_indices in kf.split(x):

        for i in train_indices :
            x1.append(x[i])
            y1.append(y[i])

        clf.fit(x1, y1)
        x1 = [] #empty the array for the next loop
        y1 = [] #empty the array for the next loop
        x2 = [] #empty the array for the next loop
        y2 = [] #empty the array for the next loop
        for i in test_indices :
            x2.append(x[i])
            y2.append(y[i])

        cross_fold_percentage = cross_fold_percentage +  clf.score(x2, y2)

        tp, fn, fp, tn = confusion_matrix(y2, clf.predict(x2)).ravel()
        sp = tn / (tn+fp) + sp
        sen = tp/(tp+fn) + sen
        GmeanSP = tn / (tn+fp)
        GmeanSen = tp/(tp+fn)
        #matrix = confusion_matrix(y2, clf.predict(x2))
        #print("Sensitivity/ Specificity : ", matrix.diagonal() / matrix.sum(axis=1))
        data = [GmeanSen,GmeanSP]
        Gm = gmean(data) + Gm


    #print the Sensitivity,Specificity, Accuracy and G-mean
    print("the 5-fold score Accuracy :   ",(cross_fold_percentage/5))
    print("Sensitivity(Recall):     ", sen / 5 )
    print ("Specificity:        ", sp/5)
    print ( "G mean    " , Gm/5)

Random_Forest_train()

