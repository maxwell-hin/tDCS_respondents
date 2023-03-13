#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 17:28:20 2022

@author: maxwell_law
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,roc_curve, auc
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import pickle


data = pd.read_excel('tDCS_repetitive_2019_clinical_ML_20220308_Maxwell.xlsx')
#features = data.columns


#========imputation
def impute_data(data, col):
    for iv in data[col]:
        while data.isna().sum()[iv] != 0:
            data[iv].fillna(data[iv].median(), inplace = True)
            
impute_data(data, data.columns[0:])
data['IQ'].fillna(data['IQ'].median(), inplace=True)
data.isna().sum()


#=======Drop other groups
data.drop(data[data['Treatment_grp'] == 2].index, inplace = True)
data.drop(data[data['Treatment_grp'] == 3].index, inplace = True)
data.drop('Treatment_grp',axis=1, inplace = True)


#======Convert dv into 0/1
def encode_dv(y, method):
    if method == 'median':
        #target convert by median
        tar_by_med_DV = y.describe()
        res_criteria = tar_by_med_DV[:].loc['50%']
        for dv in tar_by_med_DV.columns:
            for i in range(len(y[dv])):
                if y[dv][i] <= res_criteria[dv]:
                    y[dv][i] = 1
                else: y[dv][i] = 0
    else:
        for dv in y.columns:
            y[dv] = [1 if i <= float(method) else 0 for i in y[dv]]
    return y
        
data[data.columns[-6:-4]] = encode_dv(data[data.columns[-6:-4]], '-10')
data[data.columns[-4:]] = encode_dv(data[data.columns[-4:]], '-20')

X = data[data.columns[:-6]]
X['Handedness'] = [1 if i == 2 else 0 for i in X['Handedness']]
#set DV
Y=pd.DataFrame()
Y = data.iloc[:,-6:]

X=X.drop(X.iloc[:,-10:],axis=1)

DV_label = ['SRS2_tot_1mth_pre_percent','SRS2_tot_post_pre_percent','SRS2_SCI_1mth_pre_percent',
            'SRS2_SCI_post_pre_percent','SRS2_RRB_1mth_pre_percent','SRS2_RRB_post_pre_percent']

Gp = '_Gp2'

#==============Random Forest

def apply_RF(X,y):
    rf_fit = RandomForestClassifier(n_estimators=100,criterion="gini",max_depth=2,min_samples_split=2,min_samples_leaf=1, max_features=2)
    rf_fit.fit(X,y)
    pred = rf_fit.predict(X)
    probas = rf_fit.predict_proba(X)
    importances = rf_fit.feature_importances_
    indices = np.argsort(importances)[::-1]
    colnames = list(X.columns)
    # Print the feature ranking
    importance_table = []
    for f in range(X.shape[1]):
       importance_table.append([colnames[indices[f]],round(importances[indices[f]],4)])
    importance_table = pd.DataFrame(importance_table)
    importance_table.columns = ['Feature','Gini impurity reduction score']
    print(classification_report(y, pred))
    print(confusion_matrix(y,pred))
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    print(tn, fp, fn, tp)
    print(importance_table)
    return rf_fit, importance_table, pred, probas



rf_fit_1 ,rf_fit_table_1, rf_fit_pred_1, rf_fit_probas_1 = apply_RF(X,Y.iloc[:,0])
rf_fit_2 ,rf_fit_table_2, rf_fit_pred_2, rf_fit_probas_2 = apply_RF(X,Y.iloc[:,1])
rf_fit_3 ,rf_fit_table_3, rf_fit_pred_3, rf_fit_probas_3 = apply_RF(X,Y.iloc[:,2])
rf_fit_4 ,rf_fit_table_4, rf_fit_pred_4, rf_fit_probas_4 = apply_RF(X,Y.iloc[:,3])
rf_fit_5 ,rf_fit_table_5, rf_fit_pred_5, rf_fit_probas_5 = apply_RF(X,Y.iloc[:,4])
rf_fit_6 ,rf_fit_table_6, rf_fit_pred_6, rf_fit_probas_6 = apply_RF(X,Y.iloc[:,5])


for k in range(6):
    exec(f"pickle.dump(rf_fit_{k+1}, open(DV_label[{k}]+'_RF'+Gp, 'wb'))")

def return_pred(filename, X):
    loaded_model = pickle.load(open(filename,'rb'))
    pred = loaded_model.predict(X)
    return pred

def return_table(filename, X):
    loaded_model = pickle.load(open(filename,'rb'))
    importances = loaded_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    colnames = list(X.columns)
    # Print the feature ranking
    importance_table = []
    for f in range(X.shape[1]):
       importance_table.append([colnames[indices[f]],round(importances[indices[f]],4)])
    importance_table = pd.DataFrame(importance_table)
    importance_table.columns = ['Feature','Gini impurity reduction score']
    return importance_table

def return_result(pred, y):
    print(classification_report(y, pred))
    print(confusion_matrix(y,pred))
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    print(tn, fp, fn, tp)
    
def return_probas(filename, X):
    loaded_model = pickle.load(open(filename,'rb'))
    probas = loaded_model.predict_proba(X)
    return probas
    
model_name = DV_label[5]
pred = return_pred(model_name+'_RF'+Gp,X)
importance_table = return_table(model_name+'_RF'+Gp,X)
return_result(pred,Y[model_name])
rf_probas = return_probas(model_name+'_RF'+Gp,X)

#=======SVM
def apply_svm(X,y):
    svml = svm.SVC(kernel='linear',probability=True)
    svml.fit(X,y)
    probas = svml.predict_proba(X)
    return svml, probas
    
for k in range(6):
    exec(f"svml_{k+1},svml_{k+1}_probas  = apply_svm(X,Y.iloc[:,{k}])")
    print('finished' + str(k+1))

for k in range(6):
    exec(f"pickle.dump(svml_{k+1}, open(DV_label[{k}]+'_svm'+Gp, 'wb'))")


def return_pred(filename, X):
    loaded_model = pickle.load(open(filename,'rb'))
    pred = loaded_model.predict(X)
    return pred

def return_result(pred, y):
    print(classification_report(y, pred))
    print(confusion_matrix(y,pred))
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    print(tn, fp, fn, tp)

def return_probas(filename, X):
    loaded_model = pickle.load(open(filename,'rb'))
    probas = loaded_model.predict_proba(X)
    return probas

for i in range(6):
    model_name = DV_label[i]
    pred = return_pred(model_name+'_svm'+Gp,X)
    return_result(pred,Y[model_name])
svm_probas = return_probas(model_name+'_svm'+Gp,X)

#===========gNB
def apply_gNB(X,y):
    gnb = GaussianNB()
    gnb.fit(X,y)
    probas = gnb.predict_proba(X)
    return gnb, probas

for k in range(6):
    exec(f"gnb_{k+1},gnb_{k+1}_probas  = apply_gNB(X,Y.iloc[:,{k}])")
    print('finished' + str(k+1))
    
for k in range(6):
    exec(f"pickle.dump(gnb_{k+1}, open(DV_label[{k}]+'_gnb'+Gp, 'wb'))")

def return_pred(filename, X):
    loaded_model = pickle.load(open(filename,'rb'))
    pred = loaded_model.predict(X)
    return pred

def return_result(pred, y):
    print(classification_report(y, pred))
    print(confusion_matrix(y,pred))
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    print(tn, fp, fn, tp)

def return_probas(filename, X):
    loaded_model = pickle.load(open(filename,'rb'))
    probas = loaded_model.predict_proba(X)
    return probas

for i in range(6):
    model_name = DV_label[i]
    pred = return_pred(model_name+'_gnb'+Gp,X)
    return_result(pred,Y[model_name])
    
gnb_probas = return_probas(model_name+'_gnb'+Gp,X)

#===============ROC
model_name = DV_label[5]
gnb_probas = return_probas(model_name+'_gnb'+Gp,X)
svm_probas = return_probas(model_name+'_svm'+Gp,X)
rf_probas = return_probas(model_name+'_RF'+Gp,X)

fpr_1, tpr_1, thresholds_1 = roc_curve(Y[model_name], rf_probas[:,1], pos_label=1)
fpr_2, tpr_2, thresholds_2 = roc_curve(Y[model_name], svm_probas[:,0], pos_label=1)
fpr_3, tpr_3, thresholds_3 = roc_curve(Y[model_name], gnb_probas[:,1], pos_label=1)
# get area under the curve
roc_auc_1 = auc(fpr_1, tpr_1)
roc_auc_2 = auc(fpr_2, tpr_2)
roc_auc_3 = auc(fpr_3, tpr_3)
# PLOT ROC curve
plt.figure(dpi=150)
plt.plot(fpr_1, tpr_1, lw=1, color='green', label=f'RF, AUC = {roc_auc_1:.3f}')
plt.plot(fpr_2, tpr_2, lw=1, color='red', label=f'SVM, AUC = {roc_auc_2:.3f}')
plt.plot(fpr_3, tpr_3, lw=1, color='blue', label=f'Gaussian NB, AUC = {roc_auc_3:.3f}')
plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
plt.title('ROC Curve for '+model_name+' in Group 2')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.legend()
plt.show()


from sklearn.linear_model import Lasso

from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
rf_fit = RandomForestClassifier(n_estimators=100,criterion="gini",max_depth=2,min_samples_split=2,min_samples_leaf=1, max_features=2)
regression = Lasso(alpha=0.5)
scores = cross_val_score(rf_fit, X, Y['SRS2_RRB_post_pre_percent'],scoring='accuracy', cv=5)
np.mean(scores)
regression.score


X = X.drop(X.iloc[:,:13],axis=1)
X = X.iloc[:,:-2]


























