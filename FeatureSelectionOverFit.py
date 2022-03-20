import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import  XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn import  metrics
from sklearn import preprocessing
from sklearn.utils import class_weight
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import lightgbm as lgbm
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
import seaborn as sn
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE, RFECV
import scipy.sparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import itertools

pd.set_option('display.max_rows', 5000)

train = pd.read_csv("D:/Python/data/DontOverFit/train.csv")
testSub = pd.read_csv("D:/Python/data/DontOverFit/test.csv")

X_train = train[train.columns.difference(['target', 'id'])]
y_train = train[['target']]

X_testsub = testSub[testSub.columns.difference(['id'])]
testSub_Ids = testSub[['id']]
results_Single_AUC_group = pd.DataFrame(columns = ['Feature', 'Col', 'AUC'])

for x in range(0, 300):
    
    X_train_col = pd.DataFrame(X_train.iloc[:,[x]])
    splits = 5
    folds = StratifiedKFold(n_splits = splits, random_state = 0)
    oof_predicitons = np.zeros(X_train.shape[0])

    for fold_, (trn_, val_) in enumerate(folds.split(X_train_col, y_train)):
        
            trn_x = X_train_col.loc[trn_,:]
            val_x = X_train_col.loc[val_,:]
    
            trn_y = y_train.loc[trn_,:]
            val_y = y_train.loc[val_,:]
            
            clf = LogisticRegression(C = 0.1,
                                     max_iter = 2000, 
                                     class_weight = 'balanced', 
                                     penalty = 'l1', 
                                     solver = 'liblinear', 
                                     random_state = 0)
            
            clf.fit(trn_x, trn_y.values.ravel())
            probs = clf.predict_proba(val_x)[:,1]
            fpr, tpr, thresholds = metrics.roc_curve(val_y, probs)
            print("Var: ", int(list(X_train_col.columns.values)[0]), " AUC: ", metrics.auc(fpr, tpr))
            results_Single_AUC_group = results_Single_AUC_group.append({'Feature' : ('Var_' + str(int(list(X_train_col.columns.values)[0]))) , 
                                              'AUC' : metrics.auc(fpr, tpr), 
                                              'Col' : str(int(list(X_train_col.columns.values)[0]))} , ignore_index = True)
###################################################################################################################
    
results_Single_AUC_group = results_Single_AUC_group.groupby('Col', as_index = False).mean().sort_values(by = 'AUC', ascending = False)
results_Single_AUC_group = pd.DataFrame(results_Single_AUC_group[results_Single_AUC_group['AUC'] > 0.49])

results_Features = results_Single_AUC_group['Col'].reset_index(drop = True)
combn_Features = list(itertools.combinations(results_Features, 2))
combn_Features_Full = list(itertools.combinations(results_Single_AUC_group['Col'], 3))


col_Dif = train.columns.difference(['target', 'id', '33', '65'])
col_Selected_33 = pd.DataFrame({'Cols' : col_Dif, 'Col_33': '33'})
col_Selected_65 = pd.DataFrame({'Cols' : col_Dif, 'Col_65': '65'})

#############################################################################################

results_Sums_AUC_group = pd.DataFrame(columns = ['Features', 'AUC'])

for x in range(0, 1540):
    print('Features:' + combn_Features[x][0] + '+' + combn_Features[x][1])
    
    X_train_col = pd.DataFrame(X_train[combn_Features[x][0]] + X_train[combn_Features[x][1]]).reset_index(drop = True)
    y_train = pd.DataFrame(y_train).reset_index(drop = True)
    splits = 5
    folds = StratifiedKFold(n_splits = splits, random_state = 0)
    oof_predicitons = np.zeros(X_train.shape[0])
    
    for fold_, (trn_, val_) in enumerate(folds.split(X_train_col, y_train)):
        
            trn_x = X_train_col.loc[trn_,:]
            val_x = X_train_col.loc[val_,:]
    
            trn_y = y_train.loc[trn_,:]
            val_y = y_train.loc[val_,:]
            clf = LogisticRegression(C = 0.1,
                                     max_iter = 2000, 
                                     class_weight = 'balanced', 
                                     penalty = 'l1', 
                                     solver = 'liblinear', 
                                     random_state = 0)
            
            clf.fit(trn_x, trn_y.values.ravel())
            probs = clf.predict_proba(val_x)[:,1]
            fpr, tpr, thresholds = metrics.roc_curve(val_y, probs)
            print("VarSums: ", combn_Features[x][0] + '+' + combn_Features[x][1], 
                  " AUC: ", metrics.auc(fpr, tpr))
            results_Sums_AUC_group = results_Sums_AUC_group.append({'Features' : combn_Features[x][0] + '-' + combn_Features[x][1], 
                                                                    'AUC' : metrics.auc(fpr, tpr)} , ignore_index = True)

results_Sums_AUC_group = results_Sums_AUC_group.groupby('Features', as_index = False).mean().sort_values(by = 'AUC', ascending = False)
results_Sums_AUC_group = pd.DataFrame(results_Sums_AUC_group[results_Sums_AUC_group['AUC'] > 0.5])
results_Sums_AUC_group.head(10)

###################################################################################################################################


###################################################################################################################################











