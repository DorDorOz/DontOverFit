import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn import  metrics
from sklearn import preprocessing
from sklearn.utils import class_weight
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgbm
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
import seaborn as sn
from scipy import stats
import xgboost as xgb
from xgboost.sklearn import  XGBClassifier
from scipy.optimize import minimize
import scipy.sparse
import collections, numpy
from sklearn.model_selection import LeaveOneOut



train = pd.read_csv("D:/Python/data/DontOverFit/train.csv")
testSub = pd.read_csv("D:/Python/data/DontOverFit/test.csv")

pd.options.display.max_columns = testSub.shape[1]


train.describe()

train_values = train[train.columns.difference(['target', 'id'])]
train_target = train[['target']]

testSub_values = testSub[testSub.columns.difference(['id'])]
testSub_Ids = testSub[['id']]

X_train, X_test, y_train, y_test = train_test_split(train_values, train_target, test_size = 0.1, random_state = 0, stratify = train_target)

X_train = pd.concat([pd.DataFrame(X_train), 
                     pd.DataFrame(X_train),
                     pd.DataFrame(X_train),
                     pd.DataFrame(X_train)], 
                     axis = 0)


y_train = pd.concat([pd.DataFrame(y_train), 
                     pd.DataFrame(y_train),
                     pd.DataFrame(y_train),
                     pd.DataFrame(y_train)], 
                     axis = 0)

np.random.seed(0)
np.random.normal(0.1, 0.1, X_train.shape)

X_train = preprocessing.StandardScaler().fit_transform(X_train)
X_test = preprocessing.StandardScaler().fit_transform(X_test)
testSub_values = preprocessing.StandardScaler().fit_transform(testSub_values)

d_Matrix_train = xgb.DMatrix(X_train, label = y_train)
d_Matrix_test = xgb.DMatrix(X_test, label = y_test)
d_Matrix_testSub = xgb.DMatrix(testSub_values)

watchlist = [(d_Matrix_test, 'test')]



##xgb.__version__
##XGB_1
##[58]    test-auc:0.869599
xgb_param_1 = {
        'learning_rate' : 0.1,
        'max_depth' : 2,
        'colsample_bytree' : 1,
        'colsample_bylevel' : 1,
        'colsample_bynode' : 1,
        'subsample' : 1,
        'alpha' : 0,
        'lambda' : 1,
        'gamma' : 0,
        'min_child_weight' : 12,
        'scale_pos_weight' : 1,
        'max_delta_step' : 1,
        'objective': 'binary:logistic',
        'eval_metric' : 'auc'} 
xgb_model_1 = xgb.train(params = xgb_param_1, 
                        dtrain = d_Matrix_train, 
                        num_boost_round = 1000,
                        early_stopping_rounds = 50,
                        evals = watchlist)

xgb_model_1_prob = pd.DataFrame(xgb_model_1.predict(d_Matrix_test))

fpr, tpr, thresholds = metrics.roc_curve(y_test, xgb_model_1_prob)
metrics.auc(fpr, tpr)
    
#xgb_Sub_1 = pd.concat([pd.DataFrame(testSub_Ids), pd.DataFrame(xgb_model_1_prob)], axis = 1)
#xgb_Sub_1.columns = ['id', 'target']
#xgb_Sub_1.to_csv(r'D:/Python/data/DontOverFit/XgbSub_7.csv', index = False)

param_grid = { 
        'n_estimators' : [1000],
        'learning_rate' : [0.1],
        'gamma' : [0],
        'alpha' : [0],
        'lambda' : [1],
        'max_depth' : [2, 4, 6],
        'max_leaves' : [0], 
        'colsample_bytree': [1],
        'subsample' : [1],
        'max_delta_step' : [0],
        'min_child_weight' : [4, 6, 8, 10],
        'eval_metric' : ['auc'],
        'objective': ['binary:logistic']}

xgb_model = XGBClassifier(random_state = 0, n_jobs = -1, verbose = 10)

CV_model = GridSearchCV(estimator = xgb_model,
                        param_grid = param_grid,
                        n_jobs = -1,
                        cv = 5,
                        verbose = 10)

CV_model.fit(X_train_ens, y_train_ens, verbose = 10)
print(CV_model.best_estimator_)
print(CV_model.best_score_)













    
#    ##XGB_1
#    trn_x_xgb_1 = X_train.loc[trn_,:]
#    val_x_xgb_1 = X_train.loc[val_,:]
#    
#    trn_y_xgb_1 = y_train.loc[trn_,:]
#    val_y_xgb_1 = y_train.loc[val_,:]
#    
#    d_Matrix_train_xgb_1 = xgb.DMatrix(trn_x_xgb_1, label = trn_y_xgb_1)
#    d_Matrix_val_xgb_1 = xgb.DMatrix(val_x_xgb_1, label = val_y_xgb_1)
#    
#    xgb_param_1 = {
#        'learning_rate' : 0.5,
#        'max_depth' : 2,
#        'alpha' : 0,
#        'lambda' : 2,
#        'gamma' : 0,
#        'min_child_weight' : 12,
#        'objective': 'binary:logistic',
#        'eval_metric' : 'auc'} 
#    xgb_model_1 = xgb.train(params = xgb_param_1, 
#                        dtrain = d_Matrix_train_xgb_1, 
#                        num_boost_round = 100)
#    
#    oof_preds_xgb_1[val_] = xgb_model_1.predict(d_Matrix_val_xgb_1)
#    
#    fold_df_xgb_1 = pd.DataFrame(np.column_stack((oof_preds_xgb_1[val_], val_y_xgb_1)))
#    fold_results_xgb_1 = pd.DataFrame(np.row_stack((fold_results_xgb_1, fold_df_xgb_1)))
#    
#    fpr, tpr, thresholds = metrics.roc_curve(val_y_xgb_1, oof_preds_xgb_1[val_])
#    print("Xgb_1, Fold:", fold_, "oof-AUC: ", metrics.auc(fpr, tpr))
#    
#    sub_preds_xgb_1 += xgb_model_1.predict(d_Matrix_testsub) / splits
#    
#    ##XGB_2_OverSamp
#    trn_x_xgb_2 = X_train.loc[trn_,:]
#    trn_x_xgb_2 = pd.concat([pd.DataFrame(trn_x_xgb_2), 
#                             pd.DataFrame(trn_x_xgb_2),
#                             pd.DataFrame(trn_x_xgb_2),
#                             pd.DataFrame(trn_x_xgb_2)], 
#                     axis = 0)
#    
#    trn_y_xgb_2 = y_train.loc[trn_,:]
#    trn_y_xgb_2 = pd.concat([pd.DataFrame(trn_y_xgb_2), 
#                             pd.DataFrame(trn_y_xgb_2),
#                             pd.DataFrame(trn_y_xgb_2),
#                             pd.DataFrame(trn_y_xgb_2)], 
#                     axis = 0)
#    
#    val_x_xgb_2 = pd.DataFrame(X_train.loc[val_,:])
#    val_y_xgb_2 = pd.DataFrame(y_train.loc[val_,:])
#    
#    d_Matrix_train_xgb_2 = xgb.DMatrix(trn_x_xgb_2, label = trn_y_xgb_2)
#    d_Matrix_val_xgb_2 = xgb.DMatrix(val_x_xgb_2, label = val_y_xgb_2)
#    
#    xgb_param_2 = {
#        'learning_rate' : 0.5,
#        'max_depth' : 2,
#        'alpha' : 0,
#        'lambda' : 2,
#        'gamma' : 0,
#        'min_child_weight' : 12,
#        'objective': 'binary:logistic',
#        'eval_metric' : 'auc'} 
#    xgb_model_2 = xgb.train(params = xgb_param_2, 
#                        dtrain = d_Matrix_train_xgb_2, 
#                        num_boost_round = 100)
#    
#    oof_preds_xgb_2[val_] = xgb_model_2.predict(d_Matrix_val_xgb_2)
#    
#    fold_df_xgb_2 = pd.DataFrame(np.column_stack((oof_preds_xgb_2[val_], val_y_xgb_2)))
#    fold_results_xgb_2 = pd.DataFrame(np.row_stack((fold_results_xgb_2, fold_df_xgb_2)))
#    
#    fpr, tpr, thresholds = metrics.roc_curve(val_y_xgb_2, oof_preds_xgb_2[val_])
#    print("Xgb_2, Fold:", fold_, "oof-AUC: ", metrics.auc(fpr, tpr))
#    
#    sub_preds_xgb_2 += xgb_model_2.predict(d_Matrix_testsub) / splits
#
#
#
#















