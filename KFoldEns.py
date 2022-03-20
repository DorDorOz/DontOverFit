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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
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


train = pd.read_csv("D:/Python/data/DontOverFit/train.csv")
testSub = pd.read_csv("D:/Python/data/DontOverFit/test.csv")

X_train = train[train.columns.difference(['target', 'id'])]
y_train = train[['target']]

pd.options.display.max_columns = train.shape[1]
pd.options.display.max_rows = 100
X_train.head()

#X_train, X_test, y_train, y_test = train_test_split(train_values, train_target, test_size = 0.1, random_state = 0, stratify = train_target)
#
#X_train = pd.DataFrame(X_train).reset_index(drop = True)
#X_test = pd.DataFrame(X_test).reset_index(drop = True)
#y_train = pd.DataFrame(y_train).reset_index(drop = True)
#y_test = pd.DataFrame(y_test).reset_index(drop = True)

X_testsub = testSub[testSub.columns.difference(['id'])]
testSub_Ids = testSub[['id']]


oof_preds_log_1 = np.zeros(X_train.shape[0])
oof_preds_log_2 = np.zeros(X_train.shape[0])
oof_preds_log_3 = np.zeros(X_train.shape[0])
oof_preds_log_4 = np.zeros(X_train.shape[0])
oof_preds_log_5 = np.zeros(X_train.shape[0])
oof_preds_log_6 = np.zeros(X_train.shape[0])
oof_preds_log_7 = np.zeros(X_train.shape[0])
oof_preds_log_8 = np.zeros(X_train.shape[0])
oof_preds_log_9 = np.zeros(X_train.shape[0])
oof_preds_log_10 = np.zeros(X_train.shape[0])

oof_preds_log_11 = np.zeros(X_train.shape[0])
oof_preds_log_12 = np.zeros(X_train.shape[0])
oof_preds_log_13 = np.zeros(X_train.shape[0])

fold_results_log_1 = pd.DataFrame(columns = ['Prediction', 'target'])
fold_results_log_2 = pd.DataFrame(columns = ['Prediction', 'target'])
fold_results_log_3 = pd.DataFrame(columns = ['Prediction', 'target'])
fold_results_log_4 = pd.DataFrame(columns = ['Prediction', 'target'])
fold_results_log_5 = pd.DataFrame(columns = ['Prediction', 'target'])
fold_results_log_6 = pd.DataFrame(columns = ['Prediction', 'target'])
fold_results_log_7 = pd.DataFrame(columns = ['Prediction', 'target'])
fold_results_log_8 = pd.DataFrame(columns = ['Prediction', 'target'])
fold_results_log_9 = pd.DataFrame(columns = ['Prediction', 'target'])
fold_results_log_10 = pd.DataFrame(columns = ['Prediction', 'target'])

fold_results_log_11 = pd.DataFrame(columns = ['Prediction', 'target'])
fold_results_log_12 = pd.DataFrame(columns = ['Prediction', 'target'])
fold_results_log_13 = pd.DataFrame(columns = ['Prediction', 'target'])

sub_preds_log_1 = np.zeros(X_testsub.shape[0])
sub_preds_log_2 = np.zeros(X_testsub.shape[0])
sub_preds_log_3 = np.zeros(X_testsub.shape[0])
sub_preds_log_4 = np.zeros(X_testsub.shape[0])
sub_preds_log_5 = np.zeros(X_testsub.shape[0])
sub_preds_log_6 = np.zeros(X_testsub.shape[0])
sub_preds_log_7 = np.zeros(X_testsub.shape[0])
sub_preds_log_8 = np.zeros(X_testsub.shape[0])
sub_preds_log_9 = np.zeros(X_testsub.shape[0])
sub_preds_log_10 = np.zeros(X_testsub.shape[0])

sub_preds_log_11 = np.zeros(X_testsub.shape[0])
sub_preds_log_12 = np.zeros(X_testsub.shape[0])
sub_preds_log_13 = np.zeros(X_testsub.shape[0])


splits = 5
folds = RepeatedStratifiedKFold(n_splits = splits, random_state = 0, n_repeats = 5)
for fold_, (trn_, val_) in enumerate(folds.split(X_train, y_train)):

    ##Log_Model_1
    trn_x_log_1 = X_train.loc[trn_,:]
    val_x_log_1 = X_train.loc[val_,:]
    
    trn_y_log_1 = y_train.loc[trn_,:]
    val_y_log_1 = y_train.loc[val_,:]
    
    log_clf_1 = LogisticRegression(C = 0.1,
                             max_iter = 1000, 
                             class_weight = 'balanced', 
                             penalty = 'l1', 
                             solver = 'liblinear', 
                             random_state = 0)
    log_clf_1_RFE = RFE(log_clf_1, 250, step = 1)
    log_clf_1_RFE.fit(trn_x_log_1, trn_y_log_1.values.ravel())
    oof_preds_log_1[val_] = log_clf_1_RFE.predict_proba(val_x_log_1)[:,1]
    
    fold_df_log_1 = pd.DataFrame(np.column_stack((oof_preds_log_1[val_], val_y_log_1)))
    fold_results_log_1 = pd.DataFrame(np.row_stack((fold_results_log_1, fold_df_log_1)))
    
    fpr, tpr, thresholds = metrics.roc_curve(val_y_log_1, oof_preds_log_1[val_])
    print("Log_1, Fold:", fold_, "oof-AUC: ", metrics.auc(fpr, tpr))
    
    sub_preds_log_1 += log_clf_1_RFE.predict_proba(X_testsub)[:,1] / splits
    
    ##Log_Model_2
    trn_x_log_2 = X_train.loc[trn_,:]
    val_x_log_2 = X_train.loc[val_,:]
    
    trn_y_log_2 = y_train.loc[trn_,:]
    val_y_log_2 = y_train.loc[val_,:]
    
    
    log_clf_2 = LogisticRegression(C = 0.1,
                             max_iter = 1000, 
                             class_weight = 'balanced', 
                             penalty = 'l1', 
                             solver = 'liblinear', 
                             random_state = 0)
    log_clf_2_RFE = RFE(log_clf_2, 200, step = 1)
    log_clf_2_RFE.fit(trn_x_log_2, trn_y_log_2.values.ravel())
    
    oof_preds_log_2[val_] = log_clf_2_RFE.predict_proba(val_x_log_2)[:,1]
    
    fold_df_log_2 = pd.DataFrame(np.column_stack((oof_preds_log_2[val_], val_y_log_2)))
    fold_results_log_2 = pd.DataFrame(np.row_stack((fold_results_log_2, fold_df_log_2)))
    
    fpr, tpr, thresholds = metrics.roc_curve(val_y_log_2, oof_preds_log_2[val_])
    print("Log_2, Fold:", fold_, "oof-AUC: ", metrics.auc(fpr, tpr))
    
    sub_preds_log_2 += log_clf_2_RFE.predict_proba(X_testsub)[:,1] / splits
    
    ##Log_Model_3_Noise
    np.random.seed(0)
    noise = np.random.normal(0.1, 0.1, X_train.shape)
    
    X_train_noise = X_train + noise

    trn_x_log_3 = X_train_noise.loc[trn_,:]
    val_x_log_3 = X_train_noise.loc[val_,:]
    
    trn_y_log_3 = y_train.loc[trn_,:]
    val_y_log_3 = y_train.loc[val_,:]
    
    
    log_clf_3 = LogisticRegression(C = 0.1,
                             max_iter = 2000, 
                             class_weight = 'balanced', 
                             penalty = 'l1', 
                             solver = 'liblinear', 
                             random_state = 0)
    log_clf_3_RFE = RFE(log_clf_3, 250, step = 1)
    log_clf_3_RFE.fit(trn_x_log_3, trn_y_log_3.values.ravel())
    
    oof_preds_log_3[val_] = log_clf_3_RFE.predict_proba(val_x_log_3)[:,1]
    
    fold_df_log_3 = pd.DataFrame(np.column_stack((oof_preds_log_3[val_], val_y_log_3)))
    fold_results_log_3 = pd.DataFrame(np.row_stack((fold_results_log_3, fold_df_log_3)))
    
    fpr, tpr, thresholds = metrics.roc_curve(val_y_log_3, oof_preds_log_3[val_])
    print("Log_3, Fold:", fold_, "oof-AUC: ", metrics.auc(fpr, tpr))
    
    sub_preds_log_3 += log_clf_3_RFE.predict_proba(X_testsub)[:,1] / splits
    
    ##Log_Model_4_OverSampled*2
    trn_x_log_4 = X_train.loc[trn_,:]
    trn_x_log_4 = np.row_stack((trn_x_log_4, trn_x_log_4))
    trn_y_log_4 = y_train.loc[trn_,:]
    trn_y_log_4 = np.row_stack((trn_y_log_4, trn_y_log_4))
    
    val_x_log_4 = X_train.loc[val_,:]
    val_y_log_4 = y_train.loc[val_,:]
    

    log_clf_4 = LogisticRegression(C = 0.1,
                             max_iter = 2000, 
                             class_weight = 'balanced', 
                             penalty = 'l1', 
                             solver = 'liblinear', 
                             random_state = 0)
    log_clf_4_RFE = RFE(log_clf_4, 250, step = 1)
    log_clf_4_RFE.fit(trn_x_log_4, trn_y_log_4.ravel())
    
    oof_preds_log_4[val_] = log_clf_4_RFE.predict_proba(val_x_log_4)[:,1]
    
    fold_df_log_4 = pd.DataFrame(np.column_stack((oof_preds_log_4[val_], val_y_log_4)))
    fold_results_log_4 = pd.DataFrame(np.row_stack((fold_results_log_4, fold_df_log_4)))
    
    fpr, tpr, thresholds = metrics.roc_curve(val_y_log_4, oof_preds_log_4[val_])
    print("Log_4, Fold:", fold_, "oof-AUC: ", metrics.auc(fpr, tpr))
    
    sub_preds_log_4 += log_clf_4_RFE.predict_proba(X_testsub)[:,1] / splits
    
    ##Log_Model_5_Scale

    trn_x_log_5 = preprocessing.StandardScaler().fit_transform(X_train.loc[trn_,:])
    val_x_log_5 = preprocessing.StandardScaler().fit_transform(X_train.loc[val_,:])
    
    trn_y_log_5 = y_train.loc[trn_,:]
    val_y_log_5 = y_train.loc[val_,:]
    
    
    log_clf_5 = LogisticRegression(C = 0.1,
                             max_iter = 2000, 
                             class_weight = 'balanced', 
                             penalty = 'l1', 
                             solver = 'liblinear', 
                             random_state = 0)
    log_clf_5_RFE = RFE(log_clf_5, 250, step = 1)
    log_clf_5_RFE.fit(trn_x_log_5, trn_y_log_5.values.ravel())
    
    oof_preds_log_5[val_] = log_clf_5_RFE.predict_proba(val_x_log_5)[:,1]
    
    fold_df_log_5 = pd.DataFrame(np.column_stack((oof_preds_log_5[val_], val_y_log_5)))
    fold_results_log_5 = pd.DataFrame(np.row_stack((fold_results_log_5, fold_df_log_5)))
    
    fpr, tpr, thresholds = metrics.roc_curve(val_y_log_5, oof_preds_log_5[val_])
    print("Log_5, Fold:", fold_, "oof-AUC: ", metrics.auc(fpr, tpr))
    
    sub_preds_log_5 += log_clf_5_RFE.predict_proba(preprocessing.StandardScaler().fit_transform(X_testsub))[:,1] / splits
    
    ##Log_Model_6_OverSampled*6
    trn_x_log_6 = X_train.loc[trn_,:]
    trn_x_log_6 = np.row_stack((trn_x_log_6, trn_x_log_6, trn_x_log_6, trn_x_log_6, trn_x_log_6, trn_x_log_6))
    trn_y_log_6 = y_train.loc[trn_,:]
    trn_y_log_6 = np.row_stack((trn_y_log_6, trn_y_log_6, trn_y_log_6, trn_y_log_6, trn_y_log_6, trn_y_log_6))
    
    val_x_log_6 = X_train.loc[val_,:]
    val_y_log_6 = y_train.loc[val_,:]
    

    log_clf_6 = LogisticRegression(C = 0.1,
                             max_iter = 2000, 
                             class_weight = 'balanced', 
                             penalty = 'l1', 
                             solver = 'liblinear', 
                             random_state = 0)
    log_clf_6_RFE = RFE(log_clf_6, 250, step = 1)
    log_clf_6_RFE.fit(trn_x_log_6, trn_y_log_6.ravel())
    
    oof_preds_log_6[val_] = log_clf_6_RFE.predict_proba(val_x_log_6)[:,1]
    
    fold_df_log_6 = pd.DataFrame(np.column_stack((oof_preds_log_6[val_], val_y_log_6)))
    fold_results_log_6 = pd.DataFrame(np.row_stack((fold_results_log_6, fold_df_log_6)))
    
    fpr, tpr, thresholds = metrics.roc_curve(val_y_log_6, oof_preds_log_6[val_])
    print("Log_6, Fold:", fold_, "oof-AUC: ", metrics.auc(fpr, tpr))
    
    sub_preds_log_6 += log_clf_6_RFE.predict_proba(X_testsub)[:,1] / splits
    
    ##Log_Model_7_OverSampled*6+Noise
    np.random.seed(0)
    noise = np.random.normal(0.1, 0.1, X_train.shape)
    
    X_train_noise = X_train + noise
    
    trn_x_log_7 = X_train_noise.loc[trn_,:]
    trn_x_log_7 = np.row_stack((trn_x_log_7, trn_x_log_7, trn_x_log_7, trn_x_log_7, trn_x_log_7, trn_x_log_7))
    trn_y_log_7 = y_train.loc[trn_,:]
    trn_y_log_7 = np.row_stack((trn_y_log_7, trn_y_log_7, trn_y_log_7, trn_y_log_7, trn_y_log_7, trn_y_log_7))
    
    val_x_log_7 = X_train_noise.loc[val_,:]
    val_y_log_7 = y_train.loc[val_,:]
    

    log_clf_7 = LogisticRegression(C = 0.1,
                             max_iter = 2000, 
                             class_weight = 'balanced', 
                             penalty = 'l1', 
                             solver = 'liblinear', 
                             random_state = 0)
    log_clf_7_RFE = RFE(log_clf_7, 250, step = 1)
    log_clf_7_RFE.fit(trn_x_log_7, trn_y_log_7.ravel())
    
    oof_preds_log_7[val_] = log_clf_7_RFE.predict_proba(val_x_log_7)[:,1]
    
    fold_df_log_7 = pd.DataFrame(np.column_stack((oof_preds_log_7[val_], val_y_log_7)))
    fold_results_log_7 = pd.DataFrame(np.row_stack((fold_results_log_7, fold_df_log_7)))
    
    fpr, tpr, thresholds = metrics.roc_curve(val_y_log_7, oof_preds_log_7[val_])
    print("Log_7, Fold:", fold_, "oof-AUC: ", metrics.auc(fpr, tpr))
    
    sub_preds_log_7 += log_clf_7_RFE.predict_proba(X_testsub)[:,1] / splits
    
    ##Log_Model_8_OverSampled*6scaled
    
    trn_x_log_8 = preprocessing.StandardScaler().fit_transform(X_train.loc[trn_,:])
    trn_x_log_8 = np.row_stack((trn_x_log_8, trn_x_log_8, trn_x_log_8, trn_x_log_8, trn_x_log_8, trn_x_log_8))
    trn_y_log_8 = y_train.loc[trn_,:]
    trn_y_log_8 = np.row_stack((trn_y_log_8, trn_y_log_8, trn_y_log_8, trn_y_log_8, trn_y_log_8, trn_y_log_8))
    
    val_x_log_8 = preprocessing.StandardScaler().fit_transform(X_train.loc[val_,:])
    val_y_log_8 = y_train.loc[val_,:]
    

    log_clf_8 = LogisticRegression(C = 0.1,
                             max_iter = 2000, 
                             class_weight = 'balanced', 
                             penalty = 'l1', 
                             solver = 'liblinear', 
                             random_state = 0)
    log_clf_8_RFE = RFE(log_clf_8, 250, step = 1)
    log_clf_8_RFE.fit(trn_x_log_8, trn_y_log_8.ravel())
    
    oof_preds_log_8[val_] = log_clf_8_RFE.predict_proba(val_x_log_8)[:,1]
    
    fold_df_log_8 = pd.DataFrame(np.column_stack((oof_preds_log_8[val_], val_y_log_8)))
    fold_results_log_8 = pd.DataFrame(np.row_stack((fold_results_log_8, fold_df_log_8)))
    
    fpr, tpr, thresholds = metrics.roc_curve(val_y_log_8, oof_preds_log_8[val_])
    print("Log_8, Fold:", fold_, "oof-AUC: ", metrics.auc(fpr, tpr))
    
    sub_preds_log_8 += log_clf_8_RFE.predict_proba(preprocessing.StandardScaler().fit_transform(X_testsub))[:,1] / splits
    
    ##Log_Model_9_RFE
    trn_x_log_9 = X_train.loc[trn_,:]
    val_x_log_9 = X_train.loc[val_,:]
    
    trn_y_log_9 = y_train.loc[trn_,:]
    val_y_log_9 = y_train.loc[val_,:]
    
    log_clf_9 = LogisticRegression(C = 0.1,
                             max_iter = 1000, 
                             class_weight = 'balanced', 
                             penalty = 'l1', 
                             solver = 'liblinear', 
                             random_state = 0)
    
    log_clf_9_RFE = RFE(log_clf_9, 260, step = 1)
    log_clf_9_RFE.fit(trn_x_log_9, trn_y_log_9.values.ravel())
    oof_preds_log_9[val_] = log_clf_9_RFE.predict_proba(val_x_log_9)[:,1]
    
    fold_df_log_9 = pd.DataFrame(np.column_stack((oof_preds_log_9[val_], val_y_log_9)))
    fold_results_log_9 = pd.DataFrame(np.row_stack((fold_results_log_9, fold_df_log_9)))
    
    fpr, tpr, thresholds = metrics.roc_curve(val_y_log_9, oof_preds_log_9[val_])
    print("Log_9, Fold:", fold_, "oof-AUC: ", metrics.auc(fpr, tpr))
    
    sub_preds_log_9 += log_clf_9_RFE.predict_proba(X_testsub)[:,1] / splits
    
    ##Log_Model_10_Smote
    smote = SMOTE(ratio = 'minority', n_jobs = -1)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train.values.ravel())
    
    X_train_smote = pd.DataFrame(X_train).reset_index(drop = True)
    y_train_smote = pd.DataFrame(y_train).reset_index(drop = True)

    trn_x_log_10 = X_train_smote.loc[trn_,:]
    val_x_log_10 = X_train_smote.loc[val_,:]
    
    trn_y_log_10 = y_train_smote.loc[trn_,:]
    val_y_log_10 = y_train_smote.loc[val_,:]
    
    log_clf_10 = LogisticRegression(C = 0.1,
                             max_iter = 1000, 
                             penalty = 'l1', 
                             solver = 'liblinear', 
                             random_state = 0)
    
    log_clf_10_RFE = RFE(log_clf_10, 250, step = 1)
    log_clf_10_RFE.fit(trn_x_log_10, trn_y_log_10.values.ravel())
    oof_preds_log_10[val_] = log_clf_10_RFE.predict_proba(val_x_log_10)[:,1]
    
    fold_df_log_10 = pd.DataFrame(np.column_stack((oof_preds_log_10[val_], val_y_log_10)))
    fold_results_log_10 = pd.DataFrame(np.row_stack((fold_results_log_10, fold_df_log_10)))
    
    fpr, tpr, thresholds = metrics.roc_curve(val_y_log_10, oof_preds_log_10[val_])
    print("Log_10, Fold:", fold_, "oof-AUC: ", metrics.auc(fpr, tpr))
    
    sub_preds_log_10 += log_clf_10_RFE.predict_proba(X_testsub)[:,1] / splits
    
    ##Log_Model_11_noise
    np.random.seed(0)
    noise = np.random.normal(0.15, 0.15, X_train.shape)
    
    X_train_noise = X_train + noise

    trn_x_log_11 = X_train_noise.loc[trn_,:]
    val_x_log_11 = X_train_noise.loc[val_,:]
    
    trn_y_log_11 = y_train.loc[trn_,:]
    val_y_log_11 = y_train.loc[val_,:]
    
    
    log_clf_11 = LogisticRegression(C = 0.1,
                             max_iter = 2000, 
                             class_weight = 'balanced', 
                             penalty = 'l1', 
                             solver = 'liblinear', 
                             random_state = 0)
    log_clf_11_RFE = RFE(log_clf_11, 250, step = 1)
    log_clf_11_RFE.fit(trn_x_log_11, trn_y_log_11.values.ravel())
    
    oof_preds_log_11[val_] = log_clf_11_RFE.predict_proba(val_x_log_11)[:,1]
    
    fold_df_log_11 = pd.DataFrame(np.column_stack((oof_preds_log_11[val_], val_y_log_11)))
    fold_results_log_11 = pd.DataFrame(np.row_stack((fold_results_log_11, fold_df_log_11)))
    
    fpr, tpr, thresholds = metrics.roc_curve(val_y_log_11, oof_preds_log_11[val_])
    print("Log_11, Fold:", fold_, "oof-AUC: ", metrics.auc(fpr, tpr))
    
    sub_preds_log_11 += log_clf_11_RFE.predict_proba(X_testsub)[:,1] / splits
    
    ##12_noise
    np.random.seed(0)
    noise = np.random.normal(0.2, 0.2, X_train.shape)
    
    X_train_noise = X_train + noise

    trn_x_log_12 = X_train_noise.loc[trn_,:]
    val_x_log_12 = X_train_noise.loc[val_,:]
    
    trn_y_log_12 = y_train.loc[trn_,:]
    val_y_log_12 = y_train.loc[val_,:]
    
    
    log_clf_12 = LogisticRegression(C = 0.1,
                             max_iter = 2000, 
                             class_weight = 'balanced', 
                             penalty = 'l1', 
                             solver = 'liblinear', 
                             random_state = 0)
    log_clf_12_RFE = RFE(log_clf_12, 250, step = 1)
    log_clf_12_RFE.fit(trn_x_log_12, trn_y_log_12.values.ravel())
    
    oof_preds_log_12[val_] = log_clf_12_RFE.predict_proba(val_x_log_12)[:,1]
    
    fold_df_log_12 = pd.DataFrame(np.column_stack((oof_preds_log_12[val_], val_y_log_12)))
    fold_results_log_12 = pd.DataFrame(np.row_stack((fold_results_log_12, fold_df_log_12)))
    
    fpr, tpr, thresholds = metrics.roc_curve(val_y_log_12, oof_preds_log_12[val_])
    print("Log_12, Fold:", fold_, "oof-AUC: ", metrics.auc(fpr, tpr))
    
    sub_preds_log_12 += log_clf_12_RFE.predict_proba(X_testsub)[:,1] / splits
    
    ##noise_13
    np.random.seed(0)
    noise = np.random.normal(0.2, 0.2, X_train.shape)
    
    X_train_noise = X_train + noise

    trn_x_log_13 = X_train_noise.loc[trn_,:]
    val_x_log_13 = X_train_noise.loc[val_,:]
    
    trn_y_log_13 = y_train.loc[trn_,:]
    val_y_log_13 = y_train.loc[val_,:]
    
    
    log_clf_13 = LogisticRegression(C = 0.1,
                             max_iter = 2000, 
                             class_weight = 'balanced', 
                             penalty = 'l1', 
                             solver = 'liblinear', 
                             random_state = 0)
    log_clf_13_RFE = RFE(log_clf_13, 250, step = 1)
    log_clf_13_RFE.fit(trn_x_log_13, trn_y_log_13.values.ravel())
    
    oof_preds_log_13[val_] = log_clf_13_RFE.predict_proba(val_x_log_13)[:,1]
    
    fold_df_log_13 = pd.DataFrame(np.column_stack((oof_preds_log_13[val_], val_y_log_13)))
    fold_results_log_13 = pd.DataFrame(np.row_stack((fold_results_log_13, fold_df_log_13)))
    
    fpr, tpr, thresholds = metrics.roc_curve(val_y_log_13, oof_preds_log_13[val_])
    print("Log_13, Fold:", fold_, "oof-AUC: ", metrics.auc(fpr, tpr))
    
    sub_preds_log_13 += log_clf_13_RFE.predict_proba(X_testsub)[:,1] / splits
    

    
############################################################################################################3
##Models Ens
                                          
log_preds = pd.DataFrame(np.column_stack((fold_results_log_1.loc[:, 0], fold_results_log_2.loc[:, 0], fold_results_log_3.loc[:, 0],
                                          fold_results_log_4.loc[:, 0], fold_results_log_5.loc[:, 0], fold_results_log_6.loc[:, 0],
                                          fold_results_log_7.loc[:, 0], fold_results_log_8.loc[:, 0], fold_results_log_9.loc[:, 0],
                                          fold_results_log_11.loc[:, 0], fold_results_log_12.loc[:, 0], fold_results_log_13.loc[:, 0],
                                          fold_results_log_10.loc[:, 0], fold_results_log_3.loc[:, 1])))
log_preds.columns = ['Log_1', 'Log_2', 'Log_3', 'Log_4', 'Log_5', 
                     'Log_6', 'Log_7', 'Log_8', 'Log_9', 
                     'Log_11', 'Log_12', 'Log_13', 'Log_10', 'target']                              
testSub_preds = pd.DataFrame(np.column_stack((sub_preds_log_1, sub_preds_log_2, sub_preds_log_3, sub_preds_log_4,
                                              sub_preds_log_5, sub_preds_log_6, sub_preds_log_7, sub_preds_log_8,
                                              sub_preds_log_9, sub_preds_log_11, sub_preds_log_12, sub_preds_log_13,
                                              sub_preds_log_10)))
testSub_preds.columns = ['Log_1', 'Log_2', 'Log_3', 'Log_4', 'Log_5', 
                         'Log_6', 'Log_7', 'Log_8', 'Log_9', 'Log_11', 'Log_12', 'Log_13', 'Log_10']
X_train_ens = log_preds[log_preds.columns.difference(['target'])]
y_train_ens = log_preds[['target']]
y_train_ens = y_train_ens.astype('int')

#############################################################################################
##LogRegression Ensemble

splits = 5
folds = StratifiedKFold(n_splits = splits, random_state = 0)
oof_preds_ens = np.zeros(X_train_ens.shape[0])
sub_preds_ens_log = np.zeros(testSub_preds.shape[0])
AUC = 0
for fold_, (trn_, val_) in enumerate(folds.split(X_train_ens, y_train_ens)):
    trn_x = X_train_ens.loc[trn_,:]
    val_x = X_train_ens.loc[val_,:]
    trn_y = y_train_ens.loc[trn_,:]
    val_y = y_train_ens.loc[val_,:]
    log_ens_clf = LogisticRegression(C = 1, 
                                     solver = 'liblinear',
                                     penalty = 'l2',
                                     random_state = 0)
    
    log_ens_clf.fit(trn_x, trn_y.values.ravel())
    oof_preds_ens[val_] = log_ens_clf.predict_proba(val_x)[:,1]
    sub_preds_ens_log += log_ens_clf.predict_proba(testSub_preds)[:,1] / splits
    fpr, tpr, thresholds = metrics.roc_curve(val_y, oof_preds_ens[val_])
    print("Fold:", fold_, "oof-AUC: ", metrics.auc(fpr, tpr))
    AUC += metrics.auc(fpr, tpr) / splits
print(AUC)

model_1_ens_lr = pd.concat([pd.DataFrame(testSub_Ids), pd.DataFrame(sub_preds_ens_log)], axis = 1)
model_1_ens_lr.columns = ['id', 'target']
model_1_ens_lr.to_csv(r'D:/Python/data/DontOverFit/5x5x5-8101.csv', index = False)

#############################################################################################
##Random Forest Ensemble
splits = 5
folds = StratifiedKFold(n_splits = splits, random_state = 0)
oof_preds_ens = np.zeros(X_train_ens.shape[0])
AUC = 0
sub_preds_ens_rf = np.zeros(testSub_preds.shape[0])
for fold_, (trn_, val_) in enumerate(folds.split(X_train_ens, y_train_ens)):
    trn_x = X_train_ens.loc[trn_,:]
    val_x = X_train_ens.loc[val_,:]
    trn_y = y_train_ens.loc[trn_,:]
    val_y = y_train_ens.loc[val_,:]
    clf_2_rf = RandomForestClassifier(random_state = 0, 
                                      n_estimators = 400,
                                      class_weight = 'balanced', criterion = 'gini', bootstrap = 'false',
                                      max_depth = 4,
                                      max_features = 2,
                                      min_samples_leaf = 4,
                                      min_samples_split = 2,
                                      min_impurity_decrease = 0.1,
                                      n_jobs = -1)
    clf_2_rf.fit(trn_x, trn_y.values.ravel())
    oof_preds_ens[val_] = clf_2_rf.predict_proba(val_x)[:,1]
    sub_preds_ens_rf += clf_2_rf.predict_proba(testSub_preds)[:,1] / splits
    fpr, tpr, thresholds = metrics.roc_curve(val_y, oof_preds_ens[val_])
    print("Fold:", fold_, "oof-AUC: ", metrics.auc(fpr, tpr))
    AUC += metrics.auc(fpr, tpr) / splits
print(AUC)

model_1_ens_rf = pd.concat([pd.DataFrame(testSub_Ids), pd.DataFrame(sub_preds_ens_rf)], axis = 1)
model_1_ens_rf.columns = ['id', 'target']
model_1_ens_rf.to_csv(r'D:/Python/data/DontOverFit/10LR-5Fold-RFEns.csv', index = False)


model_1_ens = pd.concat([pd.DataFrame(testSub_Ids), pd.DataFrame((sub_preds_ens_rf + sub_preds_ens_log)/2)], axis = 1)
model_1_ens.columns = ['id', 'target']
model_1_ens.to_csv(r'D:/Python/data/DontOverFit/10LR-5Fold-RFEnsLREnsAVG.csv', index = False)

#############################################################################################
#clf_2_rf = RandomForestClassifier(random_state = 0, 
#                                      n_estimators = 150,
#                                      class_weight = 'balanced', criterion = 'gini', bootstrap = 'false',
#                                      max_depth = 2,
#                                      max_features = 6,
#                                      min_samples_leaf = 14,
#                                      min_samples_split = 2,
#                                      min_impurity_decrease = 0.0,
#                                      n_jobs = -1)
#Fold: 0 oof-AUC:  0.7621527777777777
#Fold: 1 oof-AUC:  0.8159722222222222
#Fold: 2 oof-AUC:  0.828125
#Fold: 3 oof-AUC:  0.8524305555555556
#Fold: 4 oof-AUC:  0.8072916666666666
#0.8131944444444443
#############################################################################################
#testSub_preds.head()
#testSubEns = pd.concat([pd.DataFrame(testSub_Ids), testSub_preds], axis = 1)
#maxDF = testSubEns[testSubEns['Log_Mean'] >= 0.64]
#minDF = testSubEns[testSubEns['Log_Mean'] < 0.64]
#minTarDF = minDF.loc[:, ['Log_1', 'Log_2', 'Log_3', 'Log_4', 'Log_5', 
#                                 'Log_6', 'Log_7', 'Log_8', 'Log_9', 'Log_10', 'Log_Mean']].min(axis = 1)
#maxTarDF = maxDF.loc[:, ['Log_1', 'Log_2', 'Log_3', 'Log_4', 'Log_5', 
#                                 'Log_6', 'Log_7', 'Log_8', 'Log_9', 'Log_10', 'Log_Mean']].max(axis = 1)
#maxDFIds = maxDF['id']
#minDFIds = minDF['id']
#minMaxSub = pd.DataFrame(np.row_stack((pd.DataFrame(np.column_stack((minDFIds, minTarDF))), 
#                                       pd.DataFrame(np.column_stack((maxDFIds, maxTarDF))))))
#minMaxSub.columns = ['id', 'target']
#minMaxSub.id = minMaxSub.id.astype(int)
#minMaxSub.to_csv(r'D:/Python/data/DontOverFit/MinMaxSub.csv', index = False)






























