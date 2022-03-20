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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from lightgbm import LGBMClassifier
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
X_train = train[train.columns.difference(['target', 'id'])]
y_train = train[['target']]

testSub = pd.read_csv("D:/Python/data/DontOverFit/test.csv")
X_testsub = testSub[testSub.columns.difference(['id'])]
testSub_Ids = testSub[['id']]

pd.options.display.max_columns = train.shape[1]
pd.options.display.max_rows = 100
X_train.head()

#X_train, X_test, y_train, y_test = train_test_split(train_values, train_target, test_size = 0.1, random_state = 0, stratify = train_target)
#
#X_train = pd.DataFrame(X_train).reset_index(drop = True)
#X_test = pd.DataFrame(X_test).reset_index(drop = True)
#y_train = pd.DataFrame(y_train).reset_index(drop = True)
#y_test = pd.DataFrame(y_test).reset_index(drop = True)




oof_preds_log = np.zeros(X_train.shape[0])
fold_results_log = pd.DataFrame(columns = ['Prediction', 'target'])
sub_preds_log = np.zeros(X_testsub.shape[0])

oof_preds_svc = np.zeros(X_train.shape[0])
fold_results_svc = pd.DataFrame(columns = ['Prediction', 'target'])
sub_preds_svc = np.zeros(X_testsub.shape[0])

oof_preds_nb = np.zeros(X_train.shape[0])
fold_results_nb = pd.DataFrame(columns = ['Prediction', 'target'])
sub_preds_nb = np.zeros(X_testsub.shape[0])

oof_preds_rf = np.zeros(X_train.shape[0])
fold_results_rf = pd.DataFrame(columns = ['Prediction', 'target'])
sub_preds_rf = np.zeros(X_testsub.shape[0])

oof_preds_knn = np.zeros(X_train.shape[0])
fold_results_knn = pd.DataFrame(columns = ['Prediction', 'target'])
sub_preds_knn = np.zeros(X_testsub.shape[0])

oof_preds_lgbm = np.zeros(X_train.shape[0])
fold_results_lgbm = pd.DataFrame(columns = ['Prediction', 'target'])
sub_preds_lgbm = np.zeros(X_testsub.shape[0])

splits = 5
folds = StratifiedKFold(n_splits = splits, random_state = 0)

for fold_, (trn_, val_) in enumerate(folds.split(X_train, y_train)):
    
    trn_x = X_train.loc[trn_,:]
    trn_y = y_train.loc[trn_,:]
    val_x = X_train.loc[val_,:]
    val_y= y_train.loc[val_,:]
    
    ##Log_Model
    log_clf_1 = LogisticRegression(C = 0.1, max_iter = 1000, penalty = 'l1', solver = 'liblinear', random_state = 0)
    log_clf_1_RFE = RFE(log_clf_1, 200, step = 1)
    log_clf_1_RFE.fit(trn_x, trn_y.values.ravel())
    oof_preds_log[val_] = log_clf_1_RFE.predict_proba(val_x)[:,1]
    fold_df_log = pd.DataFrame(np.column_stack((oof_preds_log[val_], val_y)))
    fold_results_log = pd.DataFrame(np.row_stack((fold_results_log, fold_df_log)))
    fpr, tpr, thresholds = metrics.roc_curve(val_y, oof_preds_log[val_])
    print("Log, Fold:", fold_, "oof-AUC: ", metrics.auc(fpr, tpr))
    sub_preds_log += log_clf_1_RFE.predict_proba(X_testsub)[:,1] / splits
    
    ##SVC_Model
    svc = SVC(kernel = 'rbf', C = 1.0, gamma = 'auto', probability = True)
    svc.fit(trn_x, trn_y.values.ravel())
    oof_preds_svc[val_] = svc.predict_proba(val_x)[:,1]
    fold_df_svc = pd.DataFrame(np.column_stack((oof_preds_svc[val_], val_y)))
    fold_results_svc = pd.DataFrame(np.row_stack((fold_results_svc, fold_df_svc)))
    fpr, tpr, thresholds = metrics.roc_curve(val_y, oof_preds_svc[val_])
    print("SVC, Fold:", fold_, "oof-AUC: ", metrics.auc(fpr, tpr))
    sub_preds_svc += svc.predict_proba(X_testsub)[:,1] / splits
    
    ##NB_Model
    nb = GaussianNB()
    nb.fit(trn_x, trn_y.values.ravel())
    oof_preds_nb[val_] = nb.predict_proba(val_x)[:,1]
    fold_df_nb = pd.DataFrame(np.column_stack((oof_preds_nb[val_], val_y)))
    fold_results_nb = pd.DataFrame(np.row_stack((fold_results_nb, fold_df_nb)))
    fpr, tpr, thresholds = metrics.roc_curve(val_y, oof_preds_nb[val_])
    print("NB, Fold:", fold_, "oof-AUC: ", metrics.auc(fpr, tpr))
    sub_preds_nb += nb.predict_proba(X_testsub)[:,1] / splits
    
    ##RF_Model
    rf = RandomForestClassifier(n_estimators = 400, max_depth = 2, random_state = 0)
    rf.fit(trn_x, trn_y.values.ravel())
    oof_preds_rf[val_] = rf.predict_proba(val_x)[:,1]
    fold_df_rf = pd.DataFrame(np.column_stack((oof_preds_rf[val_], val_y)))
    fold_results_rf = pd.DataFrame(np.row_stack((fold_results_rf, fold_df_rf)))
    fpr, tpr, thresholds = metrics.roc_curve(val_y, oof_preds_rf[val_])
    print("RF, Fold:", fold_, "oof-AUC: ", metrics.auc(fpr, tpr))
    sub_preds_rf += rf.predict_proba(X_testsub)[:,1] / splits
    
    ##KNN_Model
    knn = KNeighborsClassifier(n_neighbors = 63)
    knn.fit(trn_x, trn_y.values.ravel())
    oof_preds_knn[val_] = knn.predict_proba(val_x)[:,1]
    fold_df_knn = pd.DataFrame(np.column_stack((oof_preds_knn[val_], val_y)))
    fold_results_knn = pd.DataFrame(np.row_stack((fold_results_knn, fold_df_knn)))
    fpr, tpr, thresholds = metrics.roc_curve(val_y, oof_preds_knn[val_])
    print("KNN, Fold:", fold_, "oof-AUC: ", metrics.auc(fpr, tpr))
    sub_preds_knn += knn.predict_proba(X_testsub)[:,1] / splits
    
    ##LGBM_Model
    lgbm_clf = LGBMClassifier(objective = 'binary', n_estimators = 500, metric = 'auc',
                      learning_rate = 0.05, max_depth = 3, reg_alpha = 0.75, reg_lambda = 0.75)
    lgbm_clf.fit(trn_x, trn_y.values.ravel())
    oof_preds_lgbm[val_] = lgbm_clf.predict_proba(val_x)[:,1]
    fold_df_lgbm = pd.DataFrame(np.column_stack((oof_preds_lgbm[val_], val_y)))
    fold_results_lgbm = pd.DataFrame(np.row_stack((fold_results_lgbm, fold_df_lgbm)))
    fpr, tpr, thresholds = metrics.roc_curve(val_y, oof_preds_lgbm[val_])
    print("LGBM, Fold:", fold_, "oof-AUC: ", metrics.auc(fpr, tpr))
    sub_preds_lgbm += lgbm_clf.predict_proba(X_testsub)[:,1] / splits
    
############################################################################################################3
##Models Ens
ens_preds = pd.DataFrame(np.column_stack((fold_results_svc.loc[:, 0], fold_results_nb.loc[:,0], fold_results_rf.loc[:,0],
                              fold_results_knn.loc[:,0], fold_results_lgbm.loc[:,0], fold_results_log)))
ens_preds.columns = ['SVC', 'NB', 'RF', 'KNN', 'LGBM', 'Log', 'target']

                                              
testSub_preds = pd.DataFrame(np.column_stack((sub_preds_svc, sub_preds_nb, sub_preds_rf, sub_preds_knn, sub_preds_lgbm, sub_preds_log)))
testSub_preds.columns = ['SVC', 'NB', 'RF', 'KNN', 'LGBM', 'Log']

X_train_ens = ens_preds[ens_preds.columns.difference(['target'])]
y_train_ens = ens_preds[['target']]
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
    log_ens_clf = LogisticRegression(C = 240, 
                                     solver = 'liblinear',
                                     penalty = 'l1',
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
model_1_ens_lr.to_csv(r'D:/Python/data/DontOverFit/KFoldEns-2LREns.csv', index = False)

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
                                      n_estimators = 600,
                                      class_weight = 'balanced', criterion = 'gini', bootstrap = 'false',
                                      max_depth = 2,
                                      max_features = 2,
                                      min_samples_leaf = 1,
                                      min_samples_split = 2,
                                      min_impurity_decrease = 0.0,
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
model_1_ens_rf.to_csv(r'D:/Python/data/DontOverFit/KFoldEns-2RFEns.csv', index = False)

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






























