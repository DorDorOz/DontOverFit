import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn import  metrics
from sklearn import preprocessing
from sklearn.utils import class_weight
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
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



train = pd.read_csv("D:/Python/data/DontOverFit/train.csv")
testSub = pd.read_csv("D:/Python/data/DontOverFit/test.csv")

pd.options.display.max_columns = testSub.shape[1]

train_values = train[train.columns.difference(['target', 'id'])]
train_target = train[['target']]

train = pd.concat([pd.DataFrame(train), 
                     pd.DataFrame(train),
                     pd.DataFrame(train),
                     pd.DataFrame(train),
                     pd.DataFrame(train),
                     pd.DataFrame(train),
                     pd.DataFrame(train),
                     pd.DataFrame(train)], 
                     axis = 0)

train = shuffle(train, random_state = 0)

X_train = train[train.columns.difference(['target', 'id'])]
y_train = train[['target']]

smote = SMOTE(ratio = 'minority', n_jobs = -1)
X_train, y_train = smote.fit_resample(X_train, y_train.values.ravel())


testSub_values = testSub[testSub.columns.difference(['id'])]
testSub_Ids = testSub[['id']]

X_train, X_test, y_train, y_test = train_test_split(train_values, train_target, test_size = 0.3, random_state = 0, stratify = train_target)

#X_train = preprocessing.StandardScaler().fit_transform(X_train)
#X_test = preprocessing.StandardScaler().fit_transform(X_test)

X_train = preprocessing.MinMaxScaler().fit_transform(X_train)
X_test = preprocessing.MinMaxScaler().fit_transform(X_test)


rf = RandomForestClassifier(random_state = 0, 
                            n_estimators = 1000,
                            max_depth = 4,
                            bootstrap = False,
                            class_weight = 'balanced',
                            criterion = 'gini',
                            max_features = 'auto',
                            min_samples_leaf = 1,
                            min_samples_split = 6,
                            min_impurity_decrease = 0.02,
                            n_jobs = -1)
rf.fit(X_train, y_train.values.ravel())
probs = rf.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, pd.DataFrame(probs)[1])
print(metrics.auc(fpr, tpr))

feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = pd.DataFrame(X_train).columns,
                                   columns = ['importance']).sort_values('importance', ascending = False)

#plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc)
#plt.legend()
#plt.title('ROC curve')
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.grid(True)




##############################################################################################

param_grid = { 
    'n_estimators': [400, 600, 800],
    'min_samples_leaf' : [1, 2, 4, 6],
    'max_features' : [2, 4, 6],
    'min_samples_split' : [2, 3],
    'min_impurity_decrease' : [0.0, 0.1, 0.2],
    'max_depth' : [2, 4, 6],
    'bootstrap' : [False],
    'class_weight' : ['balanced'],
    'criterion' : ['gini']}

CV_rf_model_1 = GridSearchCV(estimator = RandomForestClassifier(random_state = 0),
                             param_grid = param_grid, 
                             cv = 5, 
                             n_jobs = -1,
                             verbose = 10)

CV_rf_model_1.fit(X_train_ens, y_train_ens.values.ravel())
CV_rf_model_1.best_params_

##############################################################################################
train = pd.read_csv("D:/Python/data/DontOverFit/train.csv")
testSub = pd.read_csv("D:/Python/data/DontOverFit/test.csv")

train_values = train[train.columns.difference(['target', 'id'])]
train_target = train[['target']]
train_values = pd.DataFrame(preprocessing.StandardScaler().fit_transform(train_values))

X_train, X_test, y_train, y_test = train_test_split(train_values, train_target, test_size = 0.1, random_state = 0, stratify = train_target)

X_train = pd.DataFrame(X_train.reset_index(drop = True))
X_test = pd.DataFrame(X_test.reset_index(drop = True))
y_train = pd.DataFrame(y_train.reset_index(drop = True))
y_test = pd.DataFrame(y_test.reset_index(drop = True))

splits = 20
folds = StratifiedKFold(n_splits = splits, random_state = 0, shuffle = True)

oof_preds_rf = np.zeros(X_train.shape[0])
sub_preds_rf = np.zeros(X_test.shape[0])

for fold_, (trn_, val_) in enumerate(folds.split(X_train, y_train)):
    trn_x = X_train.loc[trn_,:]
    val_x = X_train.loc[val_,:]
    
    trn_y = y_train.loc[trn_,:]
    val_y = y_train.loc[val_,:]
    
    print("Fold", fold_, "RF")
    clf_2_rf = RandomForestClassifier(random_state = 0, 
                                      n_estimators = 2000,
                                      n_jobs = -1)
    clf_2_rf.fit(trn_x, trn_y.values.ravel())
    oof_preds_rf[val_] = clf_2_rf.predict_proba(val_x)[:,1]
    sub_preds_rf += clf_2_rf.predict_proba(X_test)[:,1] / splits



fpr, tpr, thresholds = metrics.roc_curve(y_test, pd.DataFrame(sub_preds_rf))
print(metrics.auc(fpr, tpr))

plt.plot(fpr, tpr)
plt.legend()
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)







#
#    ##RF_1
#    trn_x_rf_1 = X_train.loc[trn_,:]
#    val_x_rf_1 = X_train.loc[val_,:]
#    
#    trn_y_rf_1 = y_train.loc[trn_,:]
#    val_y_rf_1 = y_train.loc[val_,:]
#    
#    rf_clf_1 = RandomForestClassifier(random_state = 0, 
#                                      n_estimators = 1000,
#                                      max_depth = 2,
#                                      bootstrap = False,
#                                      class_weight = 'balanced',
#                                      criterion = 'gini',
#                                      min_samples_leaf = 2,
#                                      min_samples_split = 2,
#                                      min_impurity_decrease = 0.01,
#                                      n_jobs = -1)
#    rf_clf_1_RFE = RFE(rf_clf_1, 150, step = 5)
#    rf_clf_1_RFE.fit(trn_x_rf_1, trn_y_rf_1.values.ravel())
#    
#    oof_preds_rf_1[val_] = rf_clf_1_RFE.predict_proba(val_x_rf_1)[:,1]
#    
#    fold_df_rf_1 = pd.DataFrame(np.column_stack((oof_preds_rf_1[val_], val_y_rf_1)))
#    fold_results_rf_1 = pd.DataFrame(np.row_stack((fold_results_rf_1, fold_df_rf_1)))
#    
#    fpr, tpr, thresholds = metrics.roc_curve(val_y_rf_1, oof_preds_rf_1[val_])
#    print("RF_1, Fold:", fold_, "oof-AUC: ", metrics.auc(fpr, tpr))
#    
#    sub_preds_rf_1 += rf_clf_1_RFE.predict_proba(X_testsub)[:,1] / splits
#    
#    ##RF_2
#    trn_x_rf_2 = X_train.loc[trn_,:]
#    val_x_rf_2 = X_train.loc[val_,:]
#    
#    trn_y_rf_2 = y_train.loc[trn_,:]
#    val_y_rf_2 = y_train.loc[val_,:]
#    
#    rf_clf_2 = RandomForestClassifier(random_state = 0, 
#                                      n_estimators = 1000,
#                                      max_depth = 4,
#                                      bootstrap = False,
#                                      class_weight = 'balanced',
#                                      criterion = 'gini',
#                                      max_features = 'auto',
#                                      min_samples_leaf = 2,
#                                      min_samples_split = 6,
#                                      min_impurity_decrease = 0.01,
#                                      n_jobs = -1)
#    rf_clf_2_RFE = RFE(rf_clf_2, 150, step = 5)
#    rf_clf_2_RFE.fit(trn_x_rf_2, trn_y_rf_2.values.ravel())
#    
#    oof_preds_rf_2[val_] = rf_clf_2_RFE.predict_proba(val_x_rf_2)[:,1]
#    
#    fold_df_rf_2 = pd.DataFrame(np.column_stack((oof_preds_rf_2[val_], val_y_rf_2)))
#    fold_results_rf_2 = pd.DataFrame(np.row_stack((fold_results_rf_2, fold_df_rf_2)))
#    
#    fpr, tpr, thresholds = metrics.roc_curve(val_y_rf_2, oof_preds_rf_2[val_])
#    print("RF_2, Fold:", fold_, "oof-AUC: ", metrics.auc(fpr, tpr))
#    
#    sub_preds_rf_2 += rf_clf_2_RFE.predict_proba(X_testsub)[:,1] / splits
#    
#    ##RF_3
#    trn_x_rf_3 = X_train.loc[trn_,:]
#    val_x_rf_3 = X_train.loc[val_,:]
#    
#    trn_y_rf_3 = y_train.loc[trn_,:]
#    val_y_rf_3 = y_train.loc[val_,:]
#    
#    rf_clf_3 = RandomForestClassifier(random_state = 0, 
#                                      n_estimators = 1000,
#                                      max_depth = 6,
#                                      bootstrap = False,
#                                      class_weight = 'balanced',
#                                      criterion = 'gini',
#                                      max_features = 'auto',
#                                      min_samples_leaf = 6,
#                                      min_samples_split = 2,
#                                      min_impurity_decrease = 0.0,
#                                      n_jobs = -1)
#    rf_clf_3_RFE = RFE(rf_clf_3, 150, step = 5)
#    rf_clf_3_RFE.fit(trn_x_rf_3, trn_y_rf_3.values.ravel())
#    
#    oof_preds_rf_3[val_] = rf_clf_3_RFE.predict_proba(val_x_rf_3)[:,1]
#    
#    fold_df_rf_3 = pd.DataFrame(np.column_stack((oof_preds_rf_3[val_], val_y_rf_3)))
#    fold_results_rf_3 = pd.DataFrame(np.row_stack((fold_results_rf_3, fold_df_rf_3)))
#    
#    fpr, tpr, thresholds = metrics.roc_curve(val_y_rf_3, oof_preds_rf_3[val_])
#    print("RF_3, Fold:", fold_, "oof-AUC: ", metrics.auc(fpr, tpr))
#    
#    sub_preds_rf_3 += rf_clf_3_RFE.predict_proba(X_testsub)[:,1] / splits
#    
#    ##RF_4
#    trn_x_rf_4 = X_train.loc[trn_,:]
#    val_x_rf_4 = X_train.loc[val_,:]
#    
#    trn_y_rf_4 = y_train.loc[trn_,:]
#    val_y_rf_4 = y_train.loc[val_,:]
#    
#    rf_clf_4 = RandomForestClassifier(random_state = 0, 
#                                      n_estimators = 1000,
#                                      max_depth = 8,
#                                      bootstrap = False,
#                                      class_weight = 'balanced',
#                                      criterion = 'gini',
#                                      max_features = 'auto',
#                                      min_samples_leaf = 6,
#                                      min_samples_split = 2,
#                                      min_impurity_decrease = 0.01,
#                                      n_jobs = -1)
#    rf_clf_4_RFE = RFE(rf_clf_4, 150, step = 5)
#    rf_clf_4_RFE.fit(trn_x_rf_4, trn_y_rf_4.values.ravel())
#    
#    oof_preds_rf_4[val_] = rf_clf_4_RFE.predict_proba(val_x_rf_4)[:,1]
#    
#    fold_df_rf_4 = pd.DataFrame(np.column_stack((oof_preds_rf_4[val_], val_y_rf_4)))
#    fold_results_rf_4 = pd.DataFrame(np.row_stack((fold_results_rf_4, fold_df_rf_4)))
#    
#    fpr, tpr, thresholds = metrics.roc_curve(val_y_rf_4, oof_preds_rf_4[val_])
#    print("RF_4, Fold:", fold_, "oof-AUC: ", metrics.auc(fpr, tpr))
#    
#    sub_preds_rf_4 += rf_clf_4_RFE.predict_proba(X_testsub)[:,1] / splits
#    
#    ##RF_5_ScaledMinMax
#    trn_x_rf_5 = preprocessing.MinMaxScaler().fit_transform(X_train.loc[trn_,:])
#    val_x_rf_5 = preprocessing.MinMaxScaler().fit_transform(X_train.loc[val_,:])
#    
#    trn_y_rf_5 = y_train.loc[trn_,:]
#    val_y_rf_5 = y_train.loc[val_,:]
#    
#    rf_clf_5 = RandomForestClassifier(random_state = 0, 
#                                      n_estimators = 1000,
#                                      max_depth = 2,
#                                      bootstrap = False,
#                                      class_weight = 'balanced',
#                                      criterion = 'gini',
#                                      max_features = 'auto',
#                                      min_samples_leaf = 2,
#                                      min_samples_split = 2,
#                                      min_impurity_decrease = 0.01,
#                                      n_jobs = -1)
#    rf_clf_5_RFE = RFE(rf_clf_5, 150, step = 5)
#    rf_clf_5_RFE.fit(trn_x_rf_5, trn_y_rf_5.values.ravel())
#    
#    oof_preds_rf_5[val_] = rf_clf_5_RFE.predict_proba(val_x_rf_5)[:,1]
#    
#    fold_df_rf_5 = pd.DataFrame(np.column_stack((oof_preds_rf_5[val_], val_y_rf_5)))
#    fold_results_rf_5 = pd.DataFrame(np.row_stack((fold_results_rf_5, fold_df_rf_5)))
#    
#    fpr, tpr, thresholds = metrics.roc_curve(val_y_rf_5, oof_preds_rf_5[val_])
#    print("RF_5, Fold:", fold_, "oof-AUC: ", metrics.auc(fpr, tpr))
#    
#    sub_preds_rf_5 += rf_clf_5_RFE.predict_proba(preprocessing.MinMaxScaler().fit_transform(X_testsub))[:,1] / splits
#    
#    ##RF_6_ScaledStandard
#    trn_x_rf_6 = preprocessing.StandardScaler().fit_transform(X_train.loc[trn_,:])
#    val_x_rf_6 = preprocessing.StandardScaler().fit_transform(X_train.loc[val_,:])
#    
#    trn_y_rf_6 = y_train.loc[trn_,:]
#    val_y_rf_6 = y_train.loc[val_,:]
#    
#    rf_clf_6 = RandomForestClassifier(random_state = 0, 
#                                      n_estimators = 1000,
#                                      max_depth = 2,
#                                      bootstrap = False,
#                                      class_weight = 'balanced',
#                                      criterion = 'gini',
#                                      max_features = 'auto',
#                                      min_samples_leaf = 2,
#                                      min_samples_split = 2,
#                                      min_impurity_decrease = 0.01,
#                                      n_jobs = -1)
#    rf_clf_6_RFE = RFE(rf_clf_6, 150, step = 5)
#    rf_clf_6_RFE.fit(trn_x_rf_6, trn_y_rf_6.values.ravel())
#    
#    oof_preds_rf_6[val_] = rf_clf_6_RFE.predict_proba(val_x_rf_6)[:,1]
#    
#    fold_df_rf_6 = pd.DataFrame(np.column_stack((oof_preds_rf_6[val_], val_y_rf_6)))
#    fold_results_rf_6 = pd.DataFrame(np.row_stack((fold_results_rf_6, fold_df_rf_6)))
#    
#    fpr, tpr, thresholds = metrics.roc_curve(val_y_rf_6, oof_preds_rf_6[val_])
#    print("RF_6, Fold:", fold_, "oof-AUC: ", metrics.auc(fpr, tpr))
#    
#    sub_preds_rf_6 += rf_clf_6_RFE.predict_proba(preprocessing.StandardScaler().fit_transform(X_testsub))[:,1] / splits
#
#
#
#fold_results_rf_1 = pd.DataFrame(columns = ['Prediction', 'target'])
#fold_results_rf_2 = pd.DataFrame(columns = ['Prediction', 'target'])
#fold_results_rf_3 = pd.DataFrame(columns = ['Prediction', 'target'])
#fold_results_rf_4 = pd.DataFrame(columns = ['Prediction', 'target'])
#fold_results_rf_5 = pd.DataFrame(columns = ['Prediction', 'target'])
#fold_results_rf_6 = pd.DataFrame(columns = ['Prediction', 'target'])
#sub_preds_rf_1 = np.zeros(X_testsub.shape[0])
#sub_preds_rf_2 = np.zeros(X_testsub.shape[0])
#sub_preds_rf_3 = np.zeros(X_testsub.shape[0])
#sub_preds_rf_4 = np.zeros(X_testsub.shape[0])
#sub_preds_rf_5 = np.zeros(X_testsub.shape[0])
#sub_preds_rf_6 = np.zeros(X_testsub.shape[0])
#oof_preds_rf_1 = np.zeros(X_train.shape[0])
#oof_preds_rf_2 = np.zeros(X_train.shape[0])
#oof_preds_rf_3 = np.zeros(X_train.shape[0])
#oof_preds_rf_4 = np.zeros(X_train.shape[0])
#oof_preds_rf_5 = np.zeros(X_train.shape[0])
#oof_preds_rf_6 = np.zeros(X_train.shape[0])










