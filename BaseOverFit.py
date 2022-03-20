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


train = pd.read_csv("D:/Python/data/DontOverFit/train.csv")
testSub = pd.read_csv("D:/Python/data/DontOverFit/test.csv")

pd.options.display.max_columns = testSub.shape[1]


train.describe()

train_values = train[train.columns.difference(['target', 'id'])]
train_target = train[['target']]

testSub_values = testSub[testSub.columns.difference(['id'])]
testSub_Ids = testSub[['id']]



train_target.describe()
train_target.target.value_counts()


#X_train, X_test, y_train, y_test = train_test_split(train_values, train_target, test_size = 0.2, random_state = 0)
#
#y_train.target.value_counts()
#y_test.target.value_counts()


#outlier_count = 0
#for var in X_train.columns:
#    var_z = stats.zscore(X_train[var])
#    if((len(var_z[var_z > 3.0]) > 0) or(len(var_z[var_z < -3.0]) > 0)):
#        #print("Feature with Outliers:",var)
#        outlier_count = outlier_count + 1
#print("Total Number of features that has outliers:", outlier_count)
#
#
#X_train = preprocessing.StandardScaler().fit_transform(X_train)
#X_test = preprocessing.StandardScaler().fit_transform(X_test)

##kFoldCV Split
########################################################################################################

train = pd.read_csv("D:/Python/data/DontOverFit/train.csv")
testSub = pd.read_csv("D:/Python/data/DontOverFit/test.csv")

X_train = train[train.columns.difference(['target', 'id'])]
y_train = train[['target']]
X_train = pd.DataFrame(preprocessing.StandardScaler().fit_transform(X_train))

X_train, X_test, y_train, y_test = train_test_split(train_values, train_target, test_size = 0.3, random_state = 0, stratify = train_target)

X_train = pd.DataFrame(X_train).reset_index(drop = True)
X_test = pd.DataFrame(X_test).reset_index(drop = True)
y_train = pd.DataFrame(y_train).reset_index(drop = True)
y_test = pd.DataFrame(y_test).reset_index(drop = True)

X_testsub = testSub[testSub.columns.difference(['id'])]
testSub_Ids = testSub[['id']]
X_testsub = pd.DataFrame(preprocessing.StandardScaler().fit_transform(X_testsub))

########################################################################################################
train = pd.read_csv("D:/Python/data/DontOverFit/train.csv")
testSub = pd.read_csv("D:/Python/data/DontOverFit/test.csv")

X_train = train[train.columns.difference(['target', 'id'])]
y_train = train[['target']]

smote = SMOTE(ratio = 'minority', n_jobs = -1)
X_train, y_train = smote.fit_resample(X_train, y_train.values.ravel())

X_train = pd.DataFrame(X_train).reset_index(drop = True)
y_train = pd.DataFrame(y_train).reset_index(drop = True)

X_train = pd.concat([pd.DataFrame(X_train), 
                     pd.DataFrame(X_train),
                     pd.DataFrame(X_train),
                     pd.DataFrame(X_train),
                     pd.DataFrame(X_train),
                     pd.DataFrame(X_train),
                     pd.DataFrame(X_train),
                     pd.DataFrame(X_train)], 
                     axis = 0)


y_train = pd.concat([pd.DataFrame(y_train), 
                     pd.DataFrame(y_train),
                     pd.DataFrame(y_train),
                     pd.DataFrame(y_train),
                     pd.DataFrame(y_train),
                     pd.DataFrame(y_train),
                     pd.DataFrame(y_train),
                     pd.DataFrame(y_train)], 
                     axis = 0)

X_train = pd.DataFrame(X_train).reset_index(drop = True)
y_train = pd.DataFrame(y_train).reset_index(drop = True)

#np.random.seed(0)
#X_train = X_train + (np.random.normal(0, 1, X_train.shape))
X_train = pd.DataFrame(preprocessing.StandardScaler().fit_transform(X_train))

X_testsub = testSub[testSub.columns.difference(['id'])]
testSub_Ids = testSub[['id']]
X_testsub = pd.DataFrame(preprocessing.StandardScaler().fit_transform(X_testsub))

splits = 20
folds = StratifiedKFold(n_splits = splits, random_state = 0)
oof_preds_log = np.zeros(X_train.shape[0])
sub_preds_log = np.zeros(X_testsub.shape[0])

for fold_, (trn_, val_) in enumerate(folds.split(X_train, y_train)):

    trn_x = X_train.loc[trn_,:]
    val_x = X_train.loc[val_,:]
    
    trn_y = y_train.loc[trn_,:]
    val_y = y_train.loc[val_,:]
    
    clf = LogisticRegression(C = 0.01,
                             max_iter = 2000, 
                             class_weight = 'balanced', 
                             penalty = 'l1', 
                             solver = 'liblinear', 
                             random_state = 0)
    
    model = RFE(clf, 150, step = 1)
    model.fit(trn_x, trn_y.values.ravel())
    oof_preds_log[val_] = model.predict_proba(val_x)[:,1]
    sub_preds_log += model.predict_proba(X_testsub)[:,1] / splits
    fpr, tpr, thresholds = metrics.roc_curve(val_y, oof_preds_log[val_])
    print("Fold:", fold_, "oof-AUC: ", metrics.auc(fpr, tpr))
    
pd.DataFrame(sub_preds_log).head(3)  
    

#    test_preds_log += model.predict_proba(X_test)[:, 1] / splits
#    
#    fpr, tpr, thresholds = metrics.roc_curve(y_test, test_preds_log)
#    print("Fold:", fold_, "test-AUC: ", metrics.auc(fpr, tpr))
#    AUC += metrics.auc(fpr, tpr) / splits
#
#print(AUC)



##[0.80108025] test-AUC split = 0.3
model_1 = pd.concat([pd.DataFrame(testSub_Ids), pd.DataFrame(sub_preds_log)], axis = 1)
model_1.columns = ['id', 'target']
model_1.to_csv(r'D:/Python/data/DontOverFit/OverSampleLRNoise.csv', index = False)

##[0.80289352] test-AUC split = 0.3
model_1 = pd.concat([pd.DataFrame(testSub_Ids), pd.DataFrame(sub_preds_log)], axis = 1)
model_1.columns = ['id', 'target']
model_1.to_csv(r'D:/Python/data/DontOverFit/OverSampleLRNoNoise.csv', index = False)

##[0.85208333] test-AUC split = 0.1
model_1 = pd.concat([pd.DataFrame(testSub_Ids), pd.DataFrame(sub_preds_log)], axis = 1)
model_1.columns = ['id', 'target']
model_1.to_csv(r'D:/Python/data/DontOverFit/OverSampleLRNoNoiseSmoteAfterSplit.csv', index = False)

##split = 0.0 0.851 LB NoNoise
model_1 = pd.concat([pd.DataFrame(testSub_Ids), pd.DataFrame(sub_preds_log)], axis = 1)
model_1.columns = ['id', 'target']
model_1.to_csv(r'D:/Python/data/DontOverFit/OverSampleLRNoNoiseSmote.csv', index = False)

##split = 0.0 0.82xx LB Noise
model_1 = pd.concat([pd.DataFrame(testSub_Ids), pd.DataFrame(sub_preds_log)], axis = 1)
model_1.columns = ['id', 'target']
model_1.to_csv(r'D:/Python/data/DontOverFit/OverSampleLRNoiseSmote.csv', index = False)










########################################################################################################
train = pd.read_csv("D:/Python/data/DontOverFit/train.csv")
testSub = pd.read_csv("D:/Python/data/DontOverFit/test.csv")

X_train = train[train.columns.difference(['target', 'id'])]
y_train = train[['target']]
X_train = pd.DataFrame(preprocessing.StandardScaler().fit_transform(X_train))

#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.3, random_state = 0, stratify = y_train)
#
#X_train = pd.DataFrame(X_train).reset_index(drop = True)
#X_test = pd.DataFrame(X_test).reset_index(drop = True)
#y_train = pd.DataFrame(y_train).reset_index(drop = True)
#y_test = pd.DataFrame(y_test).reset_index(drop = True)

X_testsub = testSub[testSub.columns.difference(['id'])]
testSub_Ids = testSub[['id']]
X_testsub = pd.DataFrame(preprocessing.StandardScaler().fit_transform(X_testsub))

########################################################################################3
splits = 10
folds = StratifiedKFold(n_splits = splits, random_state = 0)

oof_preds_xgb = np.zeros(X_train.shape[0])
#test_preds_xgb = np.zeros(X_test.shape[0])
sub_preds_xgb = np.zeros(X_testsub.shape[0])
AUC = np.zeros(1)

#d_Matrix_test = xgb.DMatrix(X_test)
d_Matrix_testsub = xgb.DMatrix(X_testsub)

for fold_, (trn_, val_) in enumerate(folds.split(X_train, y_train)):

    trn_x = X_train.loc[trn_,:]
    val_x = X_train.loc[val_,:]
    
    trn_y = y_train.loc[trn_,:]
    val_y = y_train.loc[val_,:]
    
    d_Matrix_train = xgb.DMatrix(trn_x, label = trn_y.values)
    d_Matrix_val = xgb.DMatrix(val_x, label = val_y.values)
    
    xgb_param = {
        'learning_rate' : 0.5,
        'max_depth' : 2,
        'alpha' : 0,
        'lambda' : 2,
        'gamma' : 0,
        'min_child_weight' : 12,
        'objective': 'binary:logistic',
        'eval_metric' : 'auc'} 
    xgb_model = xgb.train(params = xgb_param, 
                        dtrain = d_Matrix_train, 
                        num_boost_round = 100)
    
    oof_preds_xgb[val_] = xgb_model.predict(d_Matrix_val)
    #test_preds_xgb += xgb_model.predict(d_Matrix_test) / splits
    #sub_preds_xgb += xgb_model.predict(d_Matrix_testsub) / splits
    
    fpr, tpr, thresholds = metrics.roc_curve(val_y, oof_preds_xgb[val_])
    print("Fold:", fold_, "oof-AUC: ", metrics.auc(fpr, tpr))
    AUC += metrics.auc(fpr, tpr) / splits

print(AUC)


model_4 = pd.concat([pd.DataFrame(testSub_Ids), pd.DataFrame(sub_preds_xgb)], axis = 1)
model_4.columns = ['id', 'target']
model_4.to_csv(r'D:/Python/data/DontOverFit/xgbKFoldTunedAllData-2-200RoundsNotScaled.csv', index = False)



########################################################################################################
train = pd.read_csv("D:/Python/data/DontOverFit/train.csv")
testSub = pd.read_csv("D:/Python/data/DontOverFit/test.csv")


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

X_train = pd.DataFrame(X_train).reset_index(drop = True)
y_train = pd.DataFrame(y_train).reset_index(drop = True)


#np.random.seed(0)
#X_train = X_train + (np.random.normal(0, 1, X_train.shape))
X_train = pd.DataFrame(preprocessing.StandardScaler().fit_transform(X_train))

X_testsub = testSub[testSub.columns.difference(['id'])]
testSub_Ids = testSub[['id']]
X_testsub = pd.DataFrame(preprocessing.StandardScaler().fit_transform(X_testsub))

splits = 5
folds = StratifiedKFold(n_splits = splits, random_state = 0, shuffle = True)

oof_preds_rf = np.zeros(X_train.shape[0])
sub_preds_rf = np.zeros(X_test.shape[0])

for fold_, (trn_, val_) in enumerate(folds.split(X_train, y_train)):

    trn_x = X_train.loc[trn_,:]
    val_x = X_train.loc[val_,:]
    
    trn_y = y_train.loc[trn_,:]
    val_y = y_train.loc[val_,:]
    
    clf_2_rf = RandomForestClassifier(random_state = 0, 
                                      n_estimators = 100,
                                      bootstrap = False,
                                      class_weight = 'balanced',
                                      criterion = 'gini',
                                      max_depth = 4,
                                      max_features = 'auto',
                                      min_samples_leaf = 2,
                                      min_samples_split = 6,
                                      n_jobs = -1)
#    model = RFE(clf_2_rf, 100, step = 1)
#    model.fit(trn_x, trn_y.values.ravel())
    clf_2_rf.fit(trn_x, trn_y.values.ravel())
    
    oof_preds_rf[val_] = clf_2_rf.predict_proba(val_x)[:,1]
    sub_preds_rf += clf_2_rf.predict_proba(X_test)[:,1] / splits
    
    fpr, tpr, thresholds = metrics.roc_curve(val_y, oof_preds_rf[val_])
    print("Fold:", fold_, "RF-AUC: ", metrics.auc(fpr, tpr))


model_4 = pd.concat([pd.DataFrame(testSub_Ids), pd.DataFrame(sub_preds_rf)], axis = 1)
model_4.columns = ['id', 'target']
model_4.to_csv(r'D:/Python/data/DontOverFit/TunedRFRFE.csv', index = False)









model_2 = pd.concat([pd.DataFrame(testSub_Ids), pd.DataFrame((sub_preds_log + sub_preds_rf) / 2)], axis = 1)
model_2.columns = ['id', 'target']
model_2.to_csv(r'D:/Python/data/DontOverFit/LOGRFE-TunedRFRFE-20FOLDAVG.csv', index = False)


model_3 = pd.concat([pd.DataFrame(testSub_Ids), pd.DataFrame((sub_preds_log + sub_preds_rf + sub_preds_xgb) / 3)], axis = 1)
model_3.columns = ['id', 'target']
model_3.to_csv(r'D:/Python/data/DontOverFit/LOGRFE-TunedRFRFE-XGB-20FOLDAVG.csv', index = False)
#########################################################################################################
train = pd.read_csv("D:/Python/data/DontOverFit/train.csv")
testSub = pd.read_csv("D:/Python/data/DontOverFit/test.csv")

X_train = train[train.columns.difference(['target', 'id'])]
y_train = train[['target']]
X_train_col = pd.DataFrame({'33': X_train['33']})

X_testsub = testSub[testSub.columns.difference(['id'])]
testSub_Ids = testSub[['id']]
X_testsub = pd.DataFrame({'33': X_testsub['33']})


sub_preds_log = np.zeros(X_testsub.shape[0])
splits = 10
folds = StratifiedKFold(n_splits = splits, random_state = 0)
AUC = np.zeros(1)
for fold_, (trn_, val_) in enumerate(folds.split(X_train_col, y_train)):
    trn_x = X_train_col.loc[trn_,:]
    val_x = X_train_col.loc[val_,:]
    trn_y = y_train.loc[trn_,:]
    val_y = y_train.loc[val_,:]   
    clf = LogisticRegression(max_iter = 2000, 
                             solver = 'liblinear', 
                             random_state = 0)
    clf.fit(trn_x, trn_y.values.ravel())
    probs = clf.predict_proba(val_x)[:,1]
    fpr, tpr, thresholds = metrics.roc_curve(val_y, probs)
    print("AUC: ", metrics.auc(fpr, tpr))
    AUC += metrics.auc(fpr, tpr) / splits
    sub_preds_log += clf.predict_proba(X_testsub)[:,1] / splits
print(AUC)

model_1_ens_lr = pd.concat([pd.DataFrame(testSub_Ids), pd.DataFrame(sub_preds_log)], axis = 1)
model_1_ens_lr.columns = ['id', 'target']
model_1_ens_lr.to_csv(r'D:/Python/data/DontOverFit/33-33BinsLR-CV{.7447}.csv', index = False)









