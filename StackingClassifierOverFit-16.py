import numpy as np
import pandas as pd
from xgboost.sklearn import  XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn import  metrics
from sklearn import preprocessing
from sklearn.utils import class_weight
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression, SGDClassifier, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.classifier import StackingCVClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from lightgbm.sklearn import LGBMClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
import seaborn as sn
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE, RFECV
import scipy.sparse
from sklearn.decomposition import PCA


train = pd.read_csv('D:/Python/data/DontOverFit/train.csv').drop("id", axis = 'columns')
targets = train['target']
train.drop('target', axis = 'columns', inplace = True)

test = pd.read_csv('D:/Python/data/DontOverFit/test.csv').drop("id", axis = 'columns')

train = train[results_Single_AUC_group['Col']]
test = test[results_Single_AUC_group['Col']]

pca = PCA(n_components = 200)
test_PCA = pd.DataFrame(pca.fit_transform(test))

RANDOM_SEED = 0
rf_rfe_1 = RFECV(RandomForestClassifier(n_estimators = 100, max_depth = 4), 10, 100, cv = 5)
rf_rfe_2 = RFECV(RandomForestClassifier(n_estimators = 100, max_depth = 6), 10, 100, cv = 5)
rf_rfe_3 = RFECV(RandomForestClassifier(n_estimators = 100, max_depth = 8), 10, 100, cv = 5)
rf_rfe_4 = RFECV(RandomForestClassifier(n_estimators = 100, max_depth = 10), 10, 100, cv = 5)
rf_rfe_5 = RFECV(RandomForestClassifier(n_estimators = 100, max_depth = 12), 10, 100, cv = 5)
meta_lr = LogisticRegression(C = 2, max_iter = 2000, class_weight = 'balanced', penalty = 'l2', solver = 'liblinear')

np.random.seed(0)

sclf_2_RFE_RF = StackingCVClassifier(classifiers = [rf_rfe_1, rf_rfe_2, rf_rfe_3, rf_rfe_4, rf_rfe_5], 
                            use_probas = True,
                            use_features_in_secondary = False,
                            meta_classifier = meta_lr)

num_folds = 10
folds = StratifiedKFold(n_splits = num_folds, random_state = 0, shuffle = False)

test_result_non_transform_RF = np.zeros(len(test)) ; auc_score_non_transform_RF = 0 ; cond_non_transform_RF = 0
test_result_noise_RF = np.zeros(len(test)) ; auc_score_noise_RF = 0 ; cond_noise_RF = 0
test_result_scale_RF = np.zeros(len(test)) ; auc_score_scale_RF = 0 ; cond_scale_RF = 0
test_result_robust_RF = np.zeros(len(test)) ; auc_score_robust_RF = 0 ; cond_robust_RF = 0
test_result_noise_2_RF = np.zeros(len(test)) ; auc_score_noise_2_RF = 0 ; cond_noise_2_RF = 0
test_result_noise_3_RF = np.zeros(len(test)) ; auc_score_noise_3_RF = 0 ; cond_noise_3_RF = 0
test_result_noise_4_RF = np.zeros(len(test)) ; auc_score_noise_4_RF = 0 ; cond_noise_4_RF = 0

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, targets)):
    ##base
    X_train, y_train = train.iloc[trn_idx], targets.iloc[trn_idx]
    X_valid, y_valid = train.iloc[val_idx], targets.iloc[val_idx]
    sclf_2_RFE_RF.fit(X_train.values, y_train.values)
    y_pred = sclf_2_RFE_RF.predict_proba(X_valid.values)
    auc = roc_auc_score(y_valid, y_pred[:, 1])
    print('Fold: ', fold_ + 1, 'Non_Transform:', auc)
    auc_score_non_transform_RF += auc
    preds_non_transform = sclf_2_RFE_RF.predict_proba(test.values)
    test_result_non_transform_RF += preds_non_transform[:, 1]
    cond_non_transform_RF += 1
    ##noise
    np.random.seed(0)
    noise = np.random.normal(0.01, 0.01, train.shape)
    train_noise = train + noise
    X_train, y_train = train_noise.iloc[trn_idx], targets.iloc[trn_idx]
    X_valid, y_valid = train_noise.iloc[val_idx], targets.iloc[val_idx]
    sclf_2_RFE_RF.fit(X_train.values, y_train.values)
    y_pred = sclf_2_RFE_RF.predict_proba(X_valid.values)
    auc = roc_auc_score(y_valid, y_pred[:, 1])
    print('Fold: ', fold_ + 1, 'Noise:', auc)
    auc_score_noise_RF += auc
    preds_noise = sclf_2_RFE_RF.predict_proba(test.values)
    test_result_noise_RF += preds_noise[:, 1]
    cond_noise_RF += 1
    ##noise_2
    np.random.seed(0)
    noise_2 = np.random.normal(0.02, 0.02, train.shape)
    train_noise_2 = train + noise_2
    X_train, y_train = train_noise_2.iloc[trn_idx], targets.iloc[trn_idx]
    X_valid, y_valid = train_noise_2.iloc[val_idx], targets.iloc[val_idx]
    sclf_2_RFE_RF.fit(X_train.values, y_train.values)
    y_pred = sclf_2_RFE_RF.predict_proba(X_valid.values)
    auc = roc_auc_score(y_valid, y_pred[:, 1])
    print('Fold: ', fold_ + 1, 'Noise_2:', auc)
    auc_score_noise_2_RF += auc
    preds_noise_2 = sclf_2_RFE_RF.predict_proba(test.values)
    test_result_noise_2_RF += preds_noise_2[:, 1]
    cond_noise_2_RF += 1
    ##noise_3
    np.random.seed(0)
    noise_3 = np.random.normal(0.1, 0.1, train.shape)
    train_noise_3 = train + noise_3
    X_train, y_train = train_noise_3.iloc[trn_idx], targets.iloc[trn_idx]
    X_valid, y_valid = train_noise_3.iloc[val_idx], targets.iloc[val_idx]
    sclf_2_RFE_RF.fit(X_train.values, y_train.values)
    y_pred = sclf_2_RFE_RF.predict_proba(X_valid.values)
    auc = roc_auc_score(y_valid, y_pred[:, 1])
    print('Fold: ', fold_ + 1, 'Noise_3:', auc)
    auc_score_noise_3_RF += auc
    preds_noise_3 = sclf_2_RFE_RF.predict_proba(test.values)
    test_result_noise_3_RF += preds_noise_3[:, 1]
    cond_noise_3_RF += 1
    ##noise_4
    np.random.seed(0)
    noise_4 = np.random.normal(0.15, 0.15, train.shape)
    train_noise_4 = train + noise_4
    X_train, y_train = train_noise_4.iloc[trn_idx], targets.iloc[trn_idx]
    X_valid, y_valid = train_noise_4.iloc[val_idx], targets.iloc[val_idx]
    sclf_2_RFE_RF.fit(X_train.values, y_train.values)
    y_pred = sclf_2_RFE_RF.predict_proba(X_valid.values)
    auc = roc_auc_score(y_valid, y_pred[:, 1])
    print('Fold: ', fold_ + 1, 'Noise_4:', auc)
    auc_score_noise_4_RF += auc
    preds_noise_4 = sclf_2_RFE_RF.predict_proba(test.values)
    test_result_noise_4_RF += preds_noise_4[:, 1]
    cond_noise_4_RF += 1
    ##scaled
    train_scaled = pd.DataFrame(preprocessing.StandardScaler().fit_transform(train))
    X_train, y_train = train_scaled.iloc[trn_idx], targets.iloc[trn_idx]
    X_valid, y_valid = train_scaled.iloc[val_idx], targets.iloc[val_idx]
    sclf_2_RFE_RF.fit(X_train.values, y_train.values)
    y_pred = sclf_2_RFE_RF.predict_proba(X_valid.values)
    auc = roc_auc_score(y_valid, y_pred[:, 1])
    print('Fold: ', fold_ + 1, 'Scaled:', auc)
    auc_score_scale_RF += auc
    preds_scale = sclf_2_RFE_RF.predict_proba(preprocessing.StandardScaler().fit_transform(test))
    test_result_scale_RF += preds_scale[:, 1]
    cond_scale_RF += 1
    ##robust
    train_robust = pd.DataFrame(preprocessing.RobustScaler().fit_transform(train))
    X_train, y_train = train_robust.iloc[trn_idx], targets.iloc[trn_idx]
    X_valid, y_valid = train_robust.iloc[val_idx], targets.iloc[val_idx]
    sclf_2_RFE_RF.fit(X_train.values, y_train.values)
    y_pred = sclf_2_RFE_RF.predict_proba(X_valid.values)
    auc = roc_auc_score(y_valid, y_pred[:, 1])
    print('Fold: ', fold_ + 1, 'Robust:', auc)
    auc_score_robust_RF += auc
    preds_robust = sclf_2_RFE_RF.predict_proba(preprocessing.RobustScaler().fit_transform(test))
    test_result_robust_RF += preds_robust[:, 1]
    cond_robust_RF += 1
    
auc_score_non_transform_RF = auc_score_non_transform_RF / cond_non_transform_RF
print("AUC score_non_transform: ", auc_score_non_transform_RF)
test_result_non_transform_RF = test_result_non_transform_RF / cond_non_transform_RF

auc_score_noise_RF = auc_score_noise_RF / cond_noise_RF
print("AUC score_noise: ", auc_score_noise_RF)
test_result_noise_RF = test_result_noise_RF / cond_noise_RF

auc_score_noise_2_RF = auc_score_noise_2_RF / cond_noise_2_RF
print("AUC score_noise_2: ", auc_score_noise_2_RF)
test_result_noise_2_RF = test_result_noise_2_RF / cond_noise_2_RF

auc_score_noise_3_RF = auc_score_noise_3_RF / cond_noise_3_RF
print("AUC score_noise_3: ", auc_score_noise_3_RF)
test_result_noise_3_RF = test_result_noise_3_RF / cond_noise_3_RF

auc_score_noise_4_RF = auc_score_noise_4_RF/ cond_noise_4_RF
print("AUC score_noise_4: ", auc_score_noise_4_RF)
test_result_noise_4_RF = test_result_noise_4_RF / cond_noise_4_RF

auc_score_scale_RF = auc_score_scale_RF / cond_scale_RF
print("AUC score_scale: ", auc_score_scale_RF)
test_result_scale_RF = test_result_scale_RF / cond_scale_RF

auc_score_robust_RF = auc_score_robust_RF / cond_robust_RF
print("AUC score_robust: ", auc_score_robust_RF)
test_result_robust_RF = test_result_robust_RF / cond_robust_RF



print((auc_score_robust_RF + auc_score_scale_RF + auc_score_noise_RF + auc_score_noise_2_RF +
      auc_score_noise_3_RF + auc_score_noise_4_RF + auc_score_non_transform_RF)/7)

ens_DF_RFModels = pd.DataFrame({'Noise' : test_result_noise_RF,
                       'Noise2' : test_result_noise_2_RF,
                       'Noise3' : test_result_noise_3_RF,
                       'Noise4' : test_result_noise_4_RF,
                       'Robust' : test_result_robust_RF,
                       'Scale': test_result_scale_RF,
                       'Non-Trans': test_result_non_transform_RF})
    
ens_DF_mean_RFModels = pd.DataFrame((test_result_noise_RF +
                            test_result_noise_2_RF +
                            test_result_noise_3_RF +
                            test_result_noise_4_RF + 
                            test_result_robust_RF +
                            test_result_scale_RF + 
                            test_result_non_transform_RF) / 7)






ens_Stack_14_15_16 = pd.DataFrame((ens_DF_mean_RFModels + ens_DF_mean_LogModels + ens_DF_mean_AllModels) / 3)
ens_Stack_all_rfeLog = pd.DataFrame((ens_DF_mean_LogModels + ens_DF_mean_AllModels) / 2)


model_1_ens_lr = pd.concat([pd.DataFrame(testSub_Ids), pd.DataFrame(ens_DF_mean_LogModels)], axis = 1)
model_1_ens_lr.columns = ['id', 'target']
model_1_ens_lr.to_csv(r'D:/Python/data/DontOverFit/ens_DF_mean_LogModels.csv', index = False)













