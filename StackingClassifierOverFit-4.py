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

RANDOM_SEED = 0
lgbm_1 = LGBMClassifier(objective = 'binary', max_depth = 3, random_state = 0)
xgb_1 = XGBClassifier(objective = 'binary:logistic', max_depth = 2, random_state = 0, subsample = 0.8)
rf_1 = RandomForestClassifier(n_estimators = 100, max_depth = 4)
svc_1 = SVC(kernel = 'rbf', C = 0.1, gamma = 'auto', probability = True)
gnb_1 = GaussianNB()
knn_1 = KNeighborsClassifier(n_neighbors = 30)
sgd_1 = SGDClassifier(max_iter = 1000, tol = 1e-3, loss = 'log')
lr_1 = RFECV(LogisticRegression(C = 0.1, max_iter = 2000, class_weight = 'balanced', penalty = 'l1', solver = 'liblinear'), 20, 200, cv = 5)
lr_2 = RFECV(LogisticRegression(C = 1.5, max_iter = 2000, class_weight = 'balanced', penalty = 'l2', solver = 'sag'), 20, 200, cv = 5)
lr_3 = RFECV(LogisticRegression(C = 0.5, max_iter = 2000, class_weight = 'balanced', penalty = 'l2', solver = 'liblinear'), 20, 200, cv = 5)
meta_lr = LogisticRegression(C = 2, max_iter = 2000, class_weight = 'balanced', penalty = 'l2', solver = 'liblinear')
np.random.seed(0)
sclf = StackingCVClassifier(classifiers = [lgbm_1, xgb_1, rf_1, svc_1,
                                           lr_1, lr_2, lr_3], 
                            use_probas = True,
                            use_features_in_secondary = False,
                            meta_classifier = meta_lr,
                            cv = 5)
num_folds = 5
folds = StratifiedKFold(n_splits = num_folds, random_state = 0)
test_result_non_transform = np.zeros(len(test))
auc_score_non_transform = 0
cond_non_transform = 0
test_result_noise = np.zeros(len(test))
auc_score_noise = 0
cond_noise = 0
test_result_scale = np.zeros(len(test))
auc_score_scale = 0
cond_scale = 0
test_result_robust = np.zeros(len(test))
auc_score_robust = 0
cond_robust = 0
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, targets)):
    ##base
    X_train, y_train = train.iloc[trn_idx], targets.iloc[trn_idx]
    X_valid, y_valid = train.iloc[val_idx], targets.iloc[val_idx]
    sclf.fit(X_train.values, y_train.values)
    y_pred = sclf.predict_proba(X_valid.values)
    auc = roc_auc_score(y_valid, y_pred[:, 1])
    print('Fold: ', fold_ + 1, 'Non_Transform:', auc)
    auc_score_non_transform += auc
    preds_non_transform = sclf.predict_proba(test.values)
    test_result_non_transform += preds_non_transform[:, 1]
    cond_non_transform += 1
    ##noise
    np.random.seed(0)
    noise = np.random.normal(0.01, 0.01, train.shape)
    train_noise = train + noise
    X_train, y_train = train_noise.iloc[trn_idx], targets.iloc[trn_idx]
    X_valid, y_valid = train_noise.iloc[val_idx], targets.iloc[val_idx]
    sclf.fit(X_train.values, y_train.values)
    y_pred = sclf.predict_proba(X_valid.values)
    auc = roc_auc_score(y_valid, y_pred[:, 1])
    print('Fold: ', fold_ + 1, 'Noise:', auc)
    auc_score_noise += auc
    preds_noise = sclf.predict_proba(test.values)
    test_result_noise += preds_noise[:, 1]
    cond_noise += 1
    ##scaled
    train_scaled = pd.DataFrame(preprocessing.StandardScaler().fit_transform(train))
    X_train, y_train = train_scaled.iloc[trn_idx], targets.iloc[trn_idx]
    X_valid, y_valid = train_scaled.iloc[val_idx], targets.iloc[val_idx]
    sclf.fit(X_train.values, y_train.values)
    y_pred = sclf.predict_proba(X_valid.values)
    auc = roc_auc_score(y_valid, y_pred[:, 1])
    print('Fold: ', fold_ + 1, 'Scaled:', auc)
    auc_score_scale += auc
    preds_scale = sclf.predict_proba(preprocessing.StandardScaler().fit_transform(test))
    test_result_scale += preds_scale[:, 1]
    cond_scale += 1
    ##robust
    train_robust = pd.DataFrame(preprocessing.RobustScaler().fit_transform(train))
    X_train, y_train = train_robust.iloc[trn_idx], targets.iloc[trn_idx]
    X_valid, y_valid = train_robust.iloc[val_idx], targets.iloc[val_idx]
    sclf.fit(X_train.values, y_train.values)
    y_pred = sclf.predict_proba(X_valid.values)
    auc = roc_auc_score(y_valid, y_pred[:, 1])
    print('Fold: ', fold_ + 1, 'Robust:', auc)
    auc_score_robust += auc
    preds_robust = sclf.predict_proba(preprocessing.RobustScaler().fit_transform(test))
    test_result_robust += preds_robust[:, 1]
    cond_robust += 1

auc_score_non_transform = auc_score_non_transform / cond_non_transform
print("AUC score_non_transform: ", auc_score_non_transform)
test_result_non_transform = test_result_non_transform / cond_non_transform

auc_score_noise = auc_score_noise / cond_noise
print("AUC score_noise: ", auc_score_noise)
test_result_noise = test_result_noise / cond_noise

auc_score_scale = auc_score_scale / cond_scale
print("AUC score_scale: ", auc_score_scale)
test_result_scale = test_result_scale / cond_scale

auc_score_robust = auc_score_robust / cond_robust
print("AUC score_robust: ", auc_score_robust)
test_result_robust = test_result_robust / cond_robust

ens_DF = pd.DataFrame({'Base' : test_result_non_transform, 
                       'Noise' : test_result_noise,
                       'Scaled' : test_result_scale,
                       'Robust' : test_result_robust})
ens_DF_mean = pd.DataFrame((test_result_non_transform + 
                            test_result_noise +
                            test_result_scale + 
                            test_result_robust) / 4)
print((auc_score_noise + auc_score_non_transform + auc_score_scale + auc_score_robust) / 4)
#0.8148437500000001
model_1_ens_lr = pd.concat([pd.DataFrame(testSub_Ids), pd.DataFrame(ens_DF_mean)], axis = 1)
model_1_ens_lr.columns = ['id', 'target']
model_1_ens_lr.to_csv(r'D:/Python/data/DontOverFit/StackingClassifier-4-8148.csv', index = False)





    
    



