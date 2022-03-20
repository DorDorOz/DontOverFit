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
pca = PCA(n_components = 250)
test_PCA = pd.DataFrame(pca.fit_transform(test))

RANDOM_SEED = 0
lgbm_1 = LGBMClassifier(objective = 'binary', max_depth = 3, random_state = 0)
lgbm_2 = LGBMClassifier(objective = 'binary', max_depth = 6, random_state = 0)
lgbm_3 = LGBMClassifier(objective = 'binary', max_depth = 9, random_state = 0)
xgb_1 = XGBClassifier(objective = 'binary:logistic', max_depth = 2, random_state = 0, subsample = 0.8)
xgb_2 = XGBClassifier(objective = 'binary:logistic', max_depth = 3, random_state = 0, subsample = 0.9)
xgb_3 = XGBClassifier(objective = 'binary:logistic', max_depth = 4, random_state = 0, subsample = 1)
rf_1 = RandomForestClassifier(n_estimators = 100, max_depth = 2)
rf_2 = RandomForestClassifier(n_estimators = 100, max_depth = 4)
rf_3 = RandomForestClassifier(n_estimators = 100, max_depth = 6)
rf_4 = RFECV(RandomForestClassifier(n_estimators = 100, max_depth = 2), 25, 250, cv = 5)
rf_5 = RFECV(RandomForestClassifier(n_estimators = 100, max_depth = 4), 25, 250, cv = 5)
rf_6 = RFECV(RandomForestClassifier(n_estimators = 100, max_depth = 6), 25, 250, cv = 5)
svc_1 = SVC(kernel = 'rbf', C = 0.1, gamma = 'auto', probability = True)
svc_2 = SVC(kernel = 'rbf', C = 1, gamma = 'auto', probability = True)
svc_3 = SVC(kernel = 'rbf', C = 3, gamma = 'auto', probability = True)
gnb_1 = GaussianNB()
knn_1 = KNeighborsClassifier(n_neighbors = 29)
knn_2 = KNeighborsClassifier(n_neighbors = 39)
knn_3 = KNeighborsClassifier(n_neighbors = 49)
sgd_1 = SGDClassifier(max_iter = 1000, tol = 1e-3, loss = 'log')
lr_1 = RFECV(LogisticRegression(C = 0.1, max_iter = 2000, class_weight = 'balanced', penalty = 'l1', solver = 'liblinear'), 25, 250, cv = 5)
lr_2 = RFECV(LogisticRegression(C = 1.5, max_iter = 2000, class_weight = 'balanced', penalty = 'l2', solver = 'sag'), 25, 250, cv = 5)
lr_3 = RFECV(LogisticRegression(C = 0.5, max_iter = 2000, class_weight = 'balanced', penalty = 'l2', solver = 'liblinear'), 25, 250, cv = 5)
lr_4 = RFECV(LogisticRegression(C = 1, max_iter = 2000, class_weight = 'balanced', penalty = 'l2', solver = 'liblinear'), 25, 250, cv = 5)
lr_5 = LogisticRegression(C = 0.1, max_iter = 2000, class_weight = 'balanced', penalty = 'l1', solver = 'liblinear')
lr_6 = LogisticRegression(C = 1.5, max_iter = 2000, class_weight = 'balanced', penalty = 'l2', solver = 'sag')
lr_7 = LogisticRegression(C = 0.5, max_iter = 2000, class_weight = 'balanced', penalty = 'l2', solver = 'liblinear')
lr_8 = LogisticRegression(C = 1, max_iter = 2000, class_weight = 'balanced', penalty = 'l2', solver = 'liblinear')
meta_lr = LogisticRegression(C = 2, max_iter = 2000, class_weight = 'balanced', penalty = 'l2', solver = 'liblinear')
np.random.seed(0)
sclf = StackingCVClassifier(classifiers = [lgbm_1, lgbm_2, lgbm_3, 
                                           xgb_1, xgb_2, xgb_3,
                                           rf_1, rf_2, rf_3, rf_4, rf_5, rf_6,
                                           svc_1, svc_2, svc_3, 
                                           lr_1, lr_2, lr_3, lr_4, lr_5], 
                            use_probas = True,
                            use_features_in_secondary = False,
                            meta_classifier = meta_lr,
                            cv = 5)

num_folds = 5
folds = StratifiedKFold(n_splits = num_folds, random_state = 0, shuffle = True)
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
test_result_mm = np.zeros(len(test))
auc_score_mm = 0
cond_mm = 0
test_result_pca = np.zeros(len(test))
auc_score_pca = 0
cond_pca = 0
test_result_noise_2 = np.zeros(len(test))
auc_score_noise_2 = 0
cond_noise_2 = 0
test_result_noise_3 = np.zeros(len(test))
auc_score_noise_3 = 0
cond_noise_3 = 0
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
    ##noise_2
    np.random.seed(0)
    noise_2 = np.random.normal(0.02, 0.02, train.shape)
    train_noise_2 = train + noise_2
    X_train, y_train = train_noise_2.iloc[trn_idx], targets.iloc[trn_idx]
    X_valid, y_valid = train_noise_2.iloc[val_idx], targets.iloc[val_idx]
    sclf.fit(X_train.values, y_train.values)
    y_pred = sclf.predict_proba(X_valid.values)
    auc = roc_auc_score(y_valid, y_pred[:, 1])
    print('Fold: ', fold_ + 1, 'Noise_2:', auc)
    auc_score_noise_2 += auc
    preds_noise_2 = sclf.predict_proba(test.values)
    test_result_noise_2 += preds_noise_2[:, 1]
    cond_noise_2 += 1
    ##noise_3
    np.random.seed(0)
    noise_3 = np.random.normal(0.1, 0.1, train.shape)
    train_noise_3 = train + noise_3
    X_train, y_train = train_noise_3.iloc[trn_idx], targets.iloc[trn_idx]
    X_valid, y_valid = train_noise_3.iloc[val_idx], targets.iloc[val_idx]
    sclf.fit(X_train.values, y_train.values)
    y_pred = sclf.predict_proba(X_valid.values)
    auc = roc_auc_score(y_valid, y_pred[:, 1])
    print('Fold: ', fold_ + 1, 'Noise_3:', auc)
    auc_score_noise_3 += auc
    preds_noise_3 = sclf.predict_proba(test.values)
    test_result_noise_3 += preds_noise_3[:, 1]
    cond_noise_3 += 1
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
    ##mm
    train_mm = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(train))
    X_train, y_train = train_mm.iloc[trn_idx], targets.iloc[trn_idx]
    X_valid, y_valid = train_mm.iloc[val_idx], targets.iloc[val_idx]
    sclf.fit(X_train.values, y_train.values)
    y_pred = sclf.predict_proba(X_valid.values)
    auc = roc_auc_score(y_valid, y_pred[:, 1])
    print('Fold: ', fold_ + 1, 'MinMax:', auc)
    auc_score_mm += auc
    preds_mm = sclf.predict_proba(preprocessing.MinMaxScaler().fit_transform(test))
    test_result_mm += preds_mm[:, 1]
    cond_mm += 1
    #pca
    train_PCA = pd.DataFrame(pca.fit_transform(train))
    X_train, y_train = train_PCA.iloc[trn_idx], targets.iloc[trn_idx]
    X_valid, y_valid = train_PCA.iloc[val_idx], targets.iloc[val_idx]
    sclf.fit(X_train.values, y_train.values)
    y_pred = sclf.predict_proba(X_valid.values)
    auc = roc_auc_score(y_valid, y_pred[:, 1])
    print('Fold: ', fold_ + 1, 'PCA:', auc)
    auc_score_pca += auc
    preds_pca = sclf.predict_proba(test_PCA.values)
    test_result_pca += preds_pca[:, 1]
    cond_pca += 1
    
auc_score_non_transform = auc_score_non_transform / cond_non_transform
print("AUC score_non_transform: ", auc_score_non_transform)
test_result_non_transform = test_result_non_transform / cond_non_transform

auc_score_noise = auc_score_noise / cond_noise
print("AUC score_noise: ", auc_score_noise)
test_result_noise = test_result_noise / cond_noise

auc_score_noise_2 = auc_score_noise_2 / cond_noise_2
print("AUC score_noise_2: ", auc_score_noise_2)
test_result_noise_2 = test_result_noise_2 / cond_noise_2

auc_score_noise_3 = auc_score_noise_3 / cond_noise_3
print("AUC score_noise_3: ", auc_score_noise_3)
test_result_noise_3 = test_result_noise_3 / cond_noise_3

auc_score_scale = auc_score_scale / cond_scale
print("AUC score_scale: ", auc_score_scale)
test_result_scale = test_result_scale / cond_scale

auc_score_robust = auc_score_robust / cond_robust
print("AUC score_robust: ", auc_score_robust)
test_result_robust = test_result_robust / cond_robust

auc_score_mm = auc_score_mm / cond_mm
print("AUC score_mm: ", auc_score_mm)
test_result_mm = test_result_mm / cond_mm

auc_score_pca = auc_score_pca / cond_pca
print("AUC score_pca: ", auc_score_pca)
test_result_pca = test_result_pca / cond_pca

ens_DF = pd.DataFrame({'Base' : test_result_non_transform, 
                       'Noise' : test_result_noise,
                       'Noise2' : test_result_noise_2,
                       'Noise3' : test_result_noise_3,
                       'Scaled' : test_result_scale,
                       'Robust' : test_result_robust,
                       'MinMax': test_result_mm,
                       'PCA': test_result_pca})
ens_DF_mean = pd.DataFrame((test_result_non_transform + 
                            test_result_noise +
                            test_result_noise_2 +
                            test_result_noise_3 +
                            test_result_scale + 
                            test_result_robust) / 6)
print((auc_score_noise + auc_score_non_transform + auc_score_scale + auc_score_robust + auc_score_mm + auc_score_pca) / 6)



model_1_ens_lr = pd.concat([pd.DataFrame(testSub_Ids), pd.DataFrame(ens_DF_mean)], axis = 1)
model_1_ens_lr.columns = ['id', 'target']
model_1_ens_lr.to_csv(r'D:/Python/data/DontOverFit/StackingClassifier-9all-mmpca.csv', index = False)





    
    



