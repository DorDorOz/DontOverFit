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
from sklearn.linear_model import LogisticRegression, SGDClassifier
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

RANDOM_SEED = 0

lgbm_1 = LGBMClassifier(objective = 'binary', max_depth = 2, random_state = 0)
xgb_1 = XGBClassifier(objective = 'binary:logistic', max_depth = 2, random_state = 0)
rf_1 = RandomForestClassifier(n_estimators = 100, max_depth = 2)
svc_1 = SVC(kernel = 'rbf', C = 0.1, gamma = 'auto', probability = True)
gnb_1 = GaussianNB()
sgd_1 = SGDClassifier(max_iter = 1000, tol = 1e-3, loss = 'log')
knn_1 = KNeighborsClassifier(n_neighbors = 29)
lr_1 = LogisticRegression(C = 0.5, max_iter = 2000, class_weight = 'balanced', penalty = 'l2', solver = 'liblinear')

np.random.seed(0)
sclf = StackingClassifier(classifiers = [lgbm_1, xgb_1, rf_1, knn_1, 
                                           svc_1, lr_1, gnb_1, sgd_1], 
                            use_probas = True,
                            use_features_in_secondary = False,
                            meta_classifier = lr_1)
num_folds = 10
folds = StratifiedKFold(n_splits = num_folds, random_state = 0, shuffle = False)
test_result_noise = np.zeros(len(test))
auc_score_noise = 0
cond_noise = 0

test_result_noise_2 = np.zeros(len(test))
auc_score_noise_2 = 0
cond_noise_2 = 0

test_result_noise_3 = np.zeros(len(test))
auc_score_noise_3 = 0
cond_noise_3 = 0

test_result_noise_4 = np.zeros(len(test))
auc_score_noise_4 = 0
cond_noise_4 = 0

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, targets)):
    ##noise
    np.random.seed(0)
    noise = np.random.normal(0.01, 0.01, train.shape)
    train_noise = train + noise
    X_train, y_train = train_noise.iloc[trn_idx], targets.iloc[trn_idx]
    X_valid, y_valid = train.iloc[val_idx], targets.iloc[val_idx]
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
    X_train_2, y_train = train_noise_2.iloc[trn_idx], targets.iloc[trn_idx]
    sclf.fit(X_train_2.values, y_train.values)
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
    X_train_3, y_train = train_noise_3.iloc[trn_idx], targets.iloc[trn_idx]
    sclf.fit(X_train_3.values, y_train.values)
    y_pred = sclf.predict_proba(X_valid.values)
    auc = roc_auc_score(y_valid, y_pred[:, 1])
    print('Fold: ', fold_ + 1, 'Noise_3:', auc)
    auc_score_noise_3 += auc
    preds_noise_3 = sclf.predict_proba(test.values)
    test_result_noise_3 += preds_noise_3[:, 1]
    cond_noise_3 += 1
    ##noise_4
    np.random.seed(0)
    noise_4 = np.random.normal(0.15, 0.15, train.shape)
    train_noise_4 = train + noise_4
    X_train_4, y_train = train_noise_4.iloc[trn_idx], targets.iloc[trn_idx]
    sclf.fit(X_train_4.values, y_train.values)
    y_pred = sclf.predict_proba(X_valid.values)
    auc = roc_auc_score(y_valid, y_pred[:, 1])
    print('Fold: ', fold_ + 1, 'Noise_4:', auc)
    auc_score_noise_4 += auc
    preds_noise_4 = sclf.predict_proba(test.values)
    test_result_noise_4 += preds_noise_4[:, 1]
    cond_noise_4 += 1

auc_score_noise = auc_score_noise / cond_noise
print("AUC score_noise: ", auc_score_noise)
test_result_noise = test_result_noise / cond_noise

auc_score_noise_2 = auc_score_noise_2 / cond_noise_2
print("AUC score_noise_2: ", auc_score_noise_2)
test_result_noise_2 = test_result_noise_2 / cond_noise_2

auc_score_noise_3 = auc_score_noise_3 / cond_noise_3
print("AUC score_noise_3: ", auc_score_noise_3)
test_result_noise_3 = test_result_noise_3 / cond_noise_3

auc_score_noise_4 = auc_score_noise_4 / cond_noise_4
print("AUC score_noise_4: ", auc_score_noise_4)
test_result_noise_4 = test_result_noise_4 / cond_noise_4

ens_DF = pd.DataFrame({'Noise' : test_result_noise,
                       'Noise2' : test_result_noise_2,
                       'Noise3' : test_result_noise_3,
                       'Noise4' : test_result_noise_4})
ens_DF_mean = pd.DataFrame((test_result_noise +
                            test_result_noise_2 +
                            test_result_noise_3 +
                            test_result_noise_4) / 4)

print((auc_score_noise + auc_score_noise_2 + 
       auc_score_noise_3 + auc_score_noise_4) / 4)

0.8208
model_1_ens_lr = pd.concat([pd.DataFrame(testSub_Ids), pd.DataFrame(ens_DF_mean)], axis = 1)
model_1_ens_lr.columns = ['id', 'target']
model_1_ens_lr.to_csv(r'D:/Python/data/DontOverFit/StackingClassifier-12-8375.csv', index = False)





    
    



