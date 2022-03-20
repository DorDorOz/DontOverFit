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
from keras import utils as np_utils
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras import layers, models, optimizers
import keras.backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import adam


train = pd.read_csv("D:/Python/data/DontOverFit/train.csv")

X_train = train[train.columns.difference(['target', 'id'])]
y_train = train[['target']]

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.3, random_state = 0, stratify = y_train)

X_train = preprocessing.MinMaxScaler().fit_transform(X_train)
X_test = preprocessing.MinMaxScaler().fit_transform(X_test)
y_train = np_utils.to_categorical(y_train.values.ravel())
y_test = np_utils.to_categorical(y_test.values.ravel())


dnn_model_1 = tf.keras.models.Sequential()

dnn_model_1.add(tf.keras.layers.Dense(units = 50, input_dim = 300, activation = tf.nn.relu))
dnn_model_1.add(tf.keras.layers.Dropout(0.4))
dnn_model_1.add(tf.keras.layers.Dense(units = 50, activation = tf.nn.relu))
dnn_model_1.add(tf.keras.layers.Dropout(0.4))
dnn_model_1.add(tf.keras.layers.Dense(units = 50, activation = tf.nn.relu))
dnn_model_1.add(tf.keras.layers.Dropout(0.4))
dnn_model_1.add(tf.keras.layers.Dense(units = 2, activation = tf.nn.softmax))

dnn_model_1.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.0001),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])
#dnn_model_1.summary()
dnn_model_1.fit(X_train, y_train, epochs = 500)

dnn_model_1_prob = dnn_model_1.predict(X_test)

###################################################################################################
train = pd.read_csv("D:/Python/data/DontOverFit/train.csv")

X_train = train[train.columns.difference(['target', 'id'])]
y_train = train[['target']]

splits = 10
folds = StratifiedKFold(n_splits = splits, random_state = 0)
oof_preds_log = np.zeros(X_train.shape[0])

for fold_, (trn_, val_) in enumerate(folds.split(X_train, y_train)):

    trn_x = preprocessing.MinMaxScaler().fit_transform(X_train.loc[trn_,:])
    val_x = preprocessing.MinMaxScaler().fit_transform(X_train.loc[val_,:])
    
    trn_y = np_utils.to_categorical(y_train.loc[trn_,:])
    val_y = np_utils.to_categorical(y_train.loc[val_,:])
    
    dnn_model_1 = tf.keras.models.Sequential()

    dnn_model_1.add(tf.keras.layers.Dense(units = 150, input_dim = 300, activation = tf.nn.relu))
    dnn_model_1.add(tf.keras.layers.Dropout(0.8))
    dnn_model_1.add(tf.keras.layers.Dense(units = 75, activation = tf.nn.relu))
    dnn_model_1.add(tf.keras.layers.Dropout(0.1))
    dnn_model_1.add(tf.keras.layers.Dense(units = 2, activation = tf.nn.softmax))

    dnn_model_1.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.0001),
                                                             loss = 'sparse_categorical_crossentropy',
                                                             metrics = ['accuracy'])
    #dnn_model_1.summary()
    dnn_model_1.fit(X_train, y_train, epochs = 150, verbose = 0)

    oof_preds_log[val_] = dnn_model_1.predict(val_x)[:,1]
    fpr, tpr, thresholds = metrics.roc_curve(y_train.loc[val_,:], oof_preds_log[val_])
    print("Fold:", fold_, "oof-AUC: ", metrics.auc(fpr, tpr))















































