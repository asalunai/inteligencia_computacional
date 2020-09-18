# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context='paper', 
        style='ticks', 
        font='serif', 
        font_scale=1.5, 
        color_codes=True, 
        rc={'figure.figsize':(11.7,8.27)})

# Create DataFrame:
df = pd.read_csv('banana.dat', skiprows=7, names=['x1','x2','y'])

# DataFrame Scatter Plot:
sns.scatterplot(data=df, x="x1", y="x2", hue="y", legend=False, palette='Spectral')

# Create Mesh Grid
def make_meshgrid(x, y, h=.02):
  x_min, x_max = x.min() - 1, x.max() + 1
  y_min, y_max = y.min() - 1, y.max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
  return xx, yy

# Create SVM contours
def plot_contours(ax, clf, xx, yy, **params):
  Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  out = ax.contourf(xx, yy, Z, **params)
  return out

# Plot SVM trainig
def plot_graf(X_train,y_train):

  models = (svm.SVC(kernel='linear', C=1.),
            svm.SVC(kernel='poly', C=1.),
            svm.SVC(kernel='rbf', C=1.),
            svm.SVC(kernel='sigmoid', coef0=1.00, C=1.),
            svm.SVC(kernel='sigmoid', coef0=0.50, C=1.),
            svm.SVC(kernel='sigmoid', coef0=0.01, C=1.))
  models = (clf.fit(X_train, y_train) for clf in models)

  # title for the plots
  titles = ('SVC Linear',
            'SVC Polinomial',
            'SVC RBF',
            'SVC Sigmoide (=1.0)',
            'SVC Sigmoide (=0.5)',
            'SVC Sigmoide (=0.01)')

  # Set-up 2x2 grid for plotting.
  fig, sub = plt.subplots(2, 3)
  plt.subplots_adjust(wspace=0.4, hspace=0.4)

  y = y_train
  X0, X1 = X_train.x1, X_train.x2
  xx, yy = make_meshgrid(X0, X1)

  for clf, title, ax in zip(models, titles, sub.flatten()):
      plot_contours(ax, clf, xx, yy, cmap=plt.cm.rainbow, alpha=0.3)
      ax.scatter(X0, X1, c=y, cmap=plt.cm.rainbow, s=20, edgecolors=None, alpha=0.3)
      plot_contours(ax, clf, xx, yy, alpha=0.1, colors='black')
      ax.set_xlim(xx.min(), xx.max())
      ax.set_ylim(yy.min(), yy.max())
      ax.set_title(title)

  plt.show()

grafico = plot_graf(df.iloc[:,:2],df.y)

# Set models metrics: accuracy
def modelos_metrica(X_test,y_test,models):
  met_acuracia = []
  for clf in models:
    y_pred = clf.predict(X_test)
    metrica = accuracy_score(y_test, y_pred)
    met_acuracia.append(metrica)
  return met_acuracia

# SVM models with K-fold cross-validation
hp_kfold =  [2, 5, 10]

for folds in hp_kfold:
  kfold = KFold(n_splits=folds, shuffle=True, random_state=1)
  kmean_ac = []
  for ksets in kfold.split(df):
    train = df.iloc[ksets[0]]
    test = df.iloc[ksets[1]]
    X_train, y_train = [train.iloc[:,:2], train.y]
    X_test, y_test = [test.iloc[:,:2], test.y]
    #print('treino:', train.shape)
    #print('teste:',test.shape)

    models = (svm.SVC(kernel='linear', C=1.),
              svm.SVC(kernel='poly', C=1.),
              svm.SVC(kernel='rbf', C=1.),
              svm.SVC(kernel='sigmoid', coef0=1.00, C=1.),
              svm.SVC(kernel='sigmoid', coef0=0.50, C=1.),
              svm.SVC(kernel='sigmoid', coef0=0.01, C=1.))
    models = (clf.fit(X_train, y_train) for clf in models)

    #plot_graf(X_train,y_train,models)
    met_acuracia = modelos_metrica(X_test,y_test,models)

    kmean_ac.append(met_acuracia)
  print('Sets:', folds)
  a_kmean_ac = np.array(kmean_ac)
  ac_mean = np.mean(a_kmean_ac, axis=0)
  ac_sqrt = np.std(a_kmean_ac, axis=0)
  print(ac_mean)
  print(ac_sqrt)

  
# XGBoost implementation

import pickle
import xgboost as xgb

rng = np.random.RandomState(2)

y = df.y
X = df.iloc[:,:2]
kf = KFold(n_splits=10, shuffle=True, random_state=rng)
for train_index, test_index in kf.split(X):
    xgb_model = xgb.XGBClassifier().fit(X.iloc[train_index], y.iloc[train_index])
    predictions = xgb_model.predict(X.iloc[test_index])
    actuals = y.iloc[test_index]
    print('Acuracia:', accuracy_score(actuals, predictions))