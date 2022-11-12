import os

import h5py
import numpy as np
import pytest

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from survos2.server.superseg import obtain_classifier, train, predict


def test_obtain_classifier():
    classifier_parameters = {
        "n_estimators": 10,
        "max_depth": 5,
        "n_jobs": 1,
        "clf": "Ensemble",
        "type": "rf",
    }

    clf = obtain_classifier(classifier_parameters)
    # assert type(clf) == 'tuple'
    assert isinstance(clf[0], RandomForestClassifier)


def test_train():
    predict_params = classifier_parameters = {
        "n_estimators": 10,
        "max_depth": 5,
        "n_jobs": 1,
        "clf": "Ensemble",
        "type": "rf",
    }

    X = np.random.random((10, 4))
    y = np.ones((10, 1))
    clf, proj = train(X, y, predict_params)
    assert proj == None
    assert isinstance(clf[0], DecisionTreeClassifier)


def test_predict():
    predict_params = classifier_parameters = {
        "n_estimators": 10,
        "max_depth": 5,
        "n_jobs": 1,
        "clf": "Ensemble",
        "type": "rf",
    }

    X = np.random.random((10, 4))
    y = np.ones((10, 1))
    clf, proj = train(X, y, predict_params)
    result = predict(X, clf)
    assert isinstance(result["class"], np.ndarray)
