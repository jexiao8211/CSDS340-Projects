import sys
from contextlib import contextmanager

@contextmanager
def redirect_stdout_to_file(file_path):
    """
    write the output to a log file

    :param file_path: the file to write to
    """
    original_stdout = sys.stdout
    with open(file_path, 'w') as f:
        sys.stdout = f
        yield
        sys.stdout = original_stdout

import warnings
warnings.filterwarnings("ignore")

################################################

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering

import functools
from sklearn.metrics import make_scorer
from sklearn.metrics.cluster import adjusted_rand_score

from st_dbscan import ST_DBSCAN

"""
constant values
"""
# set random state seed
RANDOM_STATE = 123

# param grids for each model
PARAM_GRIDS = {
    'kmeans': {
        'n_clusters': [2, 4, 14, 16, 20, 32],
        'init': ['k-means++'],
        'random_state': [123],
        'n_init': ['auto']
    },
    'agg': {
        'n_clusters': [2, 4, 8, 16, 32],
        'linkage': ['ward', 'complete', 'average', 'single']
    },
    'dbscan': {
        'eps': [0.1, 0.3, 0.5, 1],
        'min_samples': [1, 3, 5],
        'algorithm': ['kd_tree', 'ball_tree', 'brute']
    },
    'spec': {
        'n_clusters': [2, 4, 8, 16, 32],
        'assign_labels': ['kmeans', 'discretize', 'cluster_qr'],
        'random_state': [123]
    },
}

# initialize all models
MODELS = {
    'kmeans': KMeans(),
    'agg': AgglomerativeClustering(),
    'dbscan': DBSCAN(),
    'spec': SpectralClustering()
}


def inspect_data(csv_path):
    """
    inspect the data from a file

    :param csv_path: the path to csv file

    :return: the dataframe of the csv file
    """
    df = pd.read_csv(csv_path, converters={'SEQUENCE_DTTM' : hh_mm_ss2seconds})
    return df


def hh_mm_ss2seconds(hh_mm_ss):
    """
    pre-defined function for converting time to readable format

    :param hh_mm_ss: the time to be converted

    :return: time in format of hour-minute-second
    """
    return functools.reduce(lambda acc, x: acc*60 + x, map(int, hh_mm_ss.split(':')))


def custom_scorer(clf, X, y):
    """
    create a custom scorer using adjusted rand score

    :param clf: the model to be used
    :param X: the features to be fitted
    :param y: the labels to be compared with

    :return: adjusted rand score of predicted labels and ground truth
    """
    y_pred = clf.fit(X).labels_
    y_true = y.flatten()
    return adjusted_rand_score(y_true, y_pred)


def predictor_baseline(csv_path):
    """
    get the prediction labels of the baseline estimator

    :param csv_path: the path to csv file

    :return: predicted labels of the model
    """
    # load data and convert hh:mm:ss to seconds
    df = pd.read_csv(csv_path, converters={'SEQUENCE_DTTM' : hh_mm_ss2seconds})
    # print(df.corr())
    # select features 
    selected_features = ['SEQUENCE_DTTM', 'LAT', 'LON', 'SPEED_OVER_GROUND' ,'COURSE_OVER_GROUND']

    # split features and labels + standardize the data
    X = df[selected_features].to_numpy()
    # Standardization
    X = StandardScaler().fit(X).transform(X)

    # k-means with K = number of unique VIDs of set1
    K = 20
    model = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init='auto').fit(X)
    # predict cluster numbers of each sample
    labels_pred = model.labels_
    return labels_pred


def get_baseline_score():
    """
    score baseline estimator based on adjusted_rand_score for each data file
    """
    file_names = ['set1.csv', 'set2.csv']
    for file_name in file_names:
        csv_path = './Data/' + file_name

        # get true labels
        labels_true = pd.read_csv(csv_path)['VID'].to_numpy()

        # get predicted labels
        labels_pred = predictor_baseline(csv_path)

        # calculate adjusted rand score
        rand_index_score = adjusted_rand_score(labels_true, labels_pred)

        print(f'Adjusted Rand Index Baseline Score of {file_name}: {rand_index_score:.4f}')


def best_estimators(X, y, csv_path):
    """
    find the best estimators for each of the model types

    :param X: the features to be fitted
    :param y: the labels to be compared with
    :param csv_path: the path to csv file

    :return: the best models for each types
    """
    best_models = []

    # segment id from csv_path
    id = csv_path.split('./Data/')[1].split('.csv')[0]

    # iterate through all model types
    for key in MODELS.keys():
        gs = GridSearchCV(
            estimator=MODELS[key],
            param_grid=PARAM_GRIDS[key],
            scoring=custom_scorer,
            cv=3,
            refit=True,
            verbose=3 # show the logs
        )

        # log the output to appropriate log files
        with redirect_stdout_to_file(f'logs/gridsearch_{key}_{id}.log'):
            gs.fit(X, y)

        # gs.fit(X, y) # uncomment this if you want to see the logs in the terminal

        # add best model to list 
        best_models.append(gs.best_estimator_)

    return best_models


def predictor(csv_path):
    """
    get the prediction labels of the best estimator 

    :param csv_path: the path to csv file

    :return: the labels of the best model
    """

    # load data and convert hh:mm:ss to seconds
    df = pd.read_csv(csv_path, converters={'SEQUENCE_DTTM' : hh_mm_ss2seconds})
    # select features 
    selected_features = ['SEQUENCE_DTTM', 'LAT', 'LON', 'SPEED_OVER_GROUND' ,'COURSE_OVER_GROUND']

    # split features and labels + standardize the data
    X = df[selected_features].to_numpy()
    X = StandardScaler().fit(X).transform(X)
    y = df['VID'].to_numpy().reshape(-1,1)

    # we do not need train test split because it is unsupervised so we can simply compare with ground truth

    # find the best models
    models = best_estimators(X, y, csv_path)

    # extract file name
    file_name = csv_path.split('./Data/')[1]

    # get best model with highest score among selected models
    best_score = 0
    best_model = None
    for model in models:
        # score each model
        score = custom_scorer(model, X, y)

        print(model, '\n-----------------')
        print(f'Adjusted Rand Index Prediction Score of {file_name}: {score:.4f}')

        # save the best model
        if score > best_score:
            best_score = score
            best_model = model

    # get the labels of the best model
    labels_pred = best_model.fit(X).labels_

    return labels_pred


def get_prediction_score():
    """
    score our estimator based on adjusted_rand_score for each data file
    """
    file_names = ['set1.csv', 'set2.csv']
    for file_name in file_names:
        csv_path = './Data/' + file_name

        # get true labels
        labels_true = pd.read_csv(csv_path)['VID'].to_numpy()

        # get predicted labels
        labels_pred = predictor(csv_path)

        # calculate adjusted rand score
        rand_index_score = adjusted_rand_score(labels_true, labels_pred)
        
        print(f'Adjusted Rand Index Prediction Score of {file_name}: {rand_index_score:.4f}')


def evaluate():
    """
    evaluate the model based on set3 data
    """
    csv_path = './Data/set3.csv'
    labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
    labels_pred = predictor(csv_path)
    rand_index_score = adjusted_rand_score(labels_true, labels_pred)
    print(f'Adjusted Rand Index Score of set3.csv: {rand_index_score:.4f}')


if __name__=="__main__":
    print('baseline score:\n-----------------')
    get_baseline_score()
    print('prediction score:\n-----------------')
    get_prediction_score()
    # print('evaluate:\n-----------------')
    # evaluate()


