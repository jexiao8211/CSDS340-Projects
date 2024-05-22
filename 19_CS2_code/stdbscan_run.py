import logging
import time
import functools

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import adjusted_rand_score

from st_dbscan import ST_DBSCAN


logger = logging.getLogger('stdbscan_logger')
logger.setLevel(logging.INFO)


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


def predictor(csv_path):
    """
    get the prediction labels of the best estimator 

    :param csv_path: the path to csv file

    :return: the labels of the best model
    """

    # segment id from csv_path
    id = csv_path.split('./Data/')[1].split('.csv')[0]

    # set the file name
    file_name = f'logs/gridsearch_stdbscan_{id}.log'

    # config file handler
    file_handler = logging.FileHandler(filename=file_name, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # load data
    df = pd.read_csv(csv_path, converters={'SEQUENCE_DTTM' : hh_mm_ss2seconds})
    selected_features = ['SEQUENCE_DTTM', 'LAT', 'LON', 'SPEED_OVER_GROUND' ,'COURSE_OVER_GROUND']

    # split features and labels + standardize the data
    X = df[selected_features].to_numpy()
    X = StandardScaler().fit(X).transform(X)
    y = df['VID'].to_numpy().reshape(-1,1)

    # the param grid of ST_DBSCAN
    param_grid = {
        'eps1': [0.1, 0.3, 0.5, 1],
        'eps2': [0.1, 0.3, 0.5, 1],
        'min_samples': [1, 3, 5],
    }

    # number of folds
    cv = 3

    # get best model with highest score among selected models
    best_score = 0
    best_model = None

    candidates_cnt = int(len(param_grid['eps1']) * len(param_grid['eps2']) * len(param_grid['min_samples']))
    logger.info(f'Fitting {cv} folds for each of {candidates_cnt}, totaling {int(cv*candidates_cnt)} fits')

    # iterate through the param grid
    for eps1 in param_grid['eps1']:
        for eps2 in param_grid['eps2']:
            for min_samples in param_grid['min_samples']:
                # the average score of the model
                avg_score = 0

                # initialize the model
                model = ST_DBSCAN(eps1=eps1, eps2=eps2, min_samples=min_samples) 

                for i in range(cv):
                    # print(f'ST_DBSCAN(eps1={eps1}, eps2={eps2}, min_samples={min_samples})')

                    # start time of the training
                    start_time = time.time()

                    # score the model
                    score = custom_scorer(model, X, y)

                    # end time of the training
                    end_time = time.time()

                    # total time of the model
                    total_time = end_time - start_time
                    
                    # add to average score
                    avg_score = avg_score + score

                    # get the logs
                    logger.info(f'[CV {i+1}/{cv}] END eps1={eps1}, eps2={eps2}, min_samples={min_samples};, score={score:.3f} total time=   {total_time:.1f}')

                avg_score = avg_score / cv

                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model

    # get the labels of the best model
    labels_pred = best_model.fit(X).labels_

    # remove handler for next file
    logger.removeHandler(file_handler)

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
    print('prediction score:\n-----------------')
    get_prediction_score()
    # print('evaluate:\n-----------------')
    # evaluate()