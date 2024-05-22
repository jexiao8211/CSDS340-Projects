"""
:class: CSDS 340
:assignment: Case Study 1
:authors: Lam Nguyen (ltn18) & Jerry Xiao (jxx419)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SequentialFeatureSelector as SFS

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

# set to universal random state
_random_state = 1

# parameter grid to perform tuning
param_grids = {
    'lr': {
        'pca': {
            'dim_reduce__n_components': list(range(2, 8)),
            'model__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100],
        },
        'lda': {
            'dim_reduce__n_components': [1],
            'model__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100],
        },
    },
    'svm': {
        'pca': {
            'dim_reduce__n_components': list(range(2, 8)),
            'model__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100],
            'model__gamma': ['scale'],
        },
        'lda': {
            'dim_reduce__n_components': [1],
            'model__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100],
            'model__gamma': ['scale', 'auto'],
        },
    },
    'dt': {
        'pca': {
            'dim_reduce__n_components': list(range(2, 8)),
            'model__max_depth': [None, 2, 4, 6, 8, 10],
        },
        'lda': {
            'dim_reduce__n_components': [1],
            'model__max_depth': [None, 2, 4, 6, 8, 10],
        },
    },
}

# initialize all models with the same random state
models = {
    'lr': LogisticRegression(random_state=_random_state),
    'svm': SVC(random_state=_random_state),
    'dt': DecisionTreeClassifier(random_state=_random_state),
}

# initialize all dimensionality reduction methods
dim_reduce_methods = {
    'pca': PCA(),
    'lda': LDA(),
}

# keys to access dimensionality reduction methods
dim_reduce_keys = ['pca', 'lda']

# keys to access models in the parameter grid
model_keys = ['lr', 'svm', 'dt']

# scoring metrics for grid searching
scorings = ['accuracy', 'f1', 'recall', 'precision']


def check_data_dist(data, X):
    """
    check the distribution of the dataset.

    :param data: dataset being checked
    :param X: features matrix of the dataset
    """

    # get the description of the data
    summary = data.describe()
    print(summary)

    # Plot histograms for each feature
    for column in X.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(X[column], kde=True)
        plt.title(f'{column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()


def print_unique_labels(y):
    """
    get all unique labels of the dataset.

    :param y: vector containing labels for the dataset
    """
    print("unique labels:", y.unique())


def print_label_count(y):
    """
    get the count for each of the labels in the dataset.

    :param y: vector containing labels for the dataset
    """
    print("Quality = 0:", len(y[y == 0]))
    print("Quality = 1:", len(y[y == 1]))


def load_data():
    """
    load the dataset
    """
    data = pd.read_csv('Data/train.csv')
    X = data.drop(columns = ['Quality'])
    y = data.Quality
    return data, X, y


def split_data(X, y):
    """
    split the dataset with equal data distribution in training and testing set.

    :param X: features matrix of the dataset
    :param y: vector containing labels for the dataset
    """

    # training set: 80% of the examples; testing set: 20% of the examples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=_random_state, stratify=y)
    return X_train, X_test, y_train, y_test


def create_pipeline(dim_reduce, model):
    """
    create a pipeline for end-to-end model evaluation.

    :param dim_reduce: dimensionality reduction method
    :param model: classifier model being considered
    :return: a pipeline with sequential steps of evaluating a model
    """
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('dim_reduce', dim_reduce),
        ('model', model)
    ])
    return pipeline


def grid_search_model(
    dim_reduce_key, model_key, scoring,
    X_train, X_test, y_train, y_test,
    cv, refit=True, n_jobs=-1
):
    """
    tune hyperparameters for a specified configuration of model and dimensionality reduction method.

    :param dim_reduce_key: key to access the dimensionality reduction method
    :param model_key: key to access the classifier model being considered
    :param scoring: the scoring metric being used for model evaluation
    :param X_train: training features matrix
    :param X_test: testing features vector
    :param y_train: training labels matrix
    :param y_test: testing labels vector
    :param cv: the number of folds being used in k-folds
    :param refit: refit an estimator using the best found parameters on the whole dataset
    :param n_jobs: the number of jobs to run in parallel
    :return: best classifier, best parameters, accuracy
    """

    # access the dimensionality reduction method
    dim_reduce = dim_reduce_methods[dim_reduce_key]

    # access the classifier model
    model = models[model_key]

    # access the param grid associated with the configurations of model + dim_reduce
    param_grid = param_grids[model_key][dim_reduce_key]

    # create a pipeline with specified dimensionality reduction method and classifier model
    pipeline = create_pipeline(dim_reduce, model)

    # grid searching to find the best model based on a predefined scoring and k-folds
    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        refit=refit,
        n_jobs=n_jobs)
    
    # fit training set to the grid search
    gs = gs.fit(X_train, y_train)

    # get the best parameter configuration
    params = gs.best_params_

    # access the best classifier
    clf = gs.best_estimator_

    # fit the training set to the best classifier
    clf.fit(X_train, y_train)

    # score the performance of the best classifier
    acc = clf.score(X_test, y_test)

    # write the tuning output to a file
    with open('tune_output.txt', 'a') as file:
        file.write(f'>>> model {model_key} + dim_reduce {dim_reduce_key} + scoring {scoring}: {acc*100:.2f}%\n')
    
    return clf, params, acc


def test_with_kaggle_data(best_clf):
    """
    test with data downloaded from kaggle.

    :param best_clf: the best classifier model obtained
    :return: accuracy, recall, f1, and precision score
    """

    # load kaggle data
    kaggle_data = pd.read_csv('Data/kaggle.csv')

    # drop the A_id column
    kaggle_data.drop(columns = ['A_id'], inplace=True)

    # drop rows containing NaN values
    kaggle_data.dropna(axis=0, inplace=True)

    # extract feature matrix
    X_k = kaggle_data.drop(columns = ['Quality'])

    # extract label matrix
    y_k = kaggle_data.Quality.map({'good': 1, 'bad': 0})

    # split data
    X_k_train, X_k_test, y_k_train, y_k_test = split_data(X_k, y_k)

    # fit training set to the best classifier model
    best_clf.fit(X_k_train, y_k_train)

    # obtain prediction of the best classifier model on testing set
    y_k_pred = best_clf.predict(X_k_test)

    # score evaluation metrics
    acc = accuracy_score(y_k_test, y_k_pred)
    rec = recall_score(y_k_test, y_k_pred)
    f1 = f1_score(y_k_test, y_k_pred)
    prec = precision_score(y_k_test, y_k_pred)

    return acc, rec, f1, prec


def find_best_clf(X, y):
    """
    find the best classifier among all candidates.

    :param X: features matrix of the dataset
    :param y: vector containing labels for the dataset
    :return: 
        best accuracy, best classifier model, best dimensionality reduction method,
        best scoring metric used, best classifier model, best parameter configuration
    """

    # split the dataset
    X_train, X_test, y_train, y_test = split_data(X, y)

    # housekeeping variables to be returned
    best_acc = 0
    best_dim_reduce = ''
    best_model = ''
    best_scoring = ''
    best_clf = None
    best_params = None

    for model_key in model_keys:
        for dim_reduce_key in dim_reduce_keys:
            for scoring in scorings:
                # search the configuration of model + dim_reduce + scoring
                clf, params, acc = grid_search_model(
                    dim_reduce_key, model_key, scoring,
                    X_train, X_test, y_train, y_test, cv=5
                )

                # save the so-far best configuration
                if acc > best_acc:
                    best_acc = acc
                    best_model = model_key
                    best_dim_reduce = dim_reduce_key
                    best_scoring = scoring
                    best_clf = clf
                    best_params = params

    return best_acc, best_model, best_dim_reduce, best_scoring, best_clf, best_params


def main():
    """
    main method for running the project
    """

    # load dataset
    data, X, y = load_data()

    # find the best classifier model
    best_acc, best_model, best_dim_reduce, best_scoring, best_clf, best_params = find_best_clf(X, y)

    # test the best classifier model with data downloaded from kaggle
    kaggle_acc, kaggle_rec, kaggle_f1, kaggle_prec = test_with_kaggle_data(best_clf)

    # write the output of training to a file
    with open('output.txt', 'w+') as file:
        file.write('------------------------------------\n')
        file.write('>>> Best Classifier Model\n')
        file.write(f'Test Accuracy: {best_acc*100:.2f}%\n')
        file.write(f'best_model: {best_model}\n')
        file.write(f'best_params: {best_params}\n')
        file.write(f'best_dim_reduce: {best_dim_reduce}\n')
        file.write('------------------------------------\n')
        file.write('>>> Test with Kaggle Data\n')
        file.write(f'accuracy_score: {kaggle_acc*100:.2f}%\n')
        file.write(f'recall_score: {kaggle_rec*100:.2f}%\n')
        file.write(f'f1_score: {kaggle_f1*100:.2f}%\n')
        file.write(f'precision_score: {kaggle_prec*100:.2f}%\n')
        file.write('------------------------------------\n')

    # write the submission test accuracy of right format to a file
    with open('submission.txt', 'w+') as file:
        file.write(f'Test Accuracy: {best_acc*100:.2f}%\n')


# main method
if __name__ == "__main__":
    main()
    

