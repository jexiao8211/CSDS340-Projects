import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


train_data = pd.read_csv('Data/train.csv')
X_train = train_data.drop(columns = ['Quality'])
y_train = train_data.Quality

test_data = pd.read_csv('Data/test.csv')
X_test = test_data.drop(columns = ['Quality'])
y_test = test_data.Quality

pipeline = make_pipeline(StandardScaler(), PCA(n_components=7), SVC(C=10.0, gamma='scale', random_state=1))
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)

print(f'accuracy_score: {acc*100:.2f}%')
print(f'recall_score: {rec*100:.2f}%')
print(f'f1_score: {f1*100:.2f}%')
print(f'precision_score: {prec*100:.2f}%')




