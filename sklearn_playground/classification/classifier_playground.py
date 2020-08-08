# knn classifier

from sklearn.datasets import make_moons, make_blobs, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import inspect
import streamlit as st

st.title("Choose your classifier")


k = st.selectbox('How would you like to be contacted?', np.arange(1, 14))
st.write('You selected:', k)

def select_dataset(dataset_selected, noise=0):
    if dataset_selected == 'moons':
        x, y = make_moons(noise=noise)
    elif dataset_selected == 'blobs':
        x, y = make_blobs(cluster_std=noise)
    elif dataset_selected == 'circles':
        x, y = make_circles(noise=noise)
    elif dataset_selected == 'linear':
        x, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                                random_state=1, n_clusters_per_class=1, class_sep=noise)
    
    return x, y

def select_model(model_selected, x, y, model_dict):
    x_train, x_test, y_train, y_test = train_test_split(x,y)
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
    if model_selected == 'SVC':
        model = model_dict.get(model_selected)(probability=True)
    elif model_selected == 'SGDClassifier':
        model = model_dict.get(model_selected)(loss='modified_huber')
    else:
        model = model_dict.get(model_selected)()
        
    st.markdown(inspect.signature(model.__init__))


    model.fit(x_train,y_train)
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)
    st.pyplot()


model_list_str = [
    'KNeighborsClassifier',
    'LogisticRegression',
    'SGDClassifier',
    'LinearDiscriminantAnalysis',
    'QuadraticDiscriminantAnalysis',
    'SVC'
]

model_list = [
    KNeighborsClassifier,
    LogisticRegression,
    SGDClassifier,
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
    SVC
]

model_dict = dict(zip(model_list_str, model_list))

h = .02
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

noise = st.number_input('Dataset noise')
x, y = select_dataset(st.selectbox('Select your dataset', ['moons', 'blobs', 'circles', 'linear']), noise)

select_model(st.selectbox('Select your model', model_list_str), x, y, model_dict)

