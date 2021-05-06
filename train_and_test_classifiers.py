from sklearn.model_selection import train_test_split
from sklearn import svm, metrics, preprocessing
from sklearn.model_selection import RandomizedSearchCV
from skopt.space import Real, Categorical, Integer
from scipy.stats import reciprocal, uniform
from sklearn.metrics import f1_score
import numpy as np
import math


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)



#=============================================================================#
#------------------------------VARIABLES TO FILL IN---------------------------#
data_path = 'root/Model Data_6/'
data_normalization = True
sklearn_scaler = preprocessing.MaxAbsScaler()
test_data_ratio = 0.25
oversampling = False
oversampling_ratio = 10
num_iter = 40  # RandomizedSearch iterations
#=============================================================================#
#-----------------------------------------------------------------------------#



def train_svm_classifier(kernel_type, param_dist, X_train, y_train):
    ''' Tune hyperparameters and train SVM

    kernel_type -- Type of kernel (Linear,RBF, Polynomial, Sigmoid)
    param_dist -- Dictionary of hyperparameters to be tuned and the search range
    X_train -- training data
    y_train -- training labels
    '''
    svm_model = svm.SVC(kernel=kernel_type)            
    rand_search = RandomizedSearchCV(svm_model, param_distributions = param_dist, \
        n_iter = num_iter, random_state = 555)

    rand_search.fit(X=X_train, y=y_train) 
    print("Best params for "+kernel_type+" svm: ",rand_search.best_params_)
    return rand_search.best_estimator_

def test_classifier(classifier, kernel, X_test, y_test):
    ''' Test model and print evaluation metrics '''
    y_pred = classifier.predict(X_test)
    print("\n"+kernel+" kernel accuracy:",metrics.accuracy_score(y_test, y_pred))
    print(kernel+" kernel precision:",metrics.precision_score(y_test, y_pred))
    print(kernel+" kernel recall:",metrics.recall_score(y_test, y_pred))    


# Location of datasets
corner_data = np.load(data_path+'corner_data.npy')
non_corner_data = np.load(data_path+'non_corner_data.npy')


if data_normalization:
    # Normalize data to [0,1]
    corner_scaler = sklearn_scaler.fit(corner_data)
    corner_data = corner_scaler.transform(corner_data)
    non_corner_scaler = sklearn_scaler.fit(non_corner_data)
    non_corner_data = non_corner_scaler.transform(non_corner_data)
    pos_len = corner_data.shape[0]
    neg_len = non_corner_data.shape[0]


total_data = np.concatenate((corner_data, non_corner_data), axis=0)
print(total_data.shape)
labels = np.ones(pos_len+neg_len)
labels[pos_len:] = np.zeros([neg_len])


X_train, X_test, y_train, y_test = train_test_split(total_data, labels, \
    test_size=test_data_ratio, random_state=879, stratify=labels)

if oversampling:
    # Do oversampling if corner dataset too small (doesn't work well).
    X_train_clone = np.tile(X_train[np.where(y_train==1)],(oversampling_ratio-1,1))
    y_train_clone = np.tile(y_train[y_train==1],(oversampling_ratio-1))

    X_train = np.concatenate((X_train, X_train_clone), axis=0)
    y_train = np.concatenate((y_train, y_train_clone), axis=0)

    shuffled_indices = np.random.permutation(len(y_train))
    X_train = X_train[shuffled_indices]
    y_train = y_train[shuffled_indices]




rbf_param_dist = {"C": uniform(1, 100), "gamma": uniform(0.0001, 10)}  
rbf_svm = train_svm_classifier('rbf', rbf_param_dist, X_train, y_train)
test_classifier(rbf_svm, 'rbf', X_test, y_test)

poly_param_dist = {"C": uniform(1, 100)}  
poly_svm = train_svm_classifier('poly', poly_param_dist, X_train, y_train)
test_classifier(poly_svm, 'poly', X_test, y_test)