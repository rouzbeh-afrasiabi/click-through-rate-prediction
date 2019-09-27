import pandas as pd
from plotnine import *
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import datetime
from time import time
import holidays
import os
import sys
import zipfile
import requests
import shutil

import warnings

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

import datetime
import re

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn.linear_model import (SGDClassifier,RidgeClassifier,Perceptron,LogisticRegression)
from sklearn.preprocessing import PolynomialFeatures

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture 
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier

from sklearn import metrics


from pycm import *
import seaborn as sns



from imblearn.pipeline import make_pipeline
from imblearn import under_sampling 
from imblearn import over_sampling 
from imblearn import combine
from imblearn.over_sampling import SMOTE

import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib as mpl


pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')
us_holidays = holidays.UnitedStates()
ca_holidays=holidays.CA()
uk_holidays=holidays.UK()


random_state=123456
np.random.seed(random_state)

cwd = str(os.getcwd())
sys.path.append(cwd)
sys.path.insert(0, cwd)


scaler_fs=[MinMaxScaler,MaxAbsScaler,StandardScaler,RobustScaler,QuantileTransformer,PowerTransformer]
scalers=dict(zip([scaler.__name__ for scaler in scaler_fs],scaler_fs))