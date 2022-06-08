import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from src.model.wncn import *
#from src.model.instance_hardness import *
#from src.model.support_subset import *
#from sklearn.model_selection import train_test_split

np.random.seed(1234)

n = 10000

mu1, mu2, sigma = 0, -2, 1

X1 = np.random.normal(mu1, sigma, size=[n, 2])
X2 = np.random.normal(mu2, sigma, size=[n, 2])
X3 = np.random.normal([-2, 2], 0.5, size=[n, 2])
X = np.vstack((X1, X2, X3))

y = np.hstack((np.zeros(int(len(X1))),np.ones(int(len(X2) + len(X3))))).astype(int)

dynamic_hardness = DkDN(k=3, n_jobs = -1)
dynamic_hardness.fit(X, y)