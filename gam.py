# %%
import numpy as np
import matplotlib.pyplot as plt
from models import *

size = np.array([1.42,1.58,1.78,1.99,1.99,1.99,2.13,2.13,2.13,2.32,2.32,2.32,2.32,2.32,2.43,2.43,2.78,2.98,2.98])
wear = np.array([4.0,4.2,2.5,2.6,2.8,2.4,3.2,2.4,2.6,4.8,2.9,3.8,3.0,2.7,3.1,3.3,3.0,2.8,1.7])

# Normalize
x = (size - min(size)) 
x /= np.max(x)

xk = np.arange(1,5) / 5
spl = Spline(x, xk)
X = spl.X()


lm = lin_reg(X, wear)
xp = np.arange(0, 101)/100
spl_p = Spline(xp, xk)
Xp = spl_p.X()
plt.plot(x, wear, '.')
# plt.plot(xp, lm.predict(Xp), '.')
b_hat = lin_reg_normal(X, wear)
plt.plot(xp, Xp @ b_hat)

# %% Penalized Regression Spline
lmda = 0.001
xk = np.arange(1,8) / 8
spl = PRSpline(x, xk)
Xa = spl.prs_matrix(lmda)
y = np.concatenate((wear.reshape(-1,1), np.zeros((spl.q, 1))), axis=0) # augment the data vector
lm_prs = lin_reg(Xa, y)
# lm_prs = MyLinearRegression(Xa, y)
# lm_prs.fit()
xp = np.arange(0, 101)/100
spl_p = PRSpline(xp, xk)
Xp = spl_p.prs_matrix(lmda)
plt.plot(x, wear, '.')
pred = lm_prs.predict(Xp)
plt.plot(xp, pred[:-spl_p.q], '.')


# %% Additive Model
import pandas as pd
data = pd.read_csv('trees.csv', index_col=0).to_numpy()
X = data[:,:-1]
y = data[:,-1]
# Rescale predictors onto [0,1]

scaler_X = Scaler(X)

scaler_y = Scaler(y)
X_s = scaler_X.transform(X)
y_s = scaler_y.transform(y)

lmdas = [0.01024, 5368.70912]
am = AdditiveModel(X_s, lmdas, 10)
am.fit(y_s)
plt.plot(y, scaler_y.rev_transform(am.predict(X_s)), '.')

# %%
from dataset import *
from models import *
import matplotlib.pyplot as plt

ames = load_dataset("ames_housing", 0.7, 0.3)
X_tr = ames.X_train[:,:2]
y_tr = ames.y_train
print("y_tr", y_tr)

scaler_X = Scaler(X_tr)
scaler_y = Scaler(y_tr)
X_trs = scaler_X.transform(X_tr)
y_trs = scaler_y.transform(y_tr)

lmdas = [0.01024, 5368.70912]
am = AdditiveModel(X_trs, lmdas, 10)
am.fit(y_trs)
y_pred = scaler_y.rev_transform(am.predict(X_trs))
print("y_pred", y_pred)
plt.plot(y_tr, y_pred, '.')

# %%
