# %%
from os import name
import dataset
data = dataset.load_dataset('taiwan_credit_risk', 0.7, 0.3)
# %%
import numpy as np
import matplotlib.pyplot as plt

size = np.array([1.42,1.58,1.78,1.99,1.99,1.99,2.13,2.13,2.13,2.32,2.32,2.32,2.32,2.32,2.43,2.43,2.78,2.98,2.98])
wear = np.array([4.0,4.2,2.5,2.6,2.8,2.4,3.2,2.4,2.6,4.8,2.9,3.8,3.0,2.7,3.1,3.3,3.0,2.8,1.7])

# Normalize
x = (size - min(size)) 
x /= np.max(x)

# %%


def mat_sqrt(X):
    ''' Returns square root of a matrix by eigen decomposition'''
    w, v = np.linalg.eig(X)
    D = np.diag(w ** 0.5)
    return v @ D @ v.T #inverse equals transpose for orthogonal matrix
class Spline:

    def __init__(self, x, xk):
        self.x = x
        self.xk = xk # vector of knots
        self.q = len(self.xk) + 2 # dimension of basis

    @staticmethod
    def R(x,z):
        ''' R(x,z) for cubic spline on [0,1]'''
        return ((z-0.5)**2-1/12) * ((x-0.5)**2-1/12)/4-((abs(x-z)-0.5)**4-(abs(x-z)-0.5)**2/2+7/240)/24

    def X(self):
        ''' Set up model matrix for cubic penalized regression spline'''
        n = len(self.x) # number of data points
        X = np.concatenate((np.ones((n,1)), self.x.reshape(n,1), np.frompyfunc(Spline.R,2,1).outer(self.x, self.xk)), axis=1)
        assert X.shape == (n,self.q)
        return X

class PRSpline(Spline):
    '''Penalized Regression Spline'''
    def __init__(self, x, xk):
        super().__init__(x, xk)

    
    def S(self):
        ''' Set up the penalized regression spline penalty matrix'''
        S = np.zeros((self.q, self.q))
        S[2:, 2:,] = np.frompyfunc(PRSpline.R,2,1).outer(self.xk, self.xk)
        return S
    
    def prs_matrix(self, lmbda):
        ''' Returns the penalized regression spline matrix'''
        return np.concatenate((self.X(), (mat_sqrt(self.S())*(lmbda**0.5))), axis=0)
    
def lin_reg(X, y):
    ''' Return linear model'''
    from sklearn.linear_model import LinearRegression
    # TODO: Don't use sklearn lol
    return LinearRegression(fit_intercept=False).fit(X, y)

def lin_reg_normal(X, y):
    ''' Returns b_hat by solving normal equations'''
    y = y.reshape(-1, 1)
    return np.linalg.inv((X.T @ X).astype('float64')) @ X.T @ y

class MyLinearRegression:
    ''' Linear regression by solving normal equations. For now.'''
    
    def __init__(self, X, y):
        self.X = X
        self.y = y.reshape(-1,1)
        self.b_hat = 0
        
    
    def fit(self):
        self.b_hat = np.linalg.inv((self.X.T @ self.X).astype('float64')) @ self.X.T @ self.y

    def predict(self, X):
        return X @ self.b_hat


class AdditiveModel:
    ''' Additive Model for multiple covariates'''
    
    def __init__(self, X_in, lmdas, q=10):
        self.X_in = X_in # covariate matrix
        self.n = X_in.shape[0] # number of data
        self.p = X_in.shape[1] # number of covariates
        self.q = q # number of knots?
        # self.X = None # model matrix
        # self.S_list = [] # list of penalty matrices
        self.lmdas = lmdas # list of smoothing parameters
        self.lm = None
    
    def get_X_S_list(self, X_in):
        ''' Get X and list of S for a simple AM'''
        i = self.p - 1
        j = self.q
        num_col = self.p * self.q - (self.p-1)
        X = np.ones((self.n , num_col))
        S_list = []

        for k in range(self.p):

            x = X_in[:, k]
            # Choose knots
            xk = np.quantile(np.unique(x), np.arange(1, self.q - 1) / (self.q -1))# TODO: not sure why unique is needed
            
            # Get penalty matrices
            S = np.zeros((num_col, num_col))  
            spl = PRSpline(x, xk)
            S[i:j, i:j] = spl.S()[1:, 1:] # drop rows and columns corresponding to the intercept of each term
            S_list.append(S)

            # Get model matrix
            X[:, i:j] = spl.X()[:, 1:]

            i = j
            j += self.q - 1

        return X, S_list

    def augment_X(self, X, S_list):
        S = np.empty(S_list[0].shape)
        for Si, lmda in zip(S_list, self.lmdas):
            S += lmda * Si
         
        rS = mat_sqrt(S) # Get sqrt of total penalty matrix
        return np.concatenate((X, rS), axis=0)

    def fit(self, y):
        ''' lmdas is a list of smoothing parameters'''
        # Get model matrix and list of penalty matrix
        X, S_list = self.get_X_S_list(self.X_in)
       
        X1 = self.augment_X(X, S_list)
        q = X.shape[1] # number of parameter in the processed model matrix
        y1 = np.concatenate((y.reshape(-1,1), np.zeros((q, 1))), axis=0)
        self.lm = MyLinearRegression(X1, y1)
        self.lm.fit()

    def predict(self, Xp):
        n = Xp.shape[0]
        X, S_list = self.get_X_S_list(Xp)
        X1 = self.augment_X(X, S_list)
        return self.lm.predict(X1)[:n, :]


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

  # %# %%

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

class Scaler:
    
    def __init__(self, X):
        self.min = np.min(X)
        self.max = np.max(X)
    
    def transform(self, X):
        return (X - self.min) / (self.max - self.min)

    def rev_transform(self, X_s):
        return X_s * (self.max - self.min) + self.min

scaler_X = Scaler(X)
scaler_y = Scaler(y)
X_s = scaler_X.transform(X)
y_s = scaler_y.transform(y)

lmdas = [0.01024, 5368.70912]
am = AdditiveModel(X_s, lmdas, 10)
am.fit(y_s)
plt.plot(y, scaler_y.rev_transform(am.predict(X_s)), '.')

# %%
