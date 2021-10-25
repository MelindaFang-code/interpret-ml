import scipy
from models_utils import *
import numpy as np
from sklearn.linear_model import LinearRegression



def check_purity_regression(data, threshold):
    return len(data) < threshold

def calculate_mse(data_below, data_above):
    # Getting the means 
    below_mean = np.mean(data_below[:,-1])
    above_mean = np.mean(data_above[:,-1]) 
    # Getting the left and right residuals 
    res_below = data_below[:,-1] - below_mean 
    res_above = data_above[:,-1] - above_mean
    # Concatenating the residuals 
    r = np.concatenate((res_below, res_above), axis=None)

    # Calculating the mse 
    n = len(r)
    r = r ** 2
    r = np.sum(r)
    mse_split = r / n
    return mse_split


def determine_best_split(data, potential_splits):
    print(f"Determining best split for {data.shape}")
    overall_mse = 2147483647
    for column_index in potential_splits:
        print(f"column index: {column_index}")
        time_split = 0
        for value in potential_splits[column_index]:
            
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_mse = calculate_mse(data_below, data_above)
            if current_overall_mse < overall_mse:
                overall_mse = current_overall_mse
                print(f"overall mse: {overall_mse}")
                best_split_column = column_index
                best_split_value = value

    return best_split_column, best_split_value

def regression_tree(x_train, y_train):
    data = np.concatenate((x_train, y_train), axis=1)
    regression_tree_helper(data, (int)(0.95*len(data)))


def regression_tree_helper(data, threshold):

    print(data.shape)
    if len(data) < threshold:
        return data[:, -1]
    else:
        potential_splits = get_potential_splits(data)
        split_column, split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)

        # instantiate sub-tree
        question = "{} <= {}".format(split_column, split_value)
        sub_tree = {question: []}

        # find answers (recursion)
        yes_answer = regression_tree_helper(data_below, threshold)
        no_answer = regression_tree_helper(data_above, threshold)

        sub_tree[question].append(yes_answer)
        sub_tree[question].append(no_answer)

        return sub_tree


def classify_example(example, decision_tree, q):
    question = list(decision_tree.keys())[0] # decision_tree.keys() returns a dictionary
    feature_name, comparison_operator, value = question.split() # question is of form '0' <= 0.5'

    # ask question
    if example[int(feature_name)] <= float(value):
        answer = decision_tree[question][0]
    else:
        answer = decision_tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return np.quantile(answer, q)

    # recursive part
    else:
        residual_tree = answer
        return classify_example(example, residual_tree)


# =================== GAM ===================
def mat_sqrt(X):
    ''' Returns square root of a matrix by eigen decomposition'''
    from scipy.linalg import sqrtm
    # w, v = np.linalg.eig(X)
    # D = np.diag(w ** 0.5)
    # return v @ D @ v.T #inverse equals transpose for orthogonal matrix
    return sqrtm(X)
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
    
class MyLinearRegression:
    ''' Linear regression by solving normal equations. For now.'''
    
    def __init__(self, X, y):
        self.X = X
        self.y = y.reshape(-1,1)
        self.b_hat = 0
        
    
    def fit(self):
        self.b_hat = np.linalg.inv((self.X.T @ self.X).astype('float64')) @ self.X.T @ self.y

    def fit_sklearn(self):
        ''' Uses sklearn'''
        self.b_hat = LinearRegression(fit_intercept=False).fit(self.X, self.y).coef_.flatten()
        print(self.b_hat)

    def predict(self, X):
        return X @ self.b_hat


class AdditiveModel:
    ''' Additive Model for multiple covariates'''
    
    def __init__(self, X_in, lmdas, q=10):
        self.X_in = X_in # covariate matrix
        self.n = X_in.shape[0] # number of data
        self.p = X_in.shape[1] # number of covariates
        self.q = q # number of knots?
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
        S = np.zeros(S_list[0].shape)
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

class Scaler:
    
    def __init__(self, X):
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
    
    def transform(self, X):
        return (X - self.min) / (self.max - self.min)

    def rev_transform(self, X_s):
        return X_s * (self.max - self.min) + self.min
