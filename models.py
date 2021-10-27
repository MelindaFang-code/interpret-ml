from models_utils import *
import numpy as np


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

def calc_mse_whole(data):
    r = data[:,-1]-np.mean(data[:,-1])
    n = len(r)
    r = r ** 2
    r = np.sum(r)
    mse = r / n
    return mse


def determine_best_split(data, potential_splits):
#     print(f"Determining best split for {data.shape}")
    overall_mse = calc_mse_whole(data)
    for column_index in potential_splits:
#         print(f"column index: {column_index}")
        time_split = 0
        for value in potential_splits[column_index]:
            
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_mse = calculate_mse(data_below, data_above)
            if current_overall_mse < overall_mse:
                overall_mse = current_overall_mse
#                 print(f"overall mse: {overall_mse}")
                best_split_column = column_index
                best_split_value = value

    return best_split_column, best_split_value

def regression_tree(x_train, y_train):
    data = np.concatenate((x_train, y_train), axis=1)
    regression_tree_helper(data, (int)(0.95*len(data)))


def regression_tree_helper(data, threshold):

#     print(data.shape)
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
        return classify_example(example, residual_tree, q)



