import numpy as np


def entropy_loss(p: np.array) -> float:
    kl_divergence = - np.sum(p * np.log(p))
    return kl_divergence


def gini_loss(p: np.array) -> float:
    gini_impurity = np.sum(p * (1 - p))
    return gini_impurity


def check_purity(data):
    label_column = data[:, -1]
    return len(np.unique(label_column)) == 1


def classify_data(data):
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]

    return classification


def get_potential_splits(data):
    print(f"finding splits for {data.shape}")
    potential_splits = {}
    n_columns = data.shape[1]
    for column_index in range(n_columns - 1):
        potential_splits[column_index] = []
        values = data[:, column_index]
        unique_values = np.unique(values)

        for index in range(1, len(unique_values)):
            previous_value = unique_values[index - 1]
            current_value = unique_values[index]
            potential_split = (current_value + previous_value) / 2

            potential_splits[column_index].append(potential_split)

    return potential_splits


def split_data(data, split_column, split_value):
    split_column_values = data[:, split_column]

    data_below = data[split_column_values <= split_value]
    data_above = data[split_column_values > split_value]

    return data_below, data_above


def calculate_entropy(data):
    labels = data[:, -1]
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    entropy = entropy_loss(p)
    return entropy


def calculate_overall_entropy(data_below, data_above):
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_entropy = (p_data_below * calculate_entropy(data_below)
                       + p_data_above * calculate_entropy(data_above))

    return overall_entropy


def determine_best_split(data, potential_splits):
    print(f"Determining best split for {data.shape}")
    overall_entropy = 2147483647
    for column_index in potential_splits:
        print(f"column index: {column_index}")
        time_split = 0
        for value in potential_splits[column_index]:
            
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)
            if current_overall_entropy < overall_entropy:
                overall_entropy = current_overall_entropy
                print(f"overall entropy: {overall_entropy}")
                best_split_column = column_index
                best_split_value = value

    return best_split_column, best_split_value


def decision_tree(x_train, y_train):
    data = np.concatenate((x_train, y_train), axis=1)
    decision_tree_helper(data)


def decision_tree_helper(data):
    print(data.shape)
    if check_purity(data):
        classification = classify_data(data)
        return classification
    else:
        potential_splits = get_potential_splits(data)
        split_column, split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)

        # instantiate sub-tree
        question = "{} <= {}".format(split_column, split_value)
        sub_tree = {question: []}

        # find answers (recursion)
        yes_answer = decision_tree_helper(data_below)
        no_answer = decision_tree_helper(data_above)

        sub_tree[question].append(yes_answer)
        sub_tree[question].append(no_answer)

        return sub_tree


def classify_example(example, decision_tree):
    question = list(decision_tree.keys())[0] # decision_tree.keys() returns a dictionary
    feature_name, comparison_operator, value = question.split() # question is of form '0' <= 0.5'

    # ask question
    if example[int(feature_name)] <= float(value):
        answer = decision_tree[question][0]
    else:
        answer = decision_tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer

    # recursive part
    else:
        residual_tree = answer
        return classify_example(example, residual_tree)


