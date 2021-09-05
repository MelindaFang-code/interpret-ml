from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, LogisticRegression


def parse_model(args):
    """ Returns a model with sklearn interface (in particular fit(X,y)) """
    if args.method == 'LinearRegression':
        model = LinearRegression()
    elif args.method == 'LogisticRegression':
        model = LogisticRegression()
    elif args.method == 'RandomForest':
        model = RandomForestClassifier()
    elif args.method == 'SVM':
        model = SVC()
    else:
        raise ValueError('Invalid model name')
    return model


def parser_add_main_args(parser):
    """ Add arguments to parser """
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--method', '-m', type=str, default='RandomForest')
    parser.add_argument('--train_prop', type=float, default=0.8, help='Proportion of data to use for training')
    parser.add_argument('--valid_prop', type=float, default=0.1, help='Proportion of data to use for validation set')




