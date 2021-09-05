import argparse
from dataset import load_dataset
from parse import parse_model, parser_add_main_args
from sklearn.metrics import accuracy_score


parser = argparse.ArgumentParser(description='General Execution Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

dataset = load_dataset(args.dataset, args.train_prop, args.valid_prop)
model = parse_model(args)

model.fit(dataset.X_train, dataset.y_train)
print(f"Model Accuracy: {model.score(dataset.X_test, dataset.y_test)}")

