import sys
import argparse
import numpy as np

from pathlib import Path


from selfchecker.utils.misc import get_model
from selfchecker.utils.paths import results_path
from selfchecker.core.kdes_generation import train_fetch_kdes, test_fetch_kdes
from selfchecker.core.layer_selection_agree import layer_selection_agree
from selfchecker.core.layer_selection_condition import layer_selection_condition
from selfchecker.core.layer_selection_condition_neg import layer_selection_condition_neg
from selfchecker.core.evaluate import eval_performance


def read_split(x: str, y: str):
    x_path = Path(x)
    y_path = Path(y)

    if not x_path.exists():
        raise ValueError(f"{x_path} does not exist")

    if not y_path.exists():
        raise ValueError(f"{y_path} does not exist")

    if x_path.suffix == '.npy':
        x_data = np.load(x_path)
    else:
        # TODO: add case for files with no headers
        x_data = np.genfromtxt(x_path, delimiter=',', skip_header=1)

    if y_path.suffix == '.npy':
        y_data = np.load(y_path)
    else:
        # TODO: add case for files with no headers
        # TODO: does not handle the case where the labels are not integers
        y_data = np.genfromtxt(y_path, dtype=int, skip_header=1)

    return x_data, y_data


def get_layers(layers: list, only_activation: bool, only_dense: bool):
    target_layers = []

    if only_dense and only_activation:
        include_next = False

        for layer in layers:
            if layer.name.startswith('dense'):
                include_next = True
                target_layers.append(layer.name)
            elif layer.name.startswith('activation') and include_next:
                include_next = False
                target_layers.append(layer.name)
        print("Dense layers and associated activation layers are considered:", target_layers)
    elif only_dense:
        target_layers = [layer.name for layer in layers if 'dense' in layer.name]
        print("Only dense layers are considered:", target_layers)
    elif only_activation:
        target_layers = [layer.name for layer in layers if 'activation' in layer.name]
        print("Only activation layers of dense layers are considered: ", target_layers)
    else:
        target_layers = [layer.name for layer in layers]

    return target_layers


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A self-checking tool for Deep Neural Networks to detect the '
                                                 'potentially incorrect model decision and generate advice to '
                                                 'auto-correct the model decision on runtime.')

    parser.add_argument('-m', '--model', type=str, help='Path to the model', required=True)
    parser.add_argument('-wd', '--workdir', type=str, help='Working directory', required=False)
    parser.add_argument("-bs", "--batch_size", help="Batch size", type=int, default=128)
    parser.add_argument("-oal", "--only_activation_layers", help="Only activation layers",
                        action='store_true')
    parser.add_argument("-odl", "--only_dense_layers", help="Only dense layers", action='store_true')

    action_parser = parser.add_subparsers(dest='action')

    analyze_parser = action_parser.add_parser('analyze')
    analyze_parser.add_argument('-tx', '--train_features', type=str, help='Train features', required=True)
    analyze_parser.add_argument('-ty', '--train_labels', type=str, help='Train labels', required=True)
    analyze_parser.add_argument('-vx', '--val_features', type=str, help='Val. features', required=True)
    analyze_parser.add_argument('-vy', '--val_labels', type=str, help='Val. labels', required=True)
    analyze_parser.add_argument("--var_threshold", "-var_threshold", help="Variance threshold", type=float,
                                default=1e-5)

    infer_parser = action_parser.add_parser('infer')
    infer_parser.add_argument('-tx', '--test_features', type=str, help='Test features', required=True)
    infer_parser.add_argument('-ty', '--test_labels', type=str, help='Test labels', required=True)

    args = parser.parse_args()
    model = get_model(model_path=args.model)
    working_dir = Path(args.workdir) if args.workdir else results_path

    layer_names = get_layers(model.layers, args.only_activation_layers, args.only_dense_layers)

    if args.action == 'analyze':
        x_train, y_train = read_split(args.train_features, args.train_labels)
        x_valid, y_valid = read_split(args.val_features, args.val_labels)

        labels = list(np.unique(y_train))
        num_classes = len(labels)
        print(f"Number of classes: {num_classes}")
        train_fetch_kdes(model, x_train, x_valid, y_train, y_valid, layer_names, working_dir, args.var_threshold,
                         num_classes, args.batch_size)

        layer_selection_agree(len(layer_names), labels, working_dir)
        layer_selection_condition(len(layer_names), labels, working_dir)
        layer_selection_condition_neg(len(layer_names), labels, working_dir)

    elif args.action == 'infer':
        x_test, y_test = read_split(args.test_features, args.test_labels)
        labels = list(np.unique(y_test))
        num_classes = len(labels)

        print(f"Number of classes: {num_classes}")

        test_fetch_kdes(model, x_test, y_test, layer_names, num_classes, args.batch_size, working_dir)
        eval_performance(num_classes, working_dir)

    else:
        print("Please specify a command ['analyze', 'infer'].", file=sys.stderr)
        exit()
