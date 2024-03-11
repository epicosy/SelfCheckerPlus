# %%
import argparse
import os
import numpy as np
import pandas as pd

from tensorflow import keras
from pathlib import Path
from keras.models import load_model

from kdes_generation import train_fetch_kdes, test_fetch_kdes

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # set GPU Limits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", help="Dataset", type=str, choices=['PD', 'HP', 'BM'],
                        required=True)
    parser.add_argument("--model", "-m", help="Model", type=str, required=True)
    parser.add_argument("--flag", "-f", help="flag", type=str, choices=["train", "test"],
                        required=True)
    parser.add_argument("--batch_size", "-batch_size", help="Batch size", type=int, default=128)
    parser.add_argument("--save_path", "-save_path", help="Save path", type=str, default="./tmp/")
    parser.add_argument("--var_threshold", "-var_threshold", help="Variance threshold", type=float,
                        default=1e-5)
    parser.add_argument("--num_classes", "-num_classes", help="The number of classes", type=int,
                        default=2)
    parser.add_argument("-r", "--regularized_kde", help="Regularize", type=bool, default=True)

    args = parser.parse_args()
    args.save_path = args.save_path + args.dataset + "/" + args.model + "/"
    dir = os.path.dirname(args.save_path)

    if not os.path.exists(dir):
        os.makedirs(dir)

    print(args)

    # load dataset and models
    model_path = "./models/" + args.model + ".h5"
    dataset_path = Path("./data/" + args.dataset)

    model = load_model(model_path)
    #model = keras.models.clone_model(tf_model)
    model.summary()

    layer_names = [layer.name for layer in model.layers]
    layer_names = layer_names[1:]

    x_train_path = dataset_path / "train/x.csv"
    y_train_path = dataset_path / "train/y.csv"

    x_val_path = dataset_path / "val/x.csv"
    y_val_path = dataset_path / "val/y.csv"

    x_test_path = dataset_path / "test/x.csv"
    y_test_path = dataset_path / "test/y.csv"

    x_train = np.genfromtxt(x_train_path, delimiter=',')
    y_train = np.loadtxt(str(y_train_path), dtype=int)
    print("Train:", x_train.shape, y_train.shape)

    x_valid = np.genfromtxt(x_val_path, delimiter=',', skip_header=1)
    y_valid = np.loadtxt(str(y_val_path), dtype=int)
    print("Val:", x_valid.shape, y_valid.shape)

    x_test = np.genfromtxt(x_test_path, delimiter=',', skip_header=1)
    y_test = np.loadtxt(str(y_test_path), dtype=int)
    print("Test:", x_test.shape, y_test.shape)

    # obtain kde functions and kde inferred classes per class
    if args.flag == "train":
        train_fetch_kdes(model, x_train, x_valid, y_train, y_valid, layer_names, args)
    else:
        test_fetch_kdes(model, x_test, y_test, layer_names, args)
