import json
import itertools
import numpy as np

from pathlib import Path


def calculate_f1(layers, pred_labels, pred_label_idx):
    # count the number of layers that agree with the final prediction
    num_selected_layers = len(layers)
    pred_count = np.zeros([pred_labels[pred_label_idx].shape[0], num_selected_layers])
    misbh_count = np.zeros([pred_labels[pred_label_idx].shape[0], num_selected_layers])

    for idx in range(pred_labels[pred_label_idx].shape[0]):
        count_idx = 0

        for layer_idx in layers:

            if pred_labels[pred_label_idx[idx]][layer_idx] == pred_labels[pred_label_idx[idx]][-2]:
                pred_count[idx][count_idx] = 1

            if pred_labels[pred_label_idx[idx]][layer_idx] != pred_labels[pred_label_idx[idx]][-2]:
                misbh_count[idx][count_idx] = 1
            count_idx += 1

    # calculate confidence
    sum_pred_example = np.sum(pred_count, axis=1)
    sum_misbh_example = np.sum(misbh_count, axis=1)
    kde_pred_positive = sum_misbh_example >= sum_pred_example
    true_mis_behaviour = pred_labels[pred_label_idx].T[-2] != pred_labels[pred_label_idx].T[-1]

    # calculate confusion metric
    tp = np.sum(true_mis_behaviour & kde_pred_positive)
    fp = np.sum(~true_mis_behaviour & kde_pred_positive)
    tn = np.sum(~true_mis_behaviour & ~kde_pred_positive)
    fn = np.sum(true_mis_behaviour & ~kde_pred_positive)

    tpr = tp / (tp + fn) if tp + fn != 0 else 0
    fpr = fp / (tn + fp) if tn + fp != 0 else 0
    f1 = 2 * tp / (2 * tp + fn + fp) if tp + fn + fp != 0 else 0

    return tpr, fpr, f1, tp, fp, fn, tn


def selected_layer_for_label(num_layers, label, selected_layers_dict: dict, working_dir: Path):
    max_f1 = 0
    selected_layers = None
    # split dataset into subset according to their predictions
    pred_labels = np.load(working_dir / "pred_labels_valid.npy")
    pred_label_idx = np.where(pred_labels.T[-2] == label)[0]
    total_layers = [x for x in range(num_layers)]

    for count in range(1, num_layers+1):
        print("count: {}".format(count))

        for layers in itertools.combinations(total_layers, count):
            tpr, fpr, f1, tp, fp, fn, tn = calculate_f1(layers, pred_labels, pred_label_idx)

            if f1 >= max_f1:
                max_f1 = f1
                selected_layers = layers

    selected_layers_dict[str(label)] = selected_layers
    print("selected layers: {}".format(selected_layers))
    tpr, fpr, f1, tp, fp, fn, tn = calculate_f1(selected_layers, pred_labels, pred_label_idx)
    print("TPR:{:.6f} FPR:{:.6f} F1:{:.6f} TP:{} FP:{} FN:{} TN:{}".format(tpr, fpr, f1, tp, fp, fn, tn))


def layer_selection_agree(num_layers: int, labels: list, working_dir: Path):
    for label in labels:
        print("label: {}".format(label))
        selected_layers_dict = {}

        # generate selected layers per class
        selected_layer_for_label(num_layers, label, selected_layers_dict, working_dir)

        # save the index of selected layers per class
        filename = working_dir / f"selected_layers_agree_{label}.json"

        with filename.open(mode='w') as json_file:
            json.dump(selected_layers_dict, json_file, ensure_ascii=False)
