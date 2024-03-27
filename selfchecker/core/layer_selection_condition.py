import json
import numpy as np
import itertools

from pathlib import Path


def calculate_accuracy(num_classes, layers, pred_label_idx, pred_labels):
    kde_preds = np.zeros([pred_label_idx.shape[0], num_classes])
    count_idx = 0

    for idx in pred_label_idx:
        for layer_idx in layers:
            kde_preds[count_idx][int(pred_labels[idx][layer_idx])] += 1
        count_idx += 1

    kde_pred = np.argmax(kde_preds, axis=1)
    kde_accuracy = np.mean(kde_pred == pred_labels[pred_label_idx].T[-1])

    return kde_accuracy


def selected_layer_condition(num_classes, num_layers, idx_in_origin, pred_labels):
    max_acc = 0
    selected_layers = None
    total_layers = [x for x in range(num_layers)]

    for count in range(1, num_layers+1):
        for layers in itertools.combinations(total_layers, count):
            acc = calculate_accuracy(num_classes, layers, idx_in_origin, pred_labels)

            if acc >= max_acc:
                max_acc = acc
                selected_layers = layers

    kde_acc = calculate_accuracy(num_classes, selected_layers, idx_in_origin, pred_labels)
    model_acc = np.mean(pred_labels[idx_in_origin].T[-2] == pred_labels[idx_in_origin].T[-1])

    print("selected layers: {}, acc: {}".format(selected_layers, kde_acc))
    print("model acc: {}\n".format(model_acc))

    return selected_layers, kde_acc


def selected_layer_for_label(num_layers, num_classes, layers_agree, label, selected_layers_dict, weights_dict,
                             working_dir: Path):
    # split dataset into subset according to their predictions
    pred_labels = np.load(working_dir / "pred_labels_valid.npy")
    pred_label_idx = np.where(pred_labels.T[-2] == label)[0]

    # count the number of layers that agree with the final prediction
    num_selected_layers = len(layers_agree[str(label)])
    pred_count = np.zeros([pred_labels[pred_label_idx].shape[0], num_selected_layers])
    misbh_count = np.zeros([pred_labels[pred_label_idx].shape[0], num_selected_layers])

    for idx in range(pred_labels[pred_label_idx].shape[0]):
        count_idx = 0
        for layer_idx in layers_agree[str(label)]:
            if pred_labels[pred_label_idx[idx]][layer_idx] == pred_labels[pred_label_idx[idx]][-2]:
                pred_count[idx][count_idx] = 1
            if pred_labels[pred_label_idx[idx]][layer_idx] != pred_labels[pred_label_idx[idx]][-2]:
                misbh_count[idx][count_idx] = 1
            count_idx += 1

    # calculate confidence
    sum_pred_example = np.sum(pred_count, axis=1)
    sum_misbh_example = np.sum(misbh_count, axis=1)
    pos_indexes = np.where(sum_misbh_example >= sum_pred_example)[0]

    kde_pred_positive = sum_misbh_example >= sum_pred_example
    true_mis_behaviour = pred_labels[pred_label_idx].T[-2] != pred_labels[pred_label_idx].T[-1]

    false_positives = np.sum(~true_mis_behaviour & kde_pred_positive)

    # searches for the best layer combination where the model predicts the input with label 'label_con' as 'label'
    for label_con in range(num_classes):
        pos_indexes_label = np.where(pred_labels[pred_label_idx[pos_indexes]].T[-1] == label_con)[0]
        print("label: {}, total_len: {}, label_con: {}, len: {}".format(label, pos_indexes.shape[0], label_con,
                                                                        pos_indexes_label.shape[0]))
        if pos_indexes_label.shape[0] == 0:
            print("check!")
            continue

        selected_layer_name = str(label) + str(label_con)
        idx_in_origin = pred_label_idx[pos_indexes[pos_indexes_label]]
        selected_layers_dict[selected_layer_name], kde_acc = selected_layer_condition(num_classes, num_layers,
                                                                                      idx_in_origin, pred_labels)

        if label_con == label:
            weights_dict[selected_layer_name] = pos_indexes_label.shape[0] * kde_acc / pos_indexes.shape[0]
        else:
            denominator = pos_indexes.shape[0] - false_positives
            weights_dict[selected_layer_name] = pos_indexes_label.shape[0] * kde_acc / denominator


def layer_selection_condition(num_layers, labels: list, working_dir: Path):
    num_classes = len(labels)

    for label in labels:
        print("label: {}".format(label))
        selected_layers_dict = {}
        weights_dict = {}

        # load selected layers for alarm
        with (working_dir / f"selected_layers_agree_{label}.json").open(mode="r") as json_file:
            layers_agree = json.load(json_file)

        # generate selected layers per class
        selected_layer_for_label(num_layers, num_classes, layers_agree, label, selected_layers_dict, weights_dict,
                                 working_dir)

        # save the index of selected layers per class
        with (working_dir / f"selected_layers_accuracy_{label}.json").open(mode='w') as json_file:
            json.dump(selected_layers_dict, json_file, ensure_ascii=False)

        with (working_dir / f"weights_{label}.json").open(mode='w') as json_file:
            json.dump(weights_dict, json_file, ensure_ascii=False)
