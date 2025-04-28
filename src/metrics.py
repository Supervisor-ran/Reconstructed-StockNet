import torch
import torch.nn.functional as F
import numpy as np
import math


def n_accurate(y, y_):
    """
        y, y_: Tensor, shape: [batch_size, y_size];
    """
    correct_y_batch = torch.eq(torch.argmax(y, dim=1), torch.argmax(y_, dim=1))
    n_accurate = torch.sum(correct_y_batch).float()  # similar to numpy.count_nonzero()
    return n_accurate


def eval_acc(n_acc, total):
    return float(n_acc) / total


def create_confusion_matrix_new(y_pred, y_true, is_distribution=True):
    """
    计算二分类任务的混淆矩阵。

    参数:
    - y_pred: 模型预测值（概率分布或直接的类别标签）
    - y_true: 真实标签（概率分布或直接的类别标签）
    - is_distribution: 是否将预测值视为概率分布，如果为 True，会通过 argmax 转为标签

    返回:
    - tp: 真阳性数
    - fp: 假阳性数
    - tn: 真阴性数
    - fn: 假阴性数
    """
    if is_distribution:
        y_pred = torch.argmax(y_pred, dim=1)
        y_true = torch.argmax(y_true, dim=1)

    # 转为布尔张量
    tp = torch.sum((y_pred == 1) & (y_true == 1)).item()
    fp = torch.sum((y_pred == 1) & (y_true == 0)).item()
    tn = torch.sum((y_pred == 0) & (y_true == 0)).item()
    fn = torch.sum((y_pred == 0) & (y_true == 1)).item()

    return tp, fp, tn, fn


def create_confusion_matrix(y, y_, is_distribution=True):
    """
        By batch. shape: [n_batch, batch_size, y_size]
    """
    n_samples = float(y_.shape[0])  # get dimension list
    if is_distribution:
        label_ref = torch.argmax(y_, dim=1)  # 1-d array of 0 and 1
        label_hyp = torch.argmax(y, dim=1)
    else:
        label_ref, label_hyp = y, y_

    # p & n in prediction
    p_in_hyp = torch.sum(label_hyp)
    n_in_hyp = n_samples - p_in_hyp

    # Positive class: up
    tp = torch.sum(label_ref * label_hyp)  # element-wise, both 1 can remain
    fp = p_in_hyp - tp  # predicted positive, but false

    # Negative class: down
    tn = n_samples - torch.count_nonzero(label_ref + label_hyp)  # both 0 can remain
    fn = n_in_hyp - tn  # predicted negative, but false



    return float(tp), float(fp), float(tn), float(fn)


def eval_mcc(tp, fp, tn, fn):
    if any(x < 0 for x in [tp, fp, tn, fn]):
        raise ValueError("tp, fp, tn, and fn must all be non-negative")
    core_de = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return (tp * tn - fp * fn) / math.sqrt(core_de) if core_de else None


def eval_res(gen_n_acc, gen_size, gen_loss_list, y_list, y_list_, use_mcc=None):
    gen_acc = eval_acc(n_acc=gen_n_acc, total=gen_size)
    gen_loss = np.average(gen_loss_list)
    results = {'loss': gen_loss,
               'acc': gen_acc,
               }

    if use_mcc:
        gen_y = torch.cat(y_list, dim=0)
        gen_y_ = torch.cat(y_list_, dim=0)
        tp, fp, tn, fn = create_confusion_matrix_new(gen_y, gen_y_)
        results['mcc'] = eval_mcc(tp, fp, tn, fn)

    return results


def basic_train_stat(train_batch_loss_list, train_epoch_n_acc, train_epoch_size):
    train_epoch_loss = np.average(train_batch_loss_list)
    train_epoch_acc = eval_acc(n_acc=train_epoch_n_acc, total=train_epoch_size)
    return train_epoch_loss, train_epoch_acc
