import json
import logging
import torch
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import shutil
from torch.backends import cudnn
import time
import random
from numpy import random
from scipy.optimize import linear_sum_assignment
from lib.config.para import parse_opt
opt = parse_opt()

labels_name = ["Attractive","BeatCompete","Communicate","CreativeUnique","CuriousAdventurousExcitingLife","EasyLife","EnjoyLife","FineDesignLearnArt-Arch","FineDesignLearnArt-Art",
              "FineDesignLearnArt-Culture","GoodParentEmoCloseChild","Happy","HardWorking","Harmony","Health","InLove","InLoveAnimal","InspirOthrs","ManagableMakePlan",
              "NatBeauty","PassionAbSmthing","Playful","ShareFeelings","SocialLifeFriendship","SuccInOccupHavGdJob", "TchOthrs","ThngsInOrdr", "WorkILike"]


def fix_randomness(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # cudnn related setting
    cudnn.benchmark = False
    cudnn.deterministic = True
    cudnn.enabled = True


def read_ann_json(file):
    with open(file, 'r') as f:
        datas = json.load(f)
    annotations = datas['annotations']

    return annotations


#!/usr/bin/env python3
"""
evalulation
"""
import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics import f1_score
SUBSET2category_idS = {
    'easy': [0, 7, 19],
    'medium': [1, 3, 4, 6, 9, 10, 11, 12, 13, 14, 15, 16, 18, 22, 26],
    'hard': [2, 5, 8, 17, 20, 21, 23, 24, 25, 27],
    'object': [0, 3, 10, 11, 12, 16, 23],
    'context': [7, 8],
    "other": [1, 2, 4, 5, 6, 9, 13, 17, 18, 19, 20, 22, 25, 27, 14, 15, 21,  24, 26],
}


def eval_all_metrics(
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    val_targets: List[List[int]],
    test_targets: List[List[int]]
) -> dict:
    """
    compute valcategory_idation and test results
    args:
        val_scores: np.ndarray of shape (val_num, num_classes),
        test_scores: np.ndarray of shape (test_num, num_classes),
        val_targets: List[List[int]],
        test_targets: List[List[int]]
    """
    # get optimal threshold using val set
    multihot_targets = multihot(val_targets, 28)
    f1_dict = get_best_f1_scores(multihot_targets, val_scores)

    # get results using the threshold found
    multihot_targets = multihot(test_targets, 28)
    test_micro, test_samples, test_macro, test_none = compute_f1(multihot_targets, test_scores, f1_dict["threshold"])
    return {
        "val_micro": f1_dict["micro"], "val_samples": f1_dict["samples"],
        "val_macro": f1_dict["macro"], "val_none": f1_dict["none"],
        "test_micro": test_micro, "test_samples": test_samples,
        "test_macro": test_macro, "test_none": test_none,
    }


def get_best_f1_scores(
    multihot_targets: np.ndarray,
    scores: np.ndarray,
    threshold_end: float = 0.05
) -> Dict[str, float]:
    """
    get the optimal macro f1 score by tuning threshold
    """
    # thrs.size = 19
    thrs = np.linspace(
        threshold_end, 0.95, int(np.round((0.95 - threshold_end) / 0.05)) + 1,
        endpoint=True
    )
    f1_micros = []
    f1_macros = []
    f1_samples = []
    f1_none = []
    for thr in thrs:
        _micros, _samples, _macros, _none = compute_f1(multihot_targets, scores, thr)
        f1_micros.append(_micros)
        f1_samples.append(_samples)
        f1_macros.append(_macros)
        f1_none.append(_none)

    f1_macros_m = max(f1_macros)
    b_thr = np.argmax(f1_macros)

    f1_micros_m = f1_micros[b_thr]
    f1_samples_m = f1_samples[b_thr]
    f1_none_m = f1_none[b_thr]
    f1 = {}
    f1["micro"] = f1_micros_m
    f1["macro"] = f1_macros_m
    f1["samples"] = f1_samples_m
    f1["threshold"] = thrs[b_thr]
    f1["none"] = f1_none_m
    return f1


def compute_f1(
        multihot_targets: np.ndarray, scores: np.ndarray, threshold: float = 0.5
) -> Tuple[float, float, float]:
    # change scores to predict_labels
    predict_labels = scores > threshold
    predict_labels = predict_labels.astype(np.int)

    # get f1 scores
    f1 = {}
    f1["micro"] = f1_score(
        y_true=multihot_targets,
        y_pred=predict_labels,
        average="micro"
    )
    f1["samples"] = f1_score(
        y_true=multihot_targets,
        y_pred=predict_labels,
        average="samples"
    )
    f1["macro"] = f1_score(
        y_true=multihot_targets,
        y_pred=predict_labels,
        average="macro"
    )
    f1["none"] = f1_score(
        y_true=multihot_targets,
        y_pred=predict_labels,
        average=None
    )
    return f1["micro"], f1["samples"], f1["macro"], f1["none"]


def multihot(x: List[List[int]], nb_classes: int) -> np.ndarray:
    """transform to multihot encoding

    Arguments:
        x: list of multi-class integer labels, in the range
            [0, nb_classes-1]
        nb_classes: number of classes for the multi-hot vector

    Returns:
        multihot: multihot vector of type int, (num_samples, nb_classes)
    """
    num_samples = len(x)
    # make a multihot which is stroe val/test  category  to  len(val/test) * 28
    # if is be labe , we set it to 1 else 0
    multihot = np.zeros((num_samples, nb_classes), dtype=np.int32)
    for category_idx, labs in enumerate(x):
        for lab in labs:
            multihot[category_idx, lab] = 1

    return multihot.astype(np.int)


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# to find the best component
def eval_center_number(data):
    plt.figure(1)
    data = np.array(data)
    SSE = []
    for k in range(1, 9):
        estimator = KMeans(n_clusters=k)
        estimator.fit(data)
        SSE.append(estimator.inertia_)
    X = range(1, 9)
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.plot(X, SSE, 'o-')
    plt.figure(2)
    Scores = []
    for k in range(2,9):
        estimator = KMeans(n_clusters=k)
        estimator.fit(data)
        Scores.append(silhouette_score(data,estimator.labels_,metric='euclidean'))
    X = range(2,9)
    plt.xlabel('k')
    plt.ylabel('rate')
    plt.plot(X,Scores,'o-')
    plt.show()


def ele_sum(before, now, component):
    c1 = get_center(before, component=component)
    c2 = get_center(now, component=component)
    part1 = c1.chunk(opt.Intent_class, dim=0)
    part2 = c2.chunk(opt.Intent_class, dim=0)
    for j, (cen1, cen2) in enumerate(zip(part1,part2)):
        cd = cosine(cen1, cen2)
        row_ind, col_ind = linear_sum_assignment(cd.detach())
        for i in range(component):
            cent1 = (opt.old_rate * cen1[row_ind[i]] + (1 - opt.old_rate) * cen2[col_ind[i]]).view(1, -1)
            if j == 0 and i==0:
                cen = torch.cat([cent1], dim=0)
            else:
                cen = torch.cat([cen,cent1], dim=0)
    return cen


def del_tensor_ele(arr, index):
    arr1 = arr[0:index]
    arr2 = arr[index + 1:]
    return torch.cat((arr1, arr2), dim=0)


def update_center_by_dloss(prototype,cen):
    for i,tp in enumerate(prototype):
        for j in range(opt.component):
            tp[j] = cen[i*opt.component + j]
    return prototype


def cosine(target,pre):
    if len(target.shape) == 1:
        target = target.view(1,-1)
    if len(pre.shape) == 1:
        pre = pre.view(1,-1)
    if isinstance(target,type(pre)):
        pre = pre.type_as(target)
    prod = torch.mm(target, pre.t())
    norm1 = torch.norm(target, p=2, dim=1).unsqueeze(0)
    norm2 = torch.norm(pre, p=2, dim=1).unsqueeze(0)
    cos = prod.div(torch.mm(norm1.t(),norm2))
    return cos


def list_tensor(l):
    """
    stack
    Concatenates a sequence of tensors along a new dimension.
    """
    final_tensor = torch.stack(l, 0)
    return final_tensor


def get_center(prototype,component):
    c = []
    for i in prototype:
        c.append(i)
    c = list_tensor(c)
    c = c.view(component*opt.Intent_class,-1)
    return c


def get_att_dis(target, behaviored):
    behaviored = behaviored.type_as(target)
    target = target.view(1,-1)
    cosinesimilarity = torch.nn.CosineSimilarity(dim=1)  # Calculate the cosine similarity of each element to the given element
    attention_score = cosinesimilarity(target,behaviored.view(1,-1))
    return attention_score

if __name__ == '__main__':
    c1 = torch.randn([112,2048])
    c2 = torch.randn([112,2048])
    c = ele_sum(c1,c2,4)