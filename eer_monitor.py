import sys
import os
sys.path.append(os.getcwd())

import torch
from numpy.linalg import norm
import numpy as np
from operator import itemgetter

def cosine_similarity(embed1, embed2):
    norm1 = norm(embed1, axis=-1)
    norm2 = norm(embed2, axis=-1)
    de_norm = norm1 * norm2

    no_mult = np.sum(np.multiply(embed1, embed2), axis=-1)
    s = np.true_divide(no_mult, de_norm)

    return s

def calculate_eer(positive_sim, negative_sim):
    target_scores = sorted(positive_sim)
    nontarget_scores = sorted(negative_sim)

    target_size = len(target_scores)
    nontarget_size = len(nontarget_scores)

    target_position = 0
    for target_position in range(target_size):
        nontarget_n = nontarget_size * target_position * 1.0 / target_size
        nontarget_position = int(nontarget_size - 1 - nontarget_n)
        if nontarget_position < 0:
            nontarget_position = 0
        if nontarget_scores[nontarget_position] < target_scores[target_position]:
            break

    threshold = target_scores[target_position]
    eer = target_position * 1.0 / target_size

    return eer, threshold

def ComputeErrorRates(scores, labels):
    # Sort the scores from smallest to largest, and also get the corresponding
    # indexes of the sorted scores.  We will treat the sorted scores as the
    # thresholds at which the the error-rates are evaluated.
    sorted_indexes, thresholds = zip(*sorted(
        [(index, threshold) for index, threshold in enumerate(scores)],
        key=itemgetter(1)))
    sorted_labels = []
    labels = [labels[i] for i in sorted_indexes]
    fnrs = []
    fprs = []

    # At the end of this loop, fnrs[i] is the number of errors made by
    # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
    # is the total number of times that we have correctly accepted scores
    # greater than thresholds[i].
    for i in range(0, len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1 - labels[i])
        else:
            fnrs.append(fnrs[i-1] + labels[i])
            fprs.append(fprs[i-1] + 1 - labels[i])
    fnrs_norm = sum(labels)
    fprs_norm = len(labels) - fnrs_norm

    # Now divide by the total number of false negative errors to
    # obtain the false positive rates across all thresholds
    fnrs = [x / float(fnrs_norm) for x in fnrs]

    # Divide by the total number of corret positives to get the
    # true positive rate.  Subtract these quantities from 1 to
    # get the false positive rates.
    fprs = [1 - x / float(fprs_norm) for x in fprs]
    return fnrs, fprs, thresholds

# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold

def calculate_minDCF(scores, labels, p_target, c_miss, c_fa):

    fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
    min_dcf, min_c_det_threshold = ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa)
    
    return min_dcf, min_c_det_threshold