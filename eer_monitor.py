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

def trial_eer(embed_model, trials, ipt_shape, device, seg_len=3.0):
    from trainer.train_utils import clipped_feature

    feed_batchsize = ipt_shape[0]
    loop_num = int(len(trials) / feed_batchsize)

    results = np.zeros(len(trials))
    sims = np.zeros(len(trials))

    for i in range(loop_num):
        trials_temp = trials[i * feed_batchsize: (i + 1) * feed_batchsize]
        sims_temp = []
        results_temp = np.zeros(feed_batchsize)

        # eer monitor
        anchor_batch = []
        pair_batch = []

        for j in range(feed_batchsize):
            anchor, pair, result = trials_temp[j]

            anchor_feature = np.load(anchor, allow_pickle=True)
            pair_feature = np.load(pair, allow_pickle=True)

            anchor_temp = clipped_feature(np.expand_dims(anchor_feature, 0), seg_len)
            pair_temp = clipped_feature(np.expand_dims(pair_feature, 0), seg_len)
            results_temp[j] = result

            anchor_batch.append(anchor_temp)
            pair_batch.append(pair_temp)

        anchor_batch = np.array(anchor_batch)
        pair_batch = np.array(pair_batch)

        with torch.no_grad():
            
            if torch.cuda.device_count() > 1:
#                 anchor_embeds, _, _ = embed_model(torch.from_numpy(anchor_batch).float().to(device))
#                 pair_embeds, _, _ = embed_model(torch.from_numpy(pair_batch).float().to(device))
                
                anchor_embeds = embed_model(torch.from_numpy(anchor_batch).float().to(device))
                pair_embeds = embed_model(torch.from_numpy(pair_batch).float().to(device))
                
            else:
                anchor_embeds = embed_model(torch.from_numpy(anchor_batch).float().to(device))
                pair_embeds = embed_model(torch.from_numpy(pair_batch).float().to(device))

            sims_temp = cosine_similarity(anchor_embeds.cpu().numpy(), pair_embeds.cpu().numpy())

            sims[i * feed_batchsize: (i + 1) * feed_batchsize] = sims_temp
            results[i * feed_batchsize: (i + 1) * feed_batchsize] = results_temp

    P_idx = np.where(results == 1)[0]
    N_idx = np.where(results == 0)[0]

    positive_similarity = sims[P_idx]
    negative_similarity = sims[N_idx]

    intra_distance = np.sqrt(2.0 - 2.0 * positive_similarity + 1e-6)
    inter_distance = np.sqrt(2.0 - 2.0 * negative_similarity + 1e-6)

    intra_distance = np.expand_dims(intra_distance, axis=0)
    inter_distance = np.expand_dims(inter_distance, axis=0)

    eer, threshold = calculate_eer(positive_similarity, negative_similarity)

    print("threshold is --> ", threshold, "eer is --> ", eer)

    return eer, threshold, intra_distance, inter_distance

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