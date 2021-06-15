import sys
import os, fnmatch
sys.path.append(os.getcwd())

import numpy as np
from tqdm import tqdm
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import importlib
import glob
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F

from eer_monitor import calculate_eer, calculate_minDCF
import feerci

import utils
import plot_utils

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-config_file", type=str, default='', help="config file")
args = parser.parse_args()
config = utils.Params(args.config_file)

def load_model(folder, model_idx):
    # model definition
    model_module = importlib.import_module('backbones.' + config.backbone)
    model = model_module.architecture().cuda()
    
    from apex import amp
    model, _ = amp.initialize(model, [], opt_level="O1") # Try if you want to boost up synthesis speed.
    
    if len(config.gpu_idx) > 1:
        model = nn.DataParallel(model)
    checkpoint = torch.load(os.path.join(folder, 'model_' + model_idx + '.pt'), map_location='cpu')
    
#     model.load_state_dict(checkpoint['embed_state_dict'])
    
    # wav2spk load
    model.load_state_dict(torch.load(os.path.join(folder, 'model_' + model_idx + '.pt')))

    return model

def compute_embeddings(embed_model, device, model_idx, seg_len=2, root=config.list):
    embed_model.eval()
    files = glob.glob(config.test_o + "/*/*.pkl")
        
    embedding_dict = {}
    
    with torch.no_grad():
        for file in tqdm(files, desc='Computing embeddings'):
            
            utt = utils.pickle2array(file)
            key = file.split('/')[-1]
            
            utt = np.expand_dims(utt, axis=0)
            utt_embedding = embed_model(torch.from_numpy(utt).float().unsqueeze(0).to(device))
            embedding_dict.update({key: utt_embedding.cpu().squeeze(0)})
    
    np.save(os.path.join(config.train_path, 'test_embeds_' + model_idx + '.npy'), embedding_dict)
    
    return embedding_dict

def test_trials_eval(embedding_dict, trials=os.path.join(config.list, 'list_test_hard2.txt')):

    root = config.list
    with open(trials, 'r') as f:
        data = f.readlines()
        positive_similarity = []
        negative_similarity = []
        for line in tqdm(data, desc='Computing cosine similarities'):
            content = line.split(' ')
            
            file1 = content[1].split('.')[0]
            file2 = content[2][:-1].split('.')[0]
            key1 = file1.split('/')[0] + '-' + file1.split('/')[1] + '-' + file1.split('/')[2] + '.pkl'
            key2 = file2.split('/')[0] + '-' + file2.split('/')[1] + '-' + file2.split('/')[2] + '.pkl'

            anchor_embeds = embedding_dict[key1]
            pair_embeds = embedding_dict[key2]
            sims_temp = F.cosine_similarity(anchor_embeds, pair_embeds, dim=0).numpy()
            if content[0] == '1':
                positive_similarity.append(sims_temp)
            if content[0] == '0':
                negative_similarity.append(sims_temp)
    
    # minDCF
    total_scores = positive_similarity + negative_similarity
    total_results = [1] * len(positive_similarity) + [0] * len(negative_similarity)
    
    min_dcf2, min_c_det_threshold2 = calculate_minDCF(total_scores, total_results, 0.01, 1, 1)
    min_dcf3, min_c_det_threshold3 = calculate_minDCF(total_scores, total_results, 0.001, 1, 1)
    
    print('minDCF:0.01 {0:0.4f},{1:0.4f}'.format(min_dcf2, min_c_det_threshold2))
    print('minDCF:0.001 :{0:0.4f},{1:0.4f}'.format(min_dcf3, min_c_det_threshold3))
    
    # eer
    positive_similarity = np.array(positive_similarity)
    negative_similarity = np.array(negative_similarity)

    eer, threshold = calculate_eer(positive_similarity, negative_similarity)
    
    print("threshold is --> {:.4f}".format(threshold), "eer is --> {:.4f}%".format(eer*100.0))
    
    # confidence interval
    eer_ci, ci_lower,ci_upper,bootstrapped_eers = feerci.feerci(negative_similarity.astype('float64'), positive_similarity.astype('float64'), is_sorted=False)
    
    print('confidence interval is:{0:0.4f},{1:0.4f},eer_ci:{2:0.4f}'.format(ci_lower, ci_upper, eer_ci))

if __name__ == '__main__':
    
    model_idx = '295'
    short_duration = False
    
    # get eval file embeddings
    if not os.path.exists(os.path.join(config.train_path, 'test_embeds_' + model_idx + '.npy')):
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_idx
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # load models
        embed_model = load_model(config.train_path, model_idx).to(device)
        
        embedding_dict = compute_embeddings(embed_model, device, model_idx, TTA=False, short_duration=short_duration)
    else:
        embedding_dict = np.load(os.path.join(config.train_path, 'test_embeds_' + model_idx + '.npy'), allow_pickle=True).item()
    
    print('Vox1-O')
    test_trials_eval(embedding_dict, trials=os.path.join(config.list, 'veri_test2.txt'))
    print('Vox1-H')
    test_trials_eval(embedding_dict)
    print('Vox1-E')
    test_trials_eval(embedding_dict, trials=os.path.join(config.list, 'list_test_all2.txt'))
    