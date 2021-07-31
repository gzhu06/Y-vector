# reference wav2vec from Facebook: 
# https://github.com/Walleclipse/Deep_Speaker-speaker_recognition_system/blob/master/pre_process.py

import utils
from utils import pickle2array
from glob import glob
import os
import librosa
import librosa.display
import numpy as np
import pandas as pd 
from multiprocessing import Pool
from tqdm import tqdm
from time import time

SR = 16000
SEG_LEN = 3.9

def find_files(directory, pattern='**/*.wav'):
    print(os.path.join(directory, pattern))
    return glob(os.path.join(directory, pattern), recursive=True)


def read_audio(filename, sample_rate=16000):
    # import soundfile as sf
    # audio, sr = sf.read(filename, samplerate=sample_rate)
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    audio = librosa.util.normalize(audio)
    return audio

def features(libri, out_dir, name='0'):

    for i in tqdm(range(len(libri))):
        filename = libri[i:i+1]['filename'].values[0]        
        
        target_filename_partial = os.path.join(out_dir, filename.split("/")[-3] + '-' + filename.split("/")[-2] + '-' + filename.split("/")[-1].split('.')[0])  #clean
        try:
            raw_audio = read_audio(filename)
        except:
            print(filename, 'file error!')
            continue
        
        sample_num = int(SEG_LEN * SR)
        
        if raw_audio.shape[0] < sample_num:
            print(raw_audio.shape[0])
            print('there is an error in file:',filename)
            continue
        else:
            target_filename = target_filename_partial + '.pkl'
            utils.array2pickle(raw_audio, target_filename)

def preprocess_and_save(wav_dir, out_dir):

    libri = data_catalog(wav_dir, pattern='**/*.wav') 

    print("extract fbank from audio and save as pickle, using multiprocessing pool........ ")
    p = Pool(5)
    patch = int(len(libri)/5)
    for i in range(5):
        if i < 4:
            slibri = libri[i * patch: (i+1) * patch]
        else:
            slibri = libri[i * patch:]
        print("task %s slibri length: %d" %(i, len(slibri)))
        p.apply_async(features, args=(slibri, out_dir, i))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()

def data_catalog(dataset_dir, pattern='*.pkl'):
    libri = pd.DataFrame()
    libri['filename'] = find_files(dataset_dir, pattern=pattern)
    if pattern == '**/*.wav':
        libri['speaker_id'] = libri['filename'].apply(lambda x: x.split('/')[-3])
        # libri['speaker_id'] = libri['filename'].apply(lambda x: x.split('/')[-2]) #(test)
    else:
        libri['speaker_id'] = libri['filename'].apply(lambda x: x.split('/')[-1].split('-')[0])

    return libri

def data2csv(data_folder, subsets=['val', 'train'], data_suffix='/*.pkl'):
    for oneset in subsets:
        subset_folder = os.path.join(data_folder, oneset)
        spk_ids = []
        frame_num = []
        file_names = []
        files = glob(subset_folder + data_suffix)
        for f in tqdm(files, desc='Parsing {} set'.format(oneset)):
            tf_data = pickle2array(f)
            f_name = f.split('/')[-1]
            file_names.append(f_name)
            spk_ids.append(f_name.split('-')[0])
            frame_num.append(tf_data.shape[-1])
        dataframe = pd.DataFrame({'file_name': file_names, 'spk_id': spk_ids, 'frame_num': frame_num})
        dataframe.to_csv(os.path.join(subset_folder, "metadata.csv".format(oneset)), index=False)

if __name__ == '__main__':
    # test
#     test()
    wavfile_folder = '/Users/gzhu/Documents/Speech/dataset/voxceleb/test_wav' # wav file folder path
    pklfile_folder = '/Users/gzhu/Documents/Speech/dataset/voxceleb/test_pkl' # pkl file folder path
    preprocess_and_save(wavfile_folder, pklfile_folder)
