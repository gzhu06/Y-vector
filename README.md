# Y-vector

## Y-vector: Multiscale Waveform Encoder for Speaker Embedding

In this paper, we use the modular architecture on raw waveform speaker embedding, to be specific: a waveform encoder and deep embedding backbone. 

Official inference code for Y-vector (https://arxiv.org/abs/2010.12951) and unofficial code for wav2spk (https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1287.pdf)

Will provide pretrained model (both Y-vector and wav2spk) and inference code soon.

In our experiment, we train on VoxCeleb2 Dev dataset, and test on VoxCeleb1 dataset.

## Results 
(Will upload pre-trained models soon and will clean inference code etc)

(cosine score EER(%)):
| System         |VoxCeleb1-O*  | VoxCeleb1-E  |VoxCeleb1-H | 
|------------------|------------------|------------------|------------------|
| wav2spk       | 3.00             | 2.78              | 4.56             |
| Y-vector.       | 2.72              |   2.38            | 3.87             |

(*Notice that VoxCeleb1-O can fluctuate a lot in our experimental setting)

It's possible to boost the performance by replacing each part with stronger networks. For example, replace backbone with F-TDNN, E-TDNN or ECAPA-TDNN.