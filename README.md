# Y-vector
Y-vector: Multiscale Waveform Encoder for Speaker Embedding
(Keep Updating)
Official inference code for Y-vector(https://arxiv.org/abs/2010.12951) and unofficial code for wav2spk (https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1287.pdf)

Will provide pretrained model (both Y-vector and wav2spk) and inference code soon.

In our experiment, we train on VoxCeleb2 Dev dataset, and test on VoxCeleb1 dataset.

Results (cosine score EER(%)):

| System         |VoxCeleb-O  | VoxCeleb-E  |VoxCeleb-H | 
|------------------|------------------|------------------|------------------|
| wav2spk       | 3.00             | 2.78              | 4.56             |
| Y-vector.      | 2.72              |   2.38            | 3.87             |
