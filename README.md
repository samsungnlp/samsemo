# SAMSEMO: New dataset for multilingual and multimodal emotion recognition
[![](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![CC BY 4.0][cc-by-shield]][cc-by]


<img align="right" src="img/HKUST.jpg" width="15%"/>

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

The repository contains the data collection and code for a new article that will be presented at the Interspeech 2024 conference: https://interspeech2024.org/.

We present training code for multilingual and multimodal emotion recognition. This code is based mainly on the code from https://github.com/wenliangdai/Multimodal-End2end-Sparse which was released under CC BY 4.0 license. The training consists of two phases. In the preprocessing phase, we extract faces from video frames, spectrograms from audio, and text features from text. In the end-to-end approach proposed in https://github.com/wenliangdai/Multimodal-End2end-Sparse, this preprocessing was performed each time a data item appeared in a training batch. Our approach considerably speeds up the training process.

[[Here]](https://huggingface.co/datasets/SamsungNLP/SAMSEMO/tree/main/data) you can find both the raw data and already preprocessed data in the pickle files.

### Training without preprocessing

To run our code directly, you can download the processed data from [[here]](https://huggingface.co/datasets/SamsungNLP/SAMSEMO/tree/main/data/pkl_files). Make sure to change appropriately the "data_path" and "dataset_name" parameters in the run.py file or in terminal.

## Training with preprocessing
To run the code with the preprocessing phase, you can download the raw data from the .zip files [[here]](https://huggingface.co/datasets/SamsungNLP/SAMSEMO/tree/main/data). Please update appropriately the parameters given in the run.py file. Below you will find a command to run the training with preprocessing phase.
You may also perform only the preprocessing phase to obtain the pickle files by running (with appropriate parameters):
```
python3 preprocessing/prep_raw.py
```

## Command examples for running
Example command for run from preprocessed data:
```
python3 run.py --train_from_raw_data ""
--data_path [dir_with_pickle_file]
--dataset_name samsemo_en_article.pkl
```


Example command for run from raw data:
```
python3 lightning_train.py --train_from_raw_data True
--base_dir [directory_with_raw_files]
--utt_names_path [dir_with_utterances_(text)]
--utt_file _split_EN.txt
--meta_path [dir_with_meta_file]
--files_path [dir_with_audio_and_frames]
--dataset_name samsemo_en_article.pkl
```


## Results
During the training, you will see the tables with the results for the training and validation sets. We provide several metrics: accuracy, f1, recall, ROC/AUC and precision. After training, we evaluate the best model on the test set and print the most important metric, which is average f1 (average is taken over f1 for all emotions).

