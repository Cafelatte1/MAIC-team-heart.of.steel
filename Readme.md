## Solution: MAIC ECG AI Challenge 2023 (Team 강철심장)

Please refer to source structure & code snipet for inference below

('torch' version depends on your environment)

### Source structure
```
root\
inference.py 
helper_functions.py
  ㄴ dataset\
      ㄴ submission.csv
      ㄴ ECG_adult_numpy_valid\
           ㄴ .npz ecg data
      ㄴ ECG_child_numpy_valid\
           ㄴ .npz ecg data
  ㄴ models\
      ㄴ dnn_rawWithkaggle_catboost_v2\
           ㄴ .pkl 5-fold models
      ㄴ dnn_rawWithkaggle_densenetLSTM_v2\
           ㄴ .pth 5-fold models
```
Models are also shared in link below

**[Link](https://drive.google.com/drive/folders/1In4K52ZNaSRYyr0nvmnpSSvacotlvj1r?usp=sharing)**

### Code snipet for inference
```
python ./inference.py
```
