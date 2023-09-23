## Solution: MAIC ECG AI Challenge 2023 (Team 강철심장)

Please refer to source structure & code snipet for inference below

('torch' version depends on your environment)

### Source structure
```
root
  ㄴ dataset\
      ㄴ submission.csv
      ㄴ ECG_adult_numpy_valid\
           ㄴ npz files
      ㄴ ECG_child_numpy_valid\
           ㄴ npz files
  ㄴ models\
      ㄴ dnn_rawWithkaggle_catboost_v2\
           ㄴ 5-fold models
      ㄴ dnn_rawWithkaggle_densenetLSTM_v2\
           ㄴ 5-fold models
```
models are shared in link below

**[Link]**(https://drive.google.com/drive/folders/1In4K52ZNaSRYyr0nvmnpSSvacotlvj1r?usp=sharing)

### Code snipet for inference
```
python ./inference.py
```
