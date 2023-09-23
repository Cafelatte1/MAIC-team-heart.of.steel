GLOBAL_SEED = 42

import os
os.environ['PYTHONHASHSEED'] = str(GLOBAL_SEED)

import numpy as np
from numpy import random as np_rnd
import pandas as pd
from tqdm import tqdm
import random as rnd
import pickle
import gc
import time
from itertools import product

from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import torch.nn.functional as F
import catboost as cat
from sklearn import metrics as skl_metrics

import warnings
warnings.filterwarnings(action='ignore')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from helper_functions import *


class CFG:
    debug = False
    dataset_root_path = "./dataset/"
    dataset_version = "v5"
    model_root_path = "./models/"
    n_comp = 64
    TF = False
    TF_type = "indiv"
    use_ecg_stats = False
    use_ecg_seq = True
    pca_seq = False
    n_folds = 1 if debug else 5
    lead_names = ["std1", "std2", "std3", "v1", "v2", "v3", "v4", "v5", "v6", "augvr", "augvl", "augvf"]
    ecg_hz = 500

    epochs = 2 if debug else 30
    early_stopping_rounds = 10
    batch_size = 32
    eta = 5e-4
    weight_decay = 1e-4

model_params = pickleIO(None, CFG.model_root_path + "dnn_model_params.pkl", "r")
fixed_params = pickleIO(None, CFG.model_root_path + "catboost_fixed_params.pkl", "r")
dynamic_params = pickleIO(None, CFG.model_root_path + "catboost_dynamic_params.pkl", "r")
ntrees = fixed_params["n_estimators"]


class CustomDataset(Dataset):
    def __init__(self, feature_seq, feature_meta, label=None):
        self.feature_seq = feature_seq
        self.feature_meta = feature_meta
        self.label = np.ones(len(feature_seq)) if label is None else label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {"seq": self.feature_seq[idx], "meta": self.feature_meta[idx], "label": self.label[idx]}

class DenseBlock(nn.Module):
    def __init__(self, in_c, in_out, kernel_size=5, stride=1, act=nn.ReLU()):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(in_c),
            act,
            nn.Conv1d(in_c, in_c, kernel_size=1, groups=in_c, padding="same"),
            nn.BatchNorm1d(in_c),
            act,
            nn.Conv1d(in_c, in_out, kernel_size=kernel_size, stride=stride, padding="same"),
        )

        self.conv2 = nn.Sequential(
            nn.BatchNorm1d(in_c + in_out),
            act,
            nn.Conv1d(in_c + in_out, in_c + in_out, kernel_size=1, groups=in_c + in_out, padding="same"),
            nn.BatchNorm1d(in_c + in_out),
            act,
            nn.Conv1d(in_c + in_out, in_out, kernel_size=kernel_size, stride=stride, padding="same"),
        )

        self.conv3 = nn.Sequential(
            nn.BatchNorm1d(in_c + in_out * 2),
            act,
            nn.Conv1d(in_c + in_out * 2, in_c + in_out * 2, kernel_size=1, groups=in_c + in_out * 2, padding="same"),
            nn.BatchNorm1d(in_c + in_out * 2),
            act,
            nn.Conv1d(in_c + in_out * 2, in_out, kernel_size=kernel_size, stride=stride, padding="same"),
        )

        self.conv4 = nn.Sequential(
            nn.BatchNorm1d(in_c + in_out * 3),
            act,
            nn.Conv1d(in_c + in_out * 3, in_c + in_out * 3, kernel_size=1, groups=in_c + in_out * 3, padding="same"),
            nn.BatchNorm1d(in_c + in_out * 3),
            act,
            nn.Conv1d(in_c + in_out * 3, in_out, kernel_size=kernel_size, stride=stride, padding="same"),
        )
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat([x, x1], dim=1))
        x3 = self.conv3(torch.cat([x, x1, x2], dim=1))
        x4 = self.conv4(torch.cat([x, x1, x2, x3], dim=1))
        return x4

class TransitionLayer(nn.Module):
    def __init__(self, pool_size):
        super(TransitionLayer,self).__init__()
        self.pooling = nn.AvgPool1d(kernel_size=pool_size)

    def forward(self, x):
        return self.pooling(x)

class DNN_CustomModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        act = nn.LeakyReLU() if params["act"] == "leakyrelu" else nn.ReLU()
        self.input_transition = nn.Sequential(
            nn.BatchNorm1d(params["seq_n_features"]),
            nn.LeakyReLU(),
            nn.Conv1d(params["seq_n_features"], params["base_hidden_layers"] * 1, kernel_size=5, stride=3, padding="valid"),
            nn.AdaptiveAvgPool1d(256),
        )

        self.dense_blocks = []
        in_c = params["base_hidden_layers"] * 1
        for size_multiplier in [2, 4, 6, 8]:
            self.dense_blocks.append(nn.Sequential(
                DenseBlock(in_c, params["base_hidden_layers"] * size_multiplier, kernel_size=3, stride=1, act=act),
                TransitionLayer(2),
            ))
            in_c = params["base_hidden_layers"] * size_multiplier
        self.dense_blocks = nn.Sequential(*self.dense_blocks)
        self.lstm = nn.LSTM(in_c, params["lstm_hidden_layers"], num_layers=2, batch_first=True)

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Dropout(params["dropoutRate"]),
            nn.Linear((params["lstm_hidden_layers"] * 16), params["concat_hidden_layers"]),
            act,
            nn.Dropout(params["dropoutRate"]),
            nn.Linear(params["concat_hidden_layers"], params["concat_hidden_layers"]//4),
            act,
        )
        self.regressor = nn.Linear(params["concat_hidden_layers"]//4, 1)

    def forward(self, seq, meta):
        x = self.input_transition(seq)
        x = self.dense_blocks(x)
        x = torch.permute(x, (0, 2, 1))
        x, _ = self.lstm(x, self.init_states(len(x)))
        seq_embed = self.flatten(x)
        x = self.fc(seq_embed)
        return seq_embed, self.regressor(x)

    def init_states(self, batch_size):
        h0 = torch.zeros((2, batch_size, self.params["lstm_hidden_layers"]), dtype=torch.float32).to(device)
        c0 = torch.zeros((2, batch_size, self.params["lstm_hidden_layers"]), dtype=torch.float32).to(device)
        return (h0, c0)


def get_embeddings(model, dl):
    output = []
    for batch in tqdm(dl):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            output.append(model(seq=batch["seq"], meta=batch["meta"])[0].detach().cpu().numpy())
    return np.concatenate(output)


class CAT_CustomModel():
    def __init__(self, output_average="best"):
        # available options : "all", "best"
        self.output_average = output_average
        self.model_list = []
        self.eval_score_list = []
        self.ntrees = ntrees
        for idx, params in enumerate(product(*dynamic_params.values())):
            tmp_params = fixed_params.copy()
            tmp_params.update({k: v for k, v in zip(dynamic_params.keys(), params)})
            self.model_list.append(cat.CatBoostRegressor(**tmp_params))

    def fit(self, x, y, groups, eval_x, eval_y, eval_groups):
        for model in tqdm(self.model_list, desc="Training..."):
            model.fit(
                x, y,
                eval_set=[(eval_x, eval_y)],
                early_stopping_rounds=int(self.ntrees * 0.2), verbose=0,
            )
            y_pred = model.predict(eval_x)
            self.eval_score_list.append(skl_metrics.mean_absolute_error(eval_y, y_pred))
        print("=== Best model & Score ===")
        print("Model :", self.model_list[np.argmin(self.eval_score_list)])
        print("MAE :", self.eval_score_list[np.argmin(self.eval_score_list)])
        print("Best Trees :", self.model_list[np.argmin(self.eval_score_list)].get_best_iteration())

    def predict(self, x):
        if self.output_average == "all":
            return np.stack([model.predict(x) for model in self.model_list], axis=0).mean(axis=0), self.eval_score_list[np.argmin(self.eval_score_list)]
        else:
            return self.model_list[np.argmin(self.eval_score_list)].predict(x), self.eval_score_list[np.argmin(self.eval_score_list)]


def main():    
    # loading test metadata
    df_test = pd.concat([
        pd.read_csv(CFG.dataset_root_path + "submission.csv"),
    ], axis=0).reset_index(drop=True)
    df_test.columns = df_test.columns.str.lower()
    df_test["age_type"] = df_test["filename"].apply(lambda x: x.split("_")[1].split("_")[0])
    df_test["gender"] = df_test["gender"].apply(lambda x: 1 if x == "MALE" else 0)
    df_test["age_type"] = df_test["age_type"].apply(lambda x: 1 if x == "adult" else 0)

    # loading test ecg sequence data
    test_ecg_seq_feature = []
    for fpath in tqdm(df_test["filename"]):
        df_ecg = np.load(CFG.dataset_root_path + f"ECG_{fpath.split('_')[1].split('_')[0]}_numpy_valid/{fpath}.npy").astype("float32")
        df_ecg = np.stack([df_ecg[((lead + 0) * 5000):((lead + 1) * 5000)] for lead in range(12)], axis=0)
        test_ecg_seq_feature.append(F.avg_pool1d(torch.from_numpy(df_ecg), 10, 10).detach().cpu().numpy())
    test_ecg_seq_feature = np.stack(test_ecg_seq_feature, axis=0)
    test_fnames = df_test["filename"].values
    df_test = df_test.drop("filename", axis=1)

    # scaling
    scaler_seq = pickleIO(None, CFG.model_root_path + "scaler_seq.pkl", "r")
    for target in df_test["age_type"].unique():
        tmp = test_ecg_seq_feature[(df_test["age_type"] == target).values]
        for i in range(len(CFG.lead_names)):
            tmp[:, i] = (tmp[:, i] - scaler_seq[target][i]["min"]) / (scaler_seq[target][i]["max"] - scaler_seq[target][i]["min"])
        test_ecg_seq_feature[df_test["age_type"] == target] = tmp.copy()

    # get embedding vector from pretrained-DNN model
    seq_embed = np.zeros((len(df_test), 1024), dtype="float32")
    for target in df_test["age_type"].unique():
        target_name = "adult" if target == 1 else "child"
        target_seq = test_ecg_seq_feature[(df_test["age_type"] == target).values]
        target_meta = df_test[(df_test["age_type"] == target).values]
        ds = CustomDataset(
            feature_seq=target_seq.astype("float32"),
            feature_meta=target_meta.drop(["age", "age_type"], axis=1).values.astype("float32"),
            label=None
        )
        embed = []
        for fold in range(5):
            model = DNN_CustomModel(model_params)
            model.load_state_dict(torch.load(CFG.model_root_path + f"dnn_rawWithkaggle_densenetLSTM_v2/model_target{target_name}_fold{fold}_best.pth", map_location="cpu")["model"])
            model.to(device)
            model.eval()
            embed.append(get_embeddings(model, DataLoader(ds, batch_size=64, shuffle=False)))
        embed = np.stack(embed, axis=0).mean(axis=0)
        seq_embed[(df_test["age_type"] == target).values] = embed
    del ds, model, embed, test_ecg_seq_feature
    gc.collect()
    torch.cuda.empty_cache()

    # inference
    test_pred_container = {}
    for target in df_test["age_type"].unique():
        test_pred_container[target] = []
        target_name = "adult" if target == 1 else "child"
        for fold in range(CFG.n_folds):
            print(f"\n=== FOLD {fold} ===")
            seed_everything(fold)
            df_test_seq = seq_embed[(df_test["age_type"] == target).values]
            model = pickleIO(None, CFG.model_root_path + f"dnn_rawWithkaggle_catboost_v2/fold{fold}_target{target}_model.pkl", 'r')
            test_pred_container[target].append(model.predict(df_test_seq)[0])
            del model
            gc.collect()
            torch.cuda.empty_cache()

    df_test.loc[df_test["age_type"] == 1, "age"] = np.stack(test_pred_container[1], axis=0).mean(axis=0)
    df_test.loc[df_test["age_type"] == 0, "age"] = np.stack(test_pred_container[0], axis=0).mean(axis=0)
    df_test["age"] = np.clip(df_test["age"], 0.0, 122.0)
    df_test["filename"] = test_fnames
    df_test[["filename", "age"]].to_csv("./result.csv", index=False, encoding="utf8")

    return 0


if __name__ == "__main__":
    main()

