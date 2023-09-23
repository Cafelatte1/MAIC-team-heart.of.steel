GLOBAL_SEED = 42

import os
os.environ['PYTHONHASHSEED'] = str(GLOBAL_SEED)

import numpy as np # linear algebra
from numpy import random as np_rnd
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import random as rnd
import pickle
import gc
import time
from itertools import product

from torch.utils.data import TensorDataset, Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup
from scipy.signal import spectrogram

import catboost as cat
from sklearn import metrics as skl_metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings(action='ignore')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from helper_functions import *

class CFG:
    debug = False
    dataset_root_path = r"/content/drive/MyDrive/Colab Notebooks/projects/etc/MAIC/심전도 데이터를 활용한 나이 예측/dataset/MAIC_data/"
    dataset_version = "v5"
    n_comp = 64
    TF = False
    TF_type = "indiv"
    use_ecg_stats = True
    use_ecg_seq = True
    pca_seq = True
    n_folds = 1 if debug else 5
    lead_names = ["std1", "std2", "std3", "v1", "v2", "v3", "v4", "v5", "v6", "augvr", "augvl", "augvf"]
    ecg_hz = 500

    epochs = 2 if debug else 30
    early_stopping_rounds = 10
    batch_size = 32
    eta = 5e-4
    weight_decay = 1e-4


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
    model.eval()
    output = []
    for batch in tqdm(dl):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            output.append(model(seq=batch["seq"], meta=batch["meta"])[0].detach().cpu().numpy())
    return np.concatenate(output)

ntrees = 100 if CFG.debug else 5000
eta = 5e-3

fixed_params = {
    "boosting_type": "Plain",
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "bootstrap_type": "Bayesian",
    "n_estimators": ntrees,
    "learning_rate": eta,
    "max_depth": 8,
    "verbose": False,
    "task_type": "GPU",
    "use_best_model": True,
}
dynamic_params = {
    "reg_lambda": [1.0],
}

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

def do_training(df_train_x, df_train_y, df_train_groups, df_valid_x, df_valid_y, df_valid_groups):

    # training
    model = CAT_CustomModel()
    model.fit(df_train_x, df_train_y, df_train_groups, df_valid_x, df_valid_y, df_valid_groups)
    y_pred, eval_score = model.predict(df_valid_x)

    return model, eval_score

def main():
    # === preprocessing ===
    prep_root_path = r"/content/drive/MyDrive/Colab Notebooks/projects/etc/MAIC/심전도 데이터를 활용한 나이 예측/dataset/prep_data/" + CFG.dataset_version + "/"
    df_full = pickleIO(None, prep_root_path + "df_full.pkl", "r")
    full_ecg_seq_feature = np.load(r"/content/drive/MyDrive/Colab Notebooks/projects/etc/MAIC/심전도 데이터를 활용한 나이 예측/dataset/prep_data/" + "v1" + "/" + "full_ecg.npz")["ecg"].astype("float32")
    fold_split = np.load(prep_root_path + "fold_split.npz")


    df_kaggle = {"age": [], "gender": [], "ecg": []}
    for chunk in range(20):
        tmp = np.load(f"chunk{chunk}.npz")
        for k in df_kaggle.keys():
            df_kaggle[k].append(tmp[k])
    del tmp; gc.collect()
    kaggle_ecg = F.avg_pool1d(torch.from_numpy(np.concatenate(df_kaggle["ecg"]).astype("float32")), 10, 10).detach().cpu().numpy()
    df_kaggle = pd.DataFrame({"age": np.concatenate(df_kaggle["age"]), "gender": np.concatenate(df_kaggle["gender"])})
    df_kaggle["age_type"] = df_kaggle["age"].apply(lambda x: 1 if x >= 19 else 0)


    df_full["gender"] = df_full["gender"].apply(lambda x: 1 if x == "MALE" else 0).astype("int32")
    df_full["age_type"] = df_full["age_type"].apply(lambda x: 1 if x == "adult" else 0).astype("int32")
    df_full = df_full.drop("filename", axis=1)
    df_full["fold"] = fold_split["fold_split"].astype("float32")
    df_full["fold_adult"] = -1; df_full["fold_child"] = -1
    df_full.loc[df_full["age_type"] == 1, "fold_adult"] = fold_split["adult"].astype("float32")
    df_full.loc[df_full["age_type"] == 0, "fold_child"] = fold_split["child"].astype("float32")


    df_full = pd.concat([df_full, df_kaggle], axis=0)
    full_ecg_seq_feature = np.concatenate([full_ecg_seq_feature, kaggle_ecg])
    del df_kaggle, kaggle_ecg; gc.collect()
    df_full[["age", "fold_adult", "fold_child"]] = df_full[["age", "fold_adult", "fold_child"]].astype("float32")
    df_full["age_type"] = df_full["age_type"].astype("int32")


    full_ecg_seq_feature = full_ecg_seq_feature[df_full["age"] > 0]
    df_full = df_full[df_full["age"] > 0].reset_index(drop=True)


    scaler_seq = {}
    for target in df_full["age_type"].unique():
        scaler_seq[target] = {}
        tmp = full_ecg_seq_feature[df_full["age_type"] == target]
        for i in range(len(CFG.lead_names)):
            scaler_seq[target][i] = {"min": 0.0, "max": 0.0}
            scaler_seq[target][i]["min"] = tmp[:, i].min()
            scaler_seq[target][i]["max"] = tmp[:, i].max()
            tmp[:, i] = (tmp[:, i] - scaler_seq[target][i]["min"]) / (scaler_seq[target][i]["max"] - scaler_seq[target][i]["min"])
        full_ecg_seq_feature[df_full["age_type"] == target] = tmp.copy()


    imputer_cols = [i for i in df_full if i not in ["age_type", "age", "fold", "fold_adult", "fold_child"]]
    imputer = {}
    for target in df_full["age_type"].unique():
        imputer[target] = df_full.loc[df_full["age_type"] == target, imputer_cols].median()
        for i in df_full.columns:
            if i in ["age_type", "age", "fold", "fold_adult", "fold_child"]:
                continue
            if df_full.loc[df_full["age_type"] == target, i].isna().sum() > 0:
                df_full.loc[(df_full["age_type"] == target) & df_full[i].isna(), i] = imputer[target][i]


    df_full = df_full[full_ecg_seq_feature.sum(axis=(1,2)) != 0].reset_index(drop=True)
    full_ecg_seq_feature = full_ecg_seq_feature[full_ecg_seq_feature.sum(axis=(1,2)) != 0]

    architecture_root_path = r"/content/drive/MyDrive/Colab Notebooks/projects/etc/MAIC/심전도 데이터를 활용한 나이 예측/" + "architecture/"
    architecture_name = "dnn_rawWithkaggle_densenetLSTM_v2"
    architecture_path = architecture_root_path + architecture_name + "/"
    createFolder(architecture_path)
    
    model_params = {
        "dropoutRate": 0.5,
        "seq_n_features": full_ecg_seq_feature.shape[1],
        "seq_len": full_ecg_seq_feature.shape[2],
        "meta_n_features": df_full.shape[1] - 5,
        "base_hidden_layers": 32,
        "lstm_hidden_layers": 64,
        "concat_hidden_layers": 512,
        "act": "leakyrelu"
    }
    seq_embed = np.zeros((len(df_full), 1024), dtype="float32")
    for target in df_full["age_type"].unique():
        target_name = "adult" if target == 1 else "child"
        target_seq = full_ecg_seq_feature[(df_full["age_type"] == target).values]
        target_meta = df_full[(df_full["age_type"] == target).values]
        ds = CustomDataset(
            feature_seq=target_seq.astype("float32"),
            feature_meta=target_meta.drop(["age", "age_type", "fold", "fold_adult", "fold_child"], axis=1).values.astype("float32"),
            label=None
        )
        embed = []
        for subfold in range(5):
            model = DNN_CustomModel(model_params)
            model.to(device)
            model.load_state_dict(torch.load(f"/content/drive/MyDrive/Colab Notebooks/projects/etc/MAIC/심전도 데이터를 활용한 나이 예측/architecture/dnn_rawWithkaggle_densenetLSTM_v2/model_target{target_name}_fold{subfold}_best.pth", map_location="cpu")["model"])
            embed.append(get_embeddings(model, DataLoader(ds, batch_size=64, shuffle=False)))
        embed = np.stack(embed, axis=0).mean(axis=0)
        seq_embed[(df_full["age_type"] == target).values] = embed
    del ds, model, embed, full_ecg_seq_feature
    gc.collect()
    torch.cuda.empty_cache()

    # === training ===
    target_output_container = {}
    for target in df_full["age_type"].unique():
        target_name = "adult" if target == 1 else "child"
        fold_score = []
        target_output_container[target] = {}

        kaggle_kfolds_spliter = StratifiedKFold(5, shuffle=True, random_state=GLOBAL_SEED)
        kaggle_split_vector = df_full.loc[(df_full["fold"].isna() & (df_full["age_type"] == target)).values, "age"]
        kaggle_fold = {fold: i for fold, (i, j) in enumerate(kaggle_kfolds_spliter.split(kaggle_split_vector, pd.qcut(kaggle_split_vector, 5).astype("str")))}

        for fold in range(CFG.n_folds):
            print(f"\n=== FOLD {fold} ===")
            seed_everything(fold)

            # === train ===
            df_train = pd.concat([
                df_full[(~df_full["fold"].isna() & (df_full[f"fold_{target_name}"] != -1) & (df_full[f"fold_{target_name}"] != fold)).values].reset_index(drop=True),
                df_full[(df_full["fold"].isna() & (df_full["age_type"] == target)).values].iloc[kaggle_fold[fold]].reset_index(drop=True),
            ], axis=0).reset_index(drop=True)

            df_train_seq = np.concatenate([
                seq_embed[(~df_full["fold"].isna() & (df_full[f"fold_{target_name}"] != -1) & (df_full[f"fold_{target_name}"] != fold)).values],
                seq_embed[(df_full["fold"].isna() & (df_full["age_type"] == target)).values][kaggle_fold[fold]],
            ], axis=0)

            # === valid ===
            df_valid = df_full[(~df_full["fold"].isna() & (df_full[f"fold_{target_name}"] != -1) & (df_full[f"fold_{target_name}"] == fold)).values].reset_index(drop=True)
            df_valid_seq = seq_embed[(~df_full["fold"].isna() & (df_full[f"fold_{target_name}"] != -1) & (df_full[f"fold_{target_name}"] == fold)).values]

            # create model
            model, score = do_training(
                df_train_x=df_train_seq, df_train_y=df_train["age"].values, df_train_groups=None,
                df_valid_x=df_valid_seq, df_valid_y=df_valid["age"].values, df_valid_groups=None,
            )

            fold_score.append({"mae": score})
            pickleIO(model, f"{architecture_path}fold{fold}_target{target}_model.pkl", 'w')

            del model
            gc.collect()
            torch.cuda.empty_cache()

            if CFG.debug:
                break

        target_output_container[target]["score"] = fold_score


    df_score = pd.DataFrame()
    df_score["adult"] = [i for i in pd.DataFrame(target_output_container[1]["score"])["mae"]]
    df_score["child"] = [i for i in pd.DataFrame(target_output_container[0]["score"])["mae"]]
    df_score.loc["average"] = df_score.mean()
    df_score.to_csv(architecture_path + "eval_score.csv", index=False)


    # === inference ===
    df_test = pickleIO(None, prep_root_path + "df_test.pkl", "r")
    test_ecg_seq_feature = np.load(r"/content/drive/MyDrive/Colab Notebooks/projects/etc/MAIC/심전도 데이터를 활용한 나이 예측/dataset/prep_data/" + "v1" + "/" + "test_ecg.npz")["ecg"].astype("float32")
    df_test["gender"] = df_test["gender"].apply(lambda x: 1 if x == "MALE" else 0)
    df_test["age_type"] = df_test["age_type"].apply(lambda x: 1 if x == "adult" else 0)
    df_test = df_test.drop("filename", axis=1)
    for target in df_test["age_type"].unique():
        tmp = test_ecg_seq_feature[df_test["age_type"] == target]
        for i in range(len(CFG.lead_names)):
            tmp[:, i] = (tmp[:, i] - scaler_seq[target][i]["min"]) / (scaler_seq[target][i]["max"] - scaler_seq[target][i]["min"])
        test_ecg_seq_feature[df_test["age_type"] == target] = tmp.copy()
    for target in df_test["age_type"].unique():
        target_name = "adult" if target == 1.0 else "child"
        for i in df_test.columns:
            if i in ["age_type", "age", "fold", "fold_adult", "fold_child"]:
                continue
            if df_test.loc[df_test["age_type"] == target, i].isna().sum() > 0:
                df_test.loc[(df_test["age_type"] == target) & df_test[i].isna(), i] = imputer[target][i]


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
        for subfold in range(5):
            model = DNN_CustomModel(model_params)
            model.to(device)
            model.load_state_dict(torch.load(f"/content/drive/MyDrive/Colab Notebooks/projects/etc/MAIC/심전도 데이터를 활용한 나이 예측/architecture/dnn_rawWithkaggle_densenetLSTM_v2/model_target{target_name}_fold{subfold}_best.pth", map_location="cpu")["model"])
            embed.append(get_embeddings(model, DataLoader(ds, batch_size=64, shuffle=False)))
        embed = np.stack(embed, axis=0).mean(axis=0)
        seq_embed[(df_test["age_type"] == target).values] = embed
    del ds, model, embed, test_ecg_seq_feature
    gc.collect()
    torch.cuda.empty_cache()


    test_pred_container = {}
    for target in df_test["age_type"].unique():
        test_pred_container[target] = []
        target_name = "adult" if target == 1 else "child"
        for fold in range(CFG.n_folds):
            print(f"\n=== FOLD {fold} ===")
            seed_everything(fold)

            df_test_x = df_test[(df_test["age_type"] == target).values].drop(["age_type", "age"], axis=1).reset_index(drop=True)
            df_test_seq = seq_embed[(df_test["age_type"] == target).values]

            model = pickleIO(None, f"{architecture_path}fold{fold}_target{target}_model.pkl", 'r')
            test_pred_container[target].append(model.predict(df_test_seq)[0])

            del model
            gc.collect()
            torch.cuda.empty_cache()

            if CFG.debug:
                break


    test_pred = np.zeros_like(df_test["age"])
    test_pred[df_test["age_type"] == 1] = np.stack(test_pred_container[1], axis=0).mean(axis=0)
    test_pred[df_test["age_type"] == 0] = np.stack(test_pred_container[0], axis=0).mean(axis=0)
    test_pred = np.clip(test_pred, 0.0, 122.0)

    submission = pd.read_csv(CFG.dataset_root_path + "submission.csv")
    submission["AGE"] = test_pred
    submission.to_csv(architecture_path + f"{architecture_name}_submission.csv", index=False)
    
    return 0


if __name__ == "__main__":
    main()

