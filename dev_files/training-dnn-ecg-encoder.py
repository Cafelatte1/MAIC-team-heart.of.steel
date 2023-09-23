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


def get_optimizer_params(model, eta, weight_decay):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        # apply weight decay
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': eta, 'weight_decay': weight_decay},
        # don't apply weight decay for LayerNormalization layer
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': eta, 'weight_decay': 0.0},
    ]
    return optimizer_parameters


def get_scheduler(optimizer, num_warmup_steps, num_training_steps):
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, power=0.5, lr_end=1e-7
    )
    return scheduler


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def train_fn(model, dl, criterion, optimizer, scheduler, grad_scaler):
    model.train()
    metrics = {
        "loss": AverageMeter("loss", fmt=":.5f"),
        "mae": AverageMeter("loss", fmt=":.5f"),
    }

    for idx, batch in enumerate(dl):
        batch = {k: v.to(device).to(torch.float32) for k, v in batch.items()}
        with torch.cuda.amp.autocast():
            _, output = model(seq=batch["seq"], meta=batch["meta"])
            loss = criterion(output, batch["label"])
        # initialization gradients to zero
        optimizer.zero_grad()
        # get scaled gradients by float16 (default)
        grad_scaler.scale(loss).backward()
        # apply original gradients (unscaling) to parameters
        # if these gradients do not contain infs or NaNs, optimizer.step() is then called.
        # otherwise, optimizer.step() is skipped.
        grad_scaler.step(optimizer)
        grad_scaler.update()
        # calcuate metrics
        metrics["loss"].update(loss.item())
        metrics["mae"].update(skl_metrics.mean_absolute_error(output.flatten().detach().cpu().numpy(), batch["label"].flatten().detach().cpu().numpy()))

        if CFG.debug:
            if idx >= 10:
                break

    # update scheduler
    scheduler.step()

    del idx, batch, output
    gc.collect()
    torch.cuda.empty_cache()
    return metrics

def valid_fn(model, dl, criterion):
    model.eval()
    metrics = {
        "loss": AverageMeter("loss", fmt=":.5f"),
        "mae": AverageMeter("loss", fmt=":.5f"),
    }

    for idx, batch in enumerate(dl):
        batch = {k: v.to(device).to(torch.float32) for k, v in batch.items()}
        with torch.no_grad():
            _, output = model(seq=batch["seq"], meta=batch["meta"])
            loss = criterion(output, batch["label"])
        # calcuate metrics
        metrics["loss"].update(loss.item())
        metrics["mae"].update(skl_metrics.mean_absolute_error(output.flatten().detach().cpu().numpy(), batch["label"].flatten().detach().cpu().numpy()))

    del idx, batch, output
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def do_training(fold, model, target_name):
    # set loss & optimizer
    optimizer_parameters = get_optimizer_params(
        model,
        eta=CFG.eta,
        weight_decay=CFG.weight_decay
    )
    optimizer = AdamW(optimizer_parameters, lr=CFG.eta, weight_decay=CFG.weight_decay)
    scheduler = get_scheduler(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=CFG.epochs
    )
    grad_scaler = torch.cuda.amp.GradScaler()
    criterion = nn.MSELoss()

    best_score = np.inf
    early_stopping_cnt = 0
    for epoch in range(CFG.epochs):
        epoch_start_time = time.time()

        # training
        train_metrics = train_fn(model, train_dl, criterion, optimizer, scheduler, grad_scaler)
        # evaluation
        valid_metrics = valid_fn(model, valid_dl, criterion)
        score = valid_metrics["mae"].avg

        print("Epoch[{0}/{1}]\n train loss : {2}\n valid loss : {3}\n train mae : {4}\n valid mae : {5}\n eta : {6}\n Elapsed : {7}\n"
              .format(
                  epoch+1, CFG.epochs,
                  round(train_metrics["loss"].avg, 5), round(valid_metrics["loss"].avg, 5),
                  round(train_metrics["mae"].avg, 5), round(valid_metrics["mae"].avg, 5),
                  round(optimizer.param_groups[-1]['lr'], 5), round(time.time() - epoch_start_time, 3)
              )
        )

        if score < best_score:
            best_score = score
            return_score_dic = {
                "fold": fold,
                "train_loss": train_metrics["loss"].avg,
                "valid_loss": valid_metrics["loss"].avg,
                "train_mae": train_metrics["mae"].avg,
                "valid_mae": valid_metrics["mae"].avg,
            }
            print("INFO: Found best weight\n\n")
            torch.save(
                {'model': model.state_dict()},
                f"{architecture_path}model_target{target_name}_fold{fold}_best.pth",
            )
            early_stopping_cnt = 0
        else:
            early_stopping_cnt += 1

        if early_stopping_cnt == CFG.early_stopping_rounds:
            break

    return return_score_dic


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

def do_inference(model, dl):
    model.eval()
    output = []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            output.append(model(seq=batch["seq"], meta=batch["meta"])[1].detach().cpu().numpy().flatten())
    return np.concatenate(output)


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
                full_ecg_seq_feature[(~df_full["fold"].isna() & (df_full[f"fold_{target_name}"] != -1) & (df_full[f"fold_{target_name}"] != fold)).values],
                full_ecg_seq_feature[(df_full["fold"].isna() & (df_full["age_type"] == target)).values][kaggle_fold[fold]],
            ], axis=0)

            assert len(df_train_seq) == len(df_train)
            train_ds = CustomDataset(
                feature_seq=df_train_seq.astype("float32"),
                feature_meta=df_train.drop(["age", "age_type", "fold", "fold_adult", "fold_child"], axis=1).values.astype("float32"),
                label=df_train[["age"]].astype("float32").values
            )
            train_dl = DataLoader(train_ds, batch_size=128 if target == 1 else 128 // 4, shuffle=True, drop_last=True)
            print("num iteration :", len(train_dl))

            # === valid ===
            df_valid = df_full[(~df_full["fold"].isna() & (df_full[f"fold_{target_name}"] != -1) & (df_full[f"fold_{target_name}"] == fold)).values].reset_index(drop=True)
            df_valid_seq = full_ecg_seq_feature[(~df_full["fold"].isna() & (df_full[f"fold_{target_name}"] != -1) & (df_full[f"fold_{target_name}"] == fold)).values]
            assert len(df_valid_seq) == len(df_valid)
            valid_ds = CustomDataset(
                feature_seq=df_valid_seq.astype("float32"),
                feature_meta=df_valid.drop(["age", "age_type", "fold", "fold_adult", "fold_child"], axis=1).values.astype("float32"),
                label=df_valid[["age"]].values.astype("float32")
            )
            valid_dl = DataLoader(valid_ds, batch_size=CFG.batch_size if target == 1 else CFG.batch_size // 4, shuffle=False)

            # create model
            model = DNN_CustomModel(model_params)
            model.to(device)

            score = do_training(fold, model, "adult" if target == 1 else "child")

            fold_score.append({"mae": score["valid_mae"]})

            del model, train_ds, train_dl, valid_ds, valid_dl
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


    test_pred_container = {}
    for target in df_test["age_type"].unique():
        test_pred_container[target] = []
        target_name = "adult" if target == 1 else "child"
        for fold in range(CFG.n_folds):
            print(f"\n=== FOLD {fold} ===")
            seed_everything(fold)

            df_test_x = df_test[(df_test["age_type"] == target).values].reset_index(drop=True)
            df_test_seq = test_ecg_seq_feature[(df_test["age_type"] == target).values]
            assert len(df_test_seq) == len(df_test_x)
            test_ds = CustomDataset(
                feature_seq=df_test_seq.astype("float32"),
                feature_meta=df_test_x.drop(["age", "age_type"], axis=1).values.astype("float32"),
                label=None
            )
            test_dl = DataLoader(test_ds, batch_size=CFG.batch_size//4 if (target == 0) else CFG.batch_size, shuffle=False)

            # create model
            model = DNN_CustomModel(model_params)
            model.to(device)
            model.load_state_dict(torch.load(f"{architecture_path}model_target{target_name}_fold{fold}_best.pth")["model"])
            test_pred_container[target].append(do_inference(model, test_dl))

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

