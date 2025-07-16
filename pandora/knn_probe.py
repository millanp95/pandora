#!/usr/bin/env python

import argparse
import sys

import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
import torch
import wandb
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from src.datasets import data_from_df
from src.models import CNN_MLM
from src.utils import load_pretrained_model


def run_knn(config):

    data_folder = config.data_dir
    train = pd.read_csv(f"{data_folder}/supervised_train.csv")
    test = pd.read_csv(f"{data_folder}/unseen.csv")

    target_level = config.target_level + "_name"  # "species_name"

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # Get pipeline for reference labels:
    labels = train[target_level].to_list()
    label_set = sorted(set(labels))
    label_pipeline = lambda x: label_set.index(x)

    X, y_train = data_from_df(train, target_level, label_pipeline)
    X_test, y_test = data_from_df(test, target_level, label_pipeline)

    numClasses = max(y_train) + 1
    print(f"[INFO]: There are {numClasses} taxonomic groups")

    model = CNN_MLM(max_len=config.max_len).to(device)
    model_path = config.model_checkpoint

    print(f"Getting the model from: {model_path}")

    # try:
    print(f"Loading training snapshot from {config.model_checkpoint}")
    model, ckpt = load_pretrained_model(config.model_checkpoint, device=device)
    print(f"Model at epoch {ckpt['epoch']} was succesfully loaded")
    model.to(device)
    model.eval()
    # except Exception:
    #    print("There was a problem loading the model")
    #    return

    # REGISTER FORWARD HOOK TO GET INTERMEDIATE REPRESENTATIONS ===

    hidden = {}

    def get_activation(name):
        def hook(model, input, output):
            hidden[name] = output.detach()

        return hook

    # Register hook on the ReLU layer
    model.transformer.register_forward_hook(get_activation("final_features"))

    # USE MODEL AS FEATURE EXTRACTOR ===================================
    dna_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(X_test.shape[0])):
            x = torch.Tensor(X_test[i]).to(device)
            # 1) Padding mask: 1 for real tokens (sum>0), 0 for padding (zero-vector)
            att_mask = (x.sum(dim=-1) > 0).int().to(device)
            out = model(x.unsqueeze(0), att_mask.unsqueeze(0))
            # z = hidden['final_features'].squeeze(0) * att_mask.squeeze(0)
            # z = z.sum(dim=1) / att_mask.sum()
            # print(z.shape)
            # print(att_mask.shape)
            # att_mask = att_mask.transpose(0,1)
            # print(att_mask.shape)
            z = hidden["final_features"]
            # print(z.squeeze(1).shape)
            z = z.squeeze(1).t() * att_mask[::4]
            z = z.sum(dim=1) / att_mask.sum()
            # z = z.mean(dim=0, dtype=float)

            dna_embeddings.extend(z.cpu().numpy())

    X_test = np.array(dna_embeddings).reshape(-1, 768)
    print(X_test.shape)

    train_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(X.shape[0])):
            x = torch.Tensor(X[i]).to(device)
            # 1) Padding mask: 1 for real tokens (sum>0), 0 for padding (zero-vector)
            att_mask = (x.sum(dim=-1) > 0).int().to(device)
            out = model(x.unsqueeze(0), att_mask.unsqueeze(0))
            # z = hidden['final_features'].squeeze(0) * att_mask.squeeze(0)
            # z = z.sum(dim=1) / att_mask.sum()
            # print(z.shape)
            # print(att_mask.shape)
            # att_mask = att_mask.transpose(0,1)
            # print(att_mask.shape)
            z = hidden["final_features"].squeeze(1).t() * att_mask[::4]
            # print(z.shape)
            z = z.sum(dim=1) / att_mask.sum()
            # z = z.mean(dim=0, dtype=float)
            train_embeddings.extend(z.cpu().numpy())

    # X_test = np.array(dna_embeddings).reshape(-1, 660)
    # print(X_test.shape)

    X = np.array(train_embeddings).reshape(-1, 768)
    print(X.shape)

    neigh = KNeighborsClassifier(n_neighbors=1, metric="cosine")
    neigh.fit(X, y_train)
    print("Accuracy:", neigh.score(X_test, y_test))
    y_pred = neigh.predict(X_test)

    # Create results dictionary
    results = {}
    results["count"] = len(y_test)
    # Note that these evaluation metrics have all been converted to percentages
    results["accuracy"] = 100.0 * sklearn.metrics.accuracy_score(y_test, y_pred)
    results["accuracy-balanced"] = 100.0 * sklearn.metrics.balanced_accuracy_score(
        y_test, y_pred
    )
    results["f1-micro"] = 100.0 * sklearn.metrics.f1_score(
        y_test, y_pred, average="micro"
    )
    results["f1-macro"] = 100.0 * sklearn.metrics.f1_score(
        y_test, y_pred, average="macro"
    )
    results["f1-support"] = 100.0 * sklearn.metrics.f1_score(
        y_test, y_pred, average="weighted"
    )

    wandb.log({f"eval/{k}": v for k, v in results.items()})

    print("Evaluation results:")
    for k, v in results.items():
        if k == "count":
            print(f"  {k + ' ':.<21s}{v:7d}")
        elif k in ["max_ram_mb", "peak_vram_mb"]:
            print(f"  {k + ' ':.<24s} {v:6.2f} MB")
        else:
            print(f"  {k + ' ':.<24s} {v:6.2f} %")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Path to the folder containing the data in the desired CSV format",
    )
    parser.add_argument(
        "--target_level",
        default="genus",
        help="Desired taxonomic rank, either 'genus' or 'species'",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=660,
        help="Maximum length used during pretraining",
    )
    parser.add_argument(
        "--model-checkpoint",
        default="model_checkpoints/best_checkpoint.pt",
        help="PATH to model checkpoint",
    )

    config = parser.parse_args()
    wandb.init(project="CNN_MLM", name="knn_CNN_CANADA-1.5M", config=vars(config))
    wandb.config.update(vars(config))  # log your CLI args
    run_knn(config)
