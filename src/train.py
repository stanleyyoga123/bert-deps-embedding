import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse

# torch.manual_seed(6783409) # Work 0.93an for Bert Large only
# torch.manual_seed(8594302) # Work 0.93an for Bert Large only
# torch.manual_seed(764243)

from transformers import BertTokenizer
from torch.utils.data import DataLoader

from model import BertDependencyModelEnc, BertClassifier
from loader import SSTRawDataset, SSTDepsExtDataset
from trainer.trainer import trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bert")
    parser.add_argument("--pretrained", type=str, default="bert-large-uncased")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--ww", type=float, default=0.5)
    parser.add_argument("--ds", type=str, default="deps")
    parser.add_argument("--dw", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", type=str, default="deps-wordsonly")
    
    args = parser.parse_args()
    model_type = args.model
    gpu = args.gpu
    name = args.name
    ww = args.ww
    dw = args.dw
    tokenizer_name = args.tokenizer
    seed = args.seed
    pretrained = args.pretrained
    ds = args.ds

    torch.manual_seed(seed)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    print("Configurations")
    print(f"Seed: {seed}")
    print(f"Dataset: {ds}")
    tokenizers = [
        BertTokenizer.from_pretrained(pretrained),
        BertTokenizer.from_pretrained(f"resources/tokenizers/tokenizer-{tokenizer_name}")
    ]
    deps_tokenizer = BertTokenizer.from_pretrained("resources/tokenizers/tokenizer-deps-raw")


    if model_type == "bert":
        model = BertClassifier(pretrained)
        # train = default_train
        train = trainer

    elif model_type == "bert-dep":
        model = BertDependencyModelEnc.from_pretrained(pretrained)
        model.create_embeddings("resources/embeddings/state-dict-deps-wordsonly.pkl", "resources/embeddings/state-dict-deps-raw.pkl")
        model.add_special_ids(tokenizers[1].vocab["[PAD]"], deps_tokenizer.vocab["[UNK]"], deps_tokenizer.vocab["[PAD]"])
        model.add_weights(ww, dw)
        model.add_device(device)
        train = trainer

    if ds == "ext":
        loader_train = DataLoader(SSTDepsExtDataset("train", tokenizers, deps_tokenizer), batch_size=8, shuffle=True)
        loader_dev = DataLoader(SSTDepsExtDataset("dev", tokenizers, deps_tokenizer), batch_size=8, shuffle=True)
        loader_test = DataLoader(SSTDepsExtDataset("test", tokenizers, deps_tokenizer), batch_size=8, shuffle=True)      
    else:
        loader_train = DataLoader(SSTRawDataset("train", tokenizers, deps_tokenizer), batch_size=8, shuffle=True)
        loader_dev = DataLoader(SSTRawDataset("dev", tokenizers, deps_tokenizer), batch_size=8, shuffle=True)
        loader_test = DataLoader(SSTRawDataset("test", tokenizers, deps_tokenizer), batch_size=8, shuffle=True)

    model.to(device)
    train(
        model,
        loader_train,
        loader_dev,
        loader_test,
        device,
        name,
        "sst-raw",
        learning_rate0=args.lr
    )