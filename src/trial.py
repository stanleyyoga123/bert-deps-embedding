import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse

torch.manual_seed(42)

from transformers import BertTokenizer
from torch.utils.data import DataLoader

from model import BertDependencyModelEnc, BertClassifier
from loader import SSTRawDataset
from trainer.trainer import trainer

if __name__ == "__main__":
    device = torch.device(f"cuda:6" if torch.cuda.is_available() else "cpu")

    tokenizers = [
        BertTokenizer.from_pretrained("bert-large-uncased"),
        BertTokenizer.from_pretrained(f"resources/tokenizers/tokenizer-deps-wordsonly")
    ]
    deps_tokenizer = BertTokenizer.from_pretrained("resources/tokenizers/tokenizer-deps-raw")

    model = BertClassifier()
    train = trainer 

    loader_train = DataLoader(SSTRawDataset("train", tokenizers, deps_tokenizer), batch_size=4, shuffle=True)
    loader_dev = DataLoader(SSTRawDataset("dev", tokenizers, deps_tokenizer), batch_size=4, shuffle=True)
    loader_test = DataLoader(SSTRawDataset("test", tokenizers, deps_tokenizer), batch_size=4, shuffle=True)

    dataset_train = SSTRawDataset("train", tokenizers, deps_tokenizer)

    model.to(device)

    print(model)
    print(dataset_train[0])

    # train(
    #     model,
    #     loader_train,
    #     loader_dev,
    #     loader_test,
    #     device,
    #     "Nani",
    #     "sst-raw",
    #     learning_rate0=args.lr
    # )