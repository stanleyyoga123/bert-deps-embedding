import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

import numpy as np
import pickle

import os


class SSTRawDataset(Dataset):
    def __init__(self, type_split_data, word_tokenizers, deps_tokenizer, conv=False, maxlen=512):
        self.conv = conv
        self.conversion = {
            "agent":"adpmod",
            "nn":"compmod",
            "npadvmod":"nmod",
            "number":"num",
            "pcomp":"adpcomp",
            "pobj":"adpobj",
            "possessive":"poss",
            "preconj":"cc",
            "predet":"det",
            "prep":"adpmod",
            "quantmod":"advmod",
            "tmod":"advmod",
        }
        self.data = self.__read_data(type_split_data)
        self.word_tokenizers = word_tokenizers
        self.deps_tokenizer = deps_tokenizer 
        self.pads = [tokenizer.vocab["[PAD]"] for tokenizer in self.word_tokenizers]
        self.maxlen = maxlen

    def __conv_dep(self, deps):
        res = []
        for dep in deps:
            if dep[0] == "root": continue
            temp1 = f"{dep[0]}_{dep[1][1]}"
            temp2 = f"{dep[0]}I_{dep[1][0]}"
            res.append(temp1)
            res.append(temp2)
        
        temp = {}
        for s in res:
            dep, word = s.split("_")
            if word not in temp:
                temp[word] = []
            temp[word].append(dep)
        return temp
        

    def __read_deps(self, type_split_data):
        f = open(f"resources/dep/sst-{type_split_data}.txt")
        content = f.read()
        deps = content.split("\n\n")[:-1]
        deps = [el.split("\n") for el in deps]
        deps_conv = [None for i in range(len(deps))]
        
        for i in range(len(deps)):
            temp = []
            for el in deps[i]:
                splitted = el.split("(")
                try:
                    splitted[1] = splitted[1][:-1].split(",")
                    splitted[0] = splitted[0].replace("_", ":")
                    
                    if self.conv:
                        k = splitted[0].split(":")[0]
                        if k in self.conversion:
                            t = splitted[0].split(":")
                            t[0] = self.conversion[t[0]]
                            splitted[0] = ":".join(t)

                    splitted[1][1] = splitted[1][1][1:]
                    splitted[1] = [a.split("-")[0] for a in splitted[1]]
                    temp.append(splitted)
                except:
                    print(splitted)
                    raise Exception
            deps[i] = temp
            deps_conv[i] = self.__conv_dep(temp)
        return deps_conv
        
    def __read_data(self, type_split_data):
        f = open(f"data/SST-2/{type_split_data}.txt", "r")
        content = f.read()
        lines = [line.split() for line in content.split("\n")[:-1]]

        sentences, labels = [], []
        for line in lines:
            labels.append(int(line[0]))
            sentences.append(" ".join(line[1:]))
        
        return {
            "text-bert": sentences,
            "text-deps": sentences,
            "deps": self.__read_deps(type_split_data),
            "label": labels,
        }

    def __len__(self):
        return len(self.data["text-bert"])
    
    def tokenize_word(self, text_bert, text_deps, maxlen=512):
        encoded = [self.word_tokenizers[0].encode(text_bert), self.word_tokenizers[1].encode(text_deps)]
        for i in range(len(encoded)):
            len_pad = maxlen - len(encoded[i])
            if len_pad <= 0:
                encoded[i] = encoded[i][:maxlen]
            pad = [self.pads[i]] * len_pad
            encoded[i] += pad
        return torch.from_numpy(np.array(encoded))
    
    def tokenize_deps(self, text, deps, pad=16, maxlen=512):
        tokens = text.split()
        temp = []
        for token in tokens:
            if token not in deps:
                temp.append(["[PAD]" for _ in range(pad)])
                continue
            temp2 = []
            for dep in deps[token]:
                temp2.append(f"{dep}_{token}")
            if len(temp2) > pad:
                temp.append(temp2[:pad])
            else:
                temp.append(temp2 + ["[PAD]" for _ in range(pad - len(temp2))])
        pad_len = maxlen - len(temp)
        temp += [["[PAD]" for _ in range(pad)] for _ in range(pad_len)]
        
        unk_id = self.deps_tokenizer.vocab["[UNK]"]
        for i in range(len(temp)):
            for j in range(len(temp[i])):
                temp[i][j] = self.deps_tokenizer.vocab.get(temp[i][j], unk_id)
        return torch.from_numpy(np.array(temp))
    
    def __getitem__(self, idx):
        text_bert, text_deps = self.data["text-bert"][idx], self.data["text-deps"][idx] 
        deps, y = self.data["deps"][idx], torch.from_numpy(np.array(self.data["label"][idx]))
        return self.tokenize_word(text_bert, text_deps), self.tokenize_deps(text_deps, deps), y