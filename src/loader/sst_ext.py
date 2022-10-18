import torch
from torch.utils.data import Dataset

import numpy as np

class SSTDepsExtDataset(Dataset):
    def __init__(self, type_split_data, word_tokenizers, deps_tokenizer, conv=False, maxlen=512):
        self.conv = conv
        self.data = self.__read_data(type_split_data)
        self.word_tokenizers = word_tokenizers
        self.deps_tokenizer = deps_tokenizer 
        self.pads = [tokenizer.vocab["[PAD]"] for tokenizer in self.word_tokenizers]
        self.maxlen = maxlen

    def __read_deps(self, type_split_data):
        f = open(f"resources/dep2/sst-{type_split_data}.txt")
        content = f.read()
        data = content.split("\n\n")[:-1]
        data = [line.split("##@@##") for line in content.split("\n\n")]
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j] = data[i][j].split(" ")
                data[i][j][0] = int(data[i][j][0])
        
        return data
        
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
    
    def tokenize_deps(self, deps, pad=16, maxlen=512):
        temp = [["[PAD]" for _ in range(pad)] for _ in range(maxlen)]
        for i, dep in enumerate(deps):
            if i >= maxlen:
                break
            idx = dep[0]
            for j, dep_ in enumerate(dep[1:]):
                if j >= pad:
                    break
                temp[idx][j] = dep_
        
        unk_id = self.deps_tokenizer.vocab["[UNK]"]
        for i in range(len(temp)):
            for j in range(len(temp[i])):
                temp[i][j] = self.deps_tokenizer.vocab.get(temp[i][j], unk_id)
        return torch.from_numpy(np.array(temp))
    
    def __getitem__(self, idx):
        text_bert, text_deps = self.data["text-bert"][idx], self.data["text-deps"][idx] 
        deps, y = self.data["deps"][idx], torch.from_numpy(np.array(self.data["label"][idx]))
        return self.tokenize_word(text_bert, text_deps), self.tokenize_deps(deps), y