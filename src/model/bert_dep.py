import torch
import torch.nn as nn
import pickle
import numpy as np
from transformers.models.bert.modeling_bert import BertEmbeddings, BertModel

class BertDepEmbeddings(BertEmbeddings):
    def __init__(self, config, dep_emb_dim=300, maxlen=512):
        super(BertDepEmbeddings, self).__init__(config)
        self.dep_embeddings = None
        self.word_dep_embeddings = None
        self.fc = nn.Linear(dep_emb_dim, config.hidden_size)
        self.maxlen = maxlen
        self.skip_ids = set([926907, 926909])
        self.device = "cuda:3"

    def create_embeddings(self, path_word, path_dep):
        print(f"Loading Pretrained Embedding from {path_word} and {path_dep}")
        state_dict_word = pickle.load(open(path_word, "rb"))
        state_dict_dep = pickle.load(open(path_dep, "rb"))
        self.word_dep_embeddings = nn.Embedding.from_pretrained(state_dict_word)
        self.dep_embeddings = nn.Embedding.from_pretrained(state_dict_dep)

        self.word_dep_embeddings.weight.requires_grad = False
        self.dep_embeddings.weight.requires_grad = False

    def add_special_ids(self, word_pad_id, unk_id, deps_pad_id):
        self.word_pad_id = word_pad_id
        self.skip_ids = set([deps_pad_id, unk_id])

    def add_weights(self, word_weight, dep_weight):
        self.word_weight = word_weight
        self.dep_weight = dep_weight
        print(self.word_weight, self.dep_weight)

    def __mask(self, ids):
        mask = []
        for id_ in ids:
            if int(id_.cpu().detach().numpy()) not in self.skip_ids:
                mask.append(1)
            else:
                mask.append(0)
        return torch.from_numpy(np.array(mask).reshape(-1, 1)).to(self.device)

    def forward(
        self, 
        input_ids, 
        deps_ids, 
        token_type_ids=None, 
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0
    ):
        dep_input_ids = input_ids[:, 1, :]
        input_ids = input_ids[:, 0, :]

        # Standard BERT Embeddings
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # Dep Embeddings
        bert_pad_loc = self.maxlen - (input_ids == 0).sum(axis=1)
        word_dep_embeddings = self.word_dep_embeddings(dep_input_ids)
        dep_embeddings = self.dep_embeddings(deps_ids)

        for i, batch_deps in enumerate(deps_ids):
            for j, deps_id in enumerate(batch_deps):
                mask = self.__mask(deps_id)
                sum_ = mask.sum()
                if sum_:
                    taken_emb = (dep_embeddings[i][j] * mask).sum(axis=0)
                    word_dep_embeddings[i][j] = self.word_weight * word_dep_embeddings[i][j] + self.dep_weight * taken_emb / sum_

        dep_pad_loc = self.maxlen - (dep_input_ids == self.word_pad_id).sum(axis=1)
        fc_dep = self.fc(word_dep_embeddings.float())
        
        for i, (bert_pad, dep_pad) in enumerate(zip(bert_pad_loc, dep_pad_loc)):
            if bert_pad == self.maxlen:
                continue
                
            left = self.maxlen - bert_pad - dep_pad
            if left <= 0:
                embeddings[i] = torch.concat((embeddings[i, :bert_pad, :], fc_dep[i, :left, :]))[:512]
            else:
                embeddings[i] = torch.concat((embeddings[i, :bert_pad, :], fc_dep[i, :dep_pad, :], embeddings[i, bert_pad:left+bert_pad, :]))
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertDependencyModelEnc(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super(BertDependencyModelEnc, self).__init__(config, add_pooling_layer)
        self.embeddings = BertDepEmbeddings(config)
        self.clf = nn.Linear(config.hidden_size, 2)
        self.act = nn.Sigmoid()
        
    def create_embeddings(self, path_word, path_dep):
        return self.embeddings.create_embeddings(path_word, path_dep)

    def add_device(self, device):
        self.embeddings.device = device
        
    def add_special_ids(self, word_pad_id, unk_id, deps_pad_id):
        self.embeddings.add_special_ids(word_pad_id, unk_id, deps_pad_id)

    def add_weights(self, word_weight, dep_weight):
        self.embeddings.add_weights(word_weight, dep_weight)

    def forward(
        self,
        word_ids, 
        deps_ids,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        encoder_hidden_states = None,
        encoder_attention_mask = None,
        past_key_values = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        input_ids = word_ids[:, 0, :]
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device=device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            word_ids,
            deps_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        pred = self.clf(pooled_output)

        return pred