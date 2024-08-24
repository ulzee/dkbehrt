
import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.models.bert.modeling_bert import BertEmbeddings

class InjectEmbeddings(BertEmbeddings):
    # Modificaton of:
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py

    def __init__(self, config, extra_embeddings_data, keep_training=False):
        super().__init__(config)
        try: assert extra_embeddings_data.shape == self.word_embeddings.weight.shape
        except: raise f'Provided emb size mismatch: {extra_embeddings_data.shape} vs {self.word_embeddings.weight.shape}'
        self.extra_embeddings = nn.Embedding(*extra_embeddings_data.shape, padding_idx=0)
        self.extra_embeddings.weight = nn.Parameter(torch.from_numpy(extra_embeddings_data).float(), requires_grad=keep_training)
        # self.word_embeddings.weight.requires_grad_(False)
        # self.word_embeddings.weight[3:] = torch.from_numpy(extra_embeddings_data).float()[3:]

        edim = self.word_embeddings.weight.shape[1]
        ntokens = len(self.extra_embeddings.weight)
        self.layer_norm_2 = nn.LayerNorm(self.LayerNorm.normalized_shape)
        self.extra_emb_mlp = nn.Sequential(
            nn.Linear(edim, 1024),
            nn.ReLU(),
            nn.Linear(1024, edim)
        )
        self.coef_learn = nn.Parameter(torch.tensor([0.5]*ntokens), requires_grad=True)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        assert token_type_ids is not None # NOTE: not supported

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

            # # NOTE: new
            # inputs_embeds += self.extra_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)

        # NOTE: new
        # NOTE: probably better to have a per token coef
        # NOTE: even better, allow to be attended differently per token
        #  External tokens may need to be passed separately to use attention
        #  They can share positional embedding to indicate that they are the same observation?
        coefs = torch.sigmoid(self.coef_learn)
        embeddings = coefs[input_ids].unsqueeze(-1) * embeddings + \
            (1-coefs)[input_ids].unsqueeze(-1) * self.layer_norm_2(self.extra_emb_mlp(self.extra_embeddings(input_ids)))

        embeddings = self.dropout(embeddings)
        return embeddings

class NonTorchVariableHolder:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class CovariateAddEmbeddings(BertEmbeddings):

    def forward(
        self,
        **kwargs
    ) -> torch.Tensor:

        # covariates will be passed as position_ids
        #  for best compatibility with hugg
        if 'position_ids' in kwargs:
            self.covariates = kwargs['position_ids']
            del kwargs['position_ids']

        out = BertEmbeddings.forward(self, **kwargs)

        if hasattr(self, 'covariates'):
            # apply covariates signal to processed embeddings
            out += self.covariates.unsqueeze(-1)

        return out

class KeepInputEmbeddings(CovariateAddEmbeddings):
    # A way to access the input_ids later by other modules

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_ids = NonTorchVariableHolder()

    def forward(
        self,
        **kwargs
    ) -> torch.Tensor:

        out = CovariateAddEmbeddings.forward(self, **kwargs)

        self.input_ids.input_ids = kwargs['input_ids']

        return out