
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

        self.layer_norm_2 = nn.LayerNorm(self.LayerNorm.normalized_shape)
        self.coef_learn = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.coef_extra = nn.Parameter(torch.tensor(0.5), requires_grad=True)

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
        embeddings = self.coef_learn * embeddings + \
            self.coef_extra * self.layer_norm_2(self.extra_embeddings(input_ids))

        embeddings = self.dropout(embeddings)
        return embeddings