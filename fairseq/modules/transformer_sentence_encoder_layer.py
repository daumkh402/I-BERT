# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch
import torch.nn as nn

from fairseq import utils
from fairseq.modules import (
    LayerNorm,
    MultiheadAttention,
)
from fairseq.modules.quant_noise import quant_noise
from fairseq.modules.fairseq_dropout import FairseqDropout

######
import pdb
from fairseq.modules.mygelu.mygelu import MyGelu
from fairseq.modules.mylayernorm.mylayernorm import MyLayernorm
######

class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = 'relu',
        export: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        init_fn: Callable = None,
    ) -> None:
        super().__init__()

        if init_fn is not None:
            init_fn()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.activation_dropout_module = FairseqDropout(activation_dropout, module_name=self.__class__.__name__)

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = self.build_self_attention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embedding_dim,
            ffn_embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )
        self.fc2 = self.build_fc2(
            ffn_embedding_dim,
            self.embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)


        ######
        self.gelu_type = 'normal'
        self.mygelu = None
        self.layernorm_type = 'normal'
        self.mylayernorm = None
        self.my = True
        #######

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), q_noise, qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), q_noise, qn_block_size
        )

    def build_self_attention(
        self,
        embed_dim,
        num_attention_heads,
        dropout,
        self_attention,
        q_noise,
        qn_block_size,
    ):
        return MultiheadAttention(
            embed_dim,
            num_attention_heads,
            dropout=dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        # import pdb; pdb.set_trace()

        if self.gelu_type == 'normal' or self.gelu_type is None:
            x = self.activation_fn(self.fc1(x))
            pdb.set_trace()
        else:
            x = self.mygelu(self.fc1(x))

        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x

        if self.layernorm_type == 'normal'  or self.layernorm_type is None:
            x = self.final_layer_norm(x)
        else:
            x = self.mylayernorm(x)

        return x, attn


    def set_softmax(self, softmax_type, data_dict):
        self.self_attn.set_softmax(softmax_type, data_dict)

    def set_gelu(self, gelu_type, data_dict=None):
            self.gelu_type = gelu_type

            if self.gelu_type == 'nn':
                gelu_dict = data_dict
                self.mygelu = MyGelu(gelu_type, gelu_dict)

            elif self.gelu_type == 'lut':
                gelu_dict = data_dict['lut']
                self.mygelu = MyGelu(gelu_type, gelu_dict)

            elif self.gelu_type == 'ibert':
                self.mygelu = MyGelu(gelu_type)
            else :
                raise Exception('This gelu type is not supported')

    def set_layernorm(self, layernorm_type, data_dict=None):
            self.layernorm_type = layernorm_type

            if self.layernorm_type == 'nn':
                layernorm_dict = data_dict
                self.mylayernorm = MyLayernorm(layernorm_type, layernorm_dict)

            elif self.layernorm_type == 'lut':
                layernorm_dict = data_dict['lut']
                self.mylayernorm = MyLayernorm(layernorm_type, layernorm_dict)

            elif self.layernorm_type == 'ibert':
                self.mylayernorm = MyLayernorm(layernorm_type)
            else :
                raise Exception('This gelu type is not supported')


        