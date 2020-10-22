""" Hierarchical Transformer class
24.09.2020 - Yash Bonde

NOTATION
========
B: batch_size
T: time_steps
H: num_heads
E: embedding_dim
DH: dim_per_head
N: n_weather_stations
G: global_embedding_dim
F: n_features
"""

from tabulate import tabulate

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import optimizer

from transformers.modeling_utils import find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers.activations import ACT2FN
from transformers.modeling_transfo_xl import (
    AdaptiveEmbedding,
    RelPartialLearnableDecoderLayer,
    PositionalEmbedding,
    TransfoXLModelOutput
)

from optimizer import AdaBelief

"""
UTILS
=====

Functions and classes
"""

# this is giant because requires managing both for GPT-2 and Transformer-XL configurations


class HeirarchicalTransformerConfig:
    vocab_size = 267735
    cutoffs = [20000, 40000, 200000]
    div_val = 4
    pre_lnorm = True
    mem_len = 1600
    clamp_len = 1000
    same_length = True
    proj_share_all_but_first = True
    sample_softmax = -1
    adaptive = True
    untie_r = True

    n_global = None
    num_nodes = None
    num_features = 8
    location_features = 3

    n_embd = 768
    n_layer = 12
    n_head = 12
    activation_function = "gelu_new"
    resid_pdrop = 0.1
    embd_pdrop = 0.1
    attn_pdrop = 0.1
    layer_norm_epsilon = 1e-5
    initializer_range = 0.02
    summary_type = "cls_index"
    summary_use_proj = True
    summary_activation = None
    summary_proj_to_labels = True
    summary_first_dropout = 0.1
    bos_token_id = 50256
    eos_token_id = 50256

    def __init__(self, **kwargs):
        self.attrs = []
        for k, v in kwargs.items():
            setattr(self, k, v)
            self.attrs.append(k)

        if "maxlen" not in self.attrs:
            raise ValueError("Need to provide argument `maxlen`")
        if "n_global" not in self.attrs:
            raise ValueError("Need to provide argument `n_global`")
        if "num_nodes" not in self.attrs:
            raise ValueError("Need to provide argument `num_nodes`")
        if "num_features" not in self.attrs:
            raise ValueError("Need to provide argument `num_features`")
        if "location_features" not in self.attrs:
            raise ValueError("Need to provide argument `location_features`")

        self.config_to_hf_format()

    def config_to_hf_format(self):
        # following are used in the globalmodeller to n_global is n_embd here
        self.d_model = self.n_global
        self.d_embed = self.n_global
        self.d_head = self.n_global // self.n_head
        self.d_inner = self.n_global * 4
        self.dropout = self.attn_pdrop
        self.dropatt = self.attn_pdrop

        # why have two? this is just annoying
        self.n_positions = self.maxlen
        self.n_ctx = self.maxlen

    def __repr__(self):
        return "========== TRAINER CONFIGURATION ==========\n" + \
            tabulate([(k, getattr(self, k)) for k in sorted(list(set([
                "vocab_size",
                "cutoffs",
                "div_val",
                "pre_lnorm",
                "mem_len",
                "clamp_len",
                "same_length",
                "proj_share_all_but_first",
                "num_nodes",
                "sample_softmax",
                "adaptive",
                "untie_r",
                "n_embd",
                "n_layer",
                "n_head",
                "activation_function",
                "resid_pdrop",
                "embd_pdrop",
                "attn_pdrop",
                "layer_norm_epsilon",
                "initializer_range",
                "summary_type",
                "summary_use_proj",
                "summary_activation",
                "summary_proj_to_labels",
                "summary_first_dropout",
                "bos_token_id",
                "eos_token_id",
                "d_model",
                "d_embed",
                "d_head",
                "d_inner",
                "dropout",
                "dropatt",
                "n_positions",
                "n_ctx",
            ] + self.attrs))
            )], ["key", "value"], tablefmt="psql")


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    Basically works like a linear layer but the weights are transposed.
    Args:
        nf (:obj:`int`): The number of output features.
        nx (:obj:`int`): The number of input features.
    """

    def __init__(self, nf, nx, bias=True):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))
        self.beta = int(bias)  # if False beta automatically becomes 0

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(
            input=self.bias,
            mat1=x.reshape(-1, x.size(-1)),
            mat2=self.weight,
            beta=self.beta,
            alpha=1
        )
        x = x.view(*size_out)
        return x


class NodeToGlobalMaxPool(nn.MaxPool1d):
    def forward(self, input):
        n, w, c, h = input.size()
        input = input.view(n*w, c, h).permute(0, 2, 1)
        pooled = F.max_pool1d(input, self.kernel_size, self.stride,
                              self.padding, self.dilation, self.ceil_mode,
                              self.return_indices)
        _, _, c = pooled.size()
        pooled = pooled.permute(0, 2, 1)
        return pooled.view(n, c, w, h).squeeze(1)


"""
GPT TRANSFORMER BLOCKS
######################
https://huggingface.co/transformers/_modules/transformers/modeling_gpt2.html

Simple transformer blocks, stripped to requirements then added extra global layers.
"""


class NodeAttention(nn.Module):
    """ trimmed and modified version of Attention from `transformers.modeling_gpt2` to be used for Nodes """

    def __init__(self, config):
        super().__init__()

        hidden_dim = config.n_embd
        n_ctx = config.n_ctx

        assert hidden_dim % config.n_head == 0
        self.register_buffer(
            "bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(
                1, 1, n_ctx, n_ctx)
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.n_head = config.n_head
        self.split_size = hidden_dim

        # layers
        self.c_attn = Conv1D(3 * hidden_dim, hidden_dim)
        self.c_proj = Conv1D(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_head, self.split_size // self.n_head, self.pruned_heads
        )
        index_attn = torch.cat(
            [index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * \
            (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, edge_matrix, attention_mask=None, output_attentions=False):
        w = torch.matmul(q, k)
        w = w / (float(v.size(-1)) ** 0.5)

        if attention_mask is not None:
            B, T, N, _ = attention_mask.size()
            a2 = attention_mask.view(B*T, 1, N, N)
            # print(w.size(), attention_mask.size(), a2.size())
            # Apply the attention mask
            w = w + a2

        # add the edge matrix
        w += edge_matrix

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        outputs = [torch.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        # if k: (batch, head, head_features, seq_length)
        # else: (batch, head, seq_length, head_features)
        return x.permute(0, 2, 3, 1) if k else x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        edge_matrix,
        attention_mask=None,
        output_attentions=False,
        **kwargs
    ):
        # need to flatten for passing to the below layers
        B, T, N, E = hidden_states.size()
        hidden_states = hidden_states.view(B*T, N, E)

        query, key, value = self.c_attn(
            hidden_states).split(self.split_size, dim=2)

        query = self.split_heads(query)  # [B*T, H, N, E]
        key = self.split_heads(key, k=True)  # [B*T, H, E, N]
        value = self.split_heads(value)  # [B*T, H, N, E]

        present = (None,)
        attn_outputs = self._attn(
            query, key, value, edge_matrix, attention_mask, output_attentions)
        a = attn_outputs[0]

        a = self.merge_heads(a)  # [B*T, N, E]
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        # reshape to 4D
        a = a.view(B, T, N, E)  # [B, T, N, E]

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)


class MLP(nn.Module):
    def __init__(self, config, inner_dim):  # in MLP: mid=3072 (4 * n_embd)
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(inner_dim, nx)
        self.c_proj = Conv1D(nx, inner_dim)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class NodeEncoderBlock(nn.Module):
    """ Standard Full MHA + MLP for node --> global"""

    def __init__(self, config):
        super().__init__()
        E = config.n_embd
        G = config.n_global
        self.att = NodeAttention(config)
        self.ln1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config, E * 4)
        self.ln2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.n2g_pool = NodeToGlobalMaxPool(config.num_nodes)
        self.n2g = Conv1D(G, E)
        self.n2g_act = ACT2FN[config.activation_function]
        self.ln3 = nn.LayerNorm(G, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states,
        global_states,
        edge_matrix,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        attn_outputs = self.att(
            self.ln1(hidden_states),
            edge_matrix,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        # residual connection
        hidden_states = attn_output + hidden_states

        feed_forward_hidden_states = self.mlp(self.ln2(hidden_states))

        # residual connection
        hidden_states = hidden_states + feed_forward_hidden_states

        # to global
        n2g = self.n2g(hidden_states)  # [B, T, N, G]
        n2g = self.n2g_pool(self.n2g_act(n2g))  # [B, T, G]
        global_states = self.ln3(global_states + n2g)  # [B, T, G]

        outputs = [hidden_states, global_states] + outputs
        return outputs


class NodeDecoderBlock(nn.Module):
    """ Standard Full MHA + MLP for global --> node"""

    def __init__(self, config):
        super().__init__()
        E = config.n_embd
        G = config.n_global
        self.att = NodeAttention(config)
        self.ln1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config, E * 4)
        self.ln2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.g2n = Conv1D(E, G)
        self.g2n_act = ACT2FN[config.activation_function]

    def forward(
        self,
        hidden_states,
        global_states,
        edge_matrix,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        global_to_hidden = self.g2n_act(self.g2n(global_states))  # [B, T, E]
        global_to_hidden = global_to_hidden.unsqueeze(2)  # [B, T, 1, E]
        hidden_states = self.ln1(hidden_states + global_to_hidden)

        attn_outputs = self.att(
            hidden_states,
            edge_matrix,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        # residual connection
        hidden_states = attn_output + hidden_states
        feed_forward_hidden_states = self.mlp(self.ln2(hidden_states))

        # residual connection
        hidden_states = hidden_states + feed_forward_hidden_states

        outputs = [hidden_states] + outputs
        return outputs


"""
TRANSFORMER-XL RECURRENT BLOCKS
###############################
https://huggingface.co/transformers/_modules/transformers/modeling_transfo_xl.html

Re-write the main model.
"""

# AdaptiveEmbedding, RelPartialLearnableDecoderLayer


class GlobalTransfoXLModeller(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.d_model % 3 == 0

        self.config = config

        self.n_token = config.vocab_size
        self.d_embed = config.d_embed
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.n_layer = config.n_layer
        self.mem_len = config.mem_len
        self.same_length = config.same_length
        self.clamp_len = config.clamp_len

        """TimeEmbedding
        'mo'  : Month
        'da'  : The day (0-31)
        'hr'  : The hour (0-23)
        """
        dim_for_time = config.d_embed // 3
        self.mo_embedding = AdaptiveEmbedding(
            n_token=12,
            d_embed=dim_for_time,
            d_proj=dim_for_time,
            cutoffs=config.cutoffs,
            div_val=1,
            sample_softmax=False
        )
        self.da_embedding = AdaptiveEmbedding(
            n_token=31,
            d_embed=dim_for_time,
            d_proj=dim_for_time,
            cutoffs=config.cutoffs,
            div_val=1,
            sample_softmax=False
        )
        self.hr_embedding = AdaptiveEmbedding(
            n_token=24,
            d_embed=dim_for_time,
            d_proj=dim_for_time,
            cutoffs=config.cutoffs,
            div_val=1,
            sample_softmax=False
        )

        self.position_embedding = PositionalEmbedding(config.d_model)

        self.layers = nn.ModuleList([
            RelPartialLearnableDecoderLayer(
                n_head=config.n_head,
                d_model=config.d_model,
                d_head=config.d_head,
                d_inner=config.d_inner,
                dropout=config.dropout,
                dropatt=config.dropatt,
                pre_lnorm=config.pre_lnorm,
                r_r_bias=None,
                r_w_bias=None,
                layer_norm_epsilon=config.layer_norm_epsilon
            ), ] * config.n_layer)

        self.drop = nn.Dropout(config.embd_pdrop)

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, new_embeddings):
        self.embedding = new_embeddings

    def backward_compatible(self):
        self.sample_softmax = -1

    def reset_memory_length(self, mem_len):
        self.mem_len = mem_len

    def init_mems(self, bsz):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer):
                empty = torch.zeros(
                    self.mem_len, bsz, self.config.d_model, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, mlen, qlen):
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hids) == len(mems), "len(hids) != len(mems)"

        # There are `mlen + qlen` steps that can be cached into mems
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def forward(
        self,
        global_states,
        month_ids,
        day_ids,
        hour_ids,
        mems=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        month_ids, day_ids, hour_ids: each should have shape [B, T]
        """
        #
        month_ids = month_ids.transpose(0, 1).contiguous()
        day_ids = day_ids.transpose(0, 1).contiguous()
        hour_ids = hour_ids.transpose(0, 1).contiguous()
        global_states = global_states.transpose(0, 1).contiguous()
        qlen, bsz = month_ids.size()

        if mems is None:
            mems = self.init_mems(bsz)

        # print(day_ids.max(), hour_ids.max())

        # perform time embeddings
        time_embedding = torch.cat([
            self.mo_embedding(month_ids),
            self.da_embedding(day_ids),
            self.hr_embedding(hour_ids),
        ], dim=-1)

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen

        if self.same_length:
            all_ones = time_embedding.new_ones((qlen, klen), dtype=torch.uint8)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1 + mlen) +
                             torch.tril(all_ones, -mask_shift_len))[:, :, None]
        else:
            dec_attn_mask = torch.triu(time_embedding.new_ones(
                (qlen, klen), dtype=torch.uint8), diagonal=1 + mlen)[:, :, None]

        hids = []
        attentions = [] if output_attentions else None
        pos_seq = torch.arange(
            klen-1, -1, -1.0, device=time_embedding.device, dtype=time_embedding.dtype)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.position_embedding(pos_seq)

        # print("---> global_states:", global_states.size())
        # print("---> time_embedding:", time_embedding.size())

        # add time embedding with global_state
        core_out = global_states + time_embedding
        core_out = self.drop(core_out)
        pos_emb = self.drop(pos_emb)

        for i, layer in enumerate(self.layers):
            hids.append(core_out)
            mems_i = None if mems is None else mems[i]
            layer_outputs = layer(
                dec_inp=core_out,
                r=pos_emb,
                dec_attn_mask=dec_attn_mask,
                mems=mems_i,
                output_attentions=output_attentions
            )
            core_out = layer_outputs[0]
            if output_attentions:
                attentions.append(layer_outputs[1])

        core_out = self.drop(core_out)
        new_mems = self._update_mems(hids, mems, mlen, qlen)

        if output_hidden_states:
            # add last layer and transpose it to [B, T, E]
            hids.append(core_out)
            hids = tuple(t.transpose(0, 1) for t in hids)
        else:
            hids = None

        if output_attentions:
            # transpose to standard
            attentions = tuple(t.permute(2, 3, 0, 1) for t in attentions)

        core_out = core_out.transpose(0, 1).contiguous()  # [B, T, G]

        if not return_dict:
            return tuple(v for v in [core_out, new_mems, hids, attentions] if v is not None)

        return TransfoXLModelOutput(
            last_hidden_state=core_out,
            mems=new_mems,
            hidden_states=hids,
            attentions=attentions,
        )


class HeirarchicalTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # assertions
        assert config.n_embd % config.num_features == 0, f"inputs require dim to be divisible by {config.num_features}"
        assert config.n_embd % config.location_features == 0, f"location requires divisible by {config.location_features}"

        # WeatherMetricLinear + Location Embedding + Time embedding
        """WeatherMetricLinear
        'prcp': Amount of precipitation in millimetres (last hour)
        'stp' : Air pressure for the hour in hPa to tenths (instant)
        'gbrd': Solar radiation KJ/m2
        'temp': Air temperature (instant) in celsius degrees
        'dewp': Dew point temperature (instant) in celsius degrees
        'hmdy': Relative humid in % (instant)
        'wdsp': Wind speed in metres per second
        'wdct': Wind direction in radius degrees (0-360)
        """

        # rather than having different linear systems, have a
        self.weather_metric_linear = Conv1D(config.n_embd, config.num_features, bias=False)

        """LocationEmbedding
        'elvt': Elevation
        'lat' : Latitude
        'lon' : Longitude
        """
        self.location_embedding = Conv1D(config.n_embd, config.location_features, bias=False)

        # encoder - temporal - decoder blocks
        self.node_encoder = nn.ModuleList([NodeEncoderBlock(config), ] * config.n_layer)
        self.temporal_encoder = GlobalTransfoXLModeller(config)
        self.node_decoder = nn.ModuleList([NodeDecoderBlock(config), ] * config.n_layer)
        self.ln1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.n_global, eps=config.layer_norm_epsilon)

        # WeatherMetricHead
        self.weather_metric_head = Conv1D(config.num_features, config.n_embd)

        self.n_global = config.n_global

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, Conv1D, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                # print("##", fpn)

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        optimizer = AdaBelief(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    @property
    def num_params(self):
        return sum(dict((p.data_ptr(), p.numel()) for p in self.parameters()).values())

    def forward(
        self,
        input,
        locations,
        edge_matrix,
        month_ids,
        day_ids,
        hour_ids,
        node_mask,
        mems=None,
        output_attentions=False,
        get_loss=False
    ):

        # print(input.size(), locations.size())

        # get the input to node encoder
        weather_station_state = self.weather_metric_linear(input)
        loc_emb = self.location_embedding(locations)
        # B = loc_emb.size(0)
        # loc_emb = loc_emb.reshape(B, 1, *loc_emb.size()[1:])
        # print(weather_station_state.size(), loc_emb.size())

        hidden_states = weather_station_state + loc_emb
        # print(hidden_states.size())

        B, T, N, E = hidden_states.size()
        global_states = torch.zeros(B, T, self.n_global)

        # print("INPUT (L0)")
        # print("hidden_states:", hidden_states.size())
        # print("global_states:", global_states.size())

        # create the node_mask for cases where 
        node_attention_mask = torch.zeros(*list(node_mask.size()), list(node_mask.size())[-1])
        for i in range(node_mask.size(0)):
            for j in range(node_mask.size(1)):
                idx = torch.masked_select(
                    torch.arange(len(node_mask[i, j])),
                    node_mask[i, j] == 0
                ).tolist()
                if idx:
                    for k in idx:
                        node_attention_mask[i, j, k] = 1e-6
                        node_attention_mask[i, j, :, k] = 1e-6
        # del node_mask

        attentions = []

        # pass through layers --> first the encoder layers
        enc_attentions = []
        for i, ne_layer in enumerate(self.node_encoder):
            layer_outputs = ne_layer(
                hidden_states=hidden_states,
                global_states=global_states,
                edge_matrix=edge_matrix,
                attention_mask=node_attention_mask,
                output_attentions=output_attentions
            )
            hidden_states, global_states = layer_outputs[:2]
            enc_attentions.append(layer_outputs[2])
        global_states_prime = global_states
        attentions.append(enc_attentions)

        # print("AFTER ENCODER (L1)")
        # print("hidden_states:", hidden_states.size())
        # print("global_states_prime:", global_states_prime.size())

        # then through the global context layers
        temporal_output = self.temporal_encoder(
            global_states=global_states_prime,
            month_ids=month_ids,
            day_ids=day_ids,
            hour_ids=hour_ids,
            mems=mems,
            return_dict=True,
            output_attentions=output_attentions
        )
        global_states_prime2 = temporal_output.last_hidden_state
        attentions.append(temporal_output.attentions)

        # print("AFTER TEMPORAL (L2)")
        # print("global_states_prime2:", global_states_prime2.size())

        # and finally through the node decoder layers
        dec_attentions = []
        for i, nd_layer in enumerate(self.node_decoder):
            layer_outputs = nd_layer(
                hidden_states=hidden_states,
                global_states=global_states_prime2,
                edge_matrix=edge_matrix,
                output_attentions=output_attentions
            )
            hidden_states = layer_outputs[0]  # no global state
            dec_attentions.append(layer_outputs[1])
        node_state_prime2 = hidden_states
        attentions.append(dec_attentions)

        # print("AFTER DECODER (L3)")
        # print("node_state_prime2:", node_state_prime2.size())

        node_states = self.ln1(node_state_prime2)
        global_states = self.ln2(global_states_prime2)

        # get the output values
        out = self.weather_metric_head(node_states)  # [B, T, N, F]
        F = out.size(-1)

        output = [out, temporal_output.mems]

        if get_loss:
            # shift the values and calculate MSE loss
            logits = out[:, :-1, :, :].contiguous().view(-1, F)
            targets = input[:, 1:, :, :].contiguous().view(-1, F)
            node_mask = node_mask[:, 1:, :].contiguous().view(-1, 1)

            lossfn = nn.MSELoss(reduction="none")
            loss = lossfn(logits, targets)

            # only select those losses where we have the data for usage
            masked_loss = torch.masked_select(loss, node_mask == 1)
            output.append(masked_loss.mean())

        if output_attentions:
            output.append(attentions)

        return output


# if __name__ == "__main__":
#     B = 3
#     N = 13
#     T = 5
#     config = HeirarchicalTransformerConfig(
#         n_embd=144,
#         n_global=288,
#         maxlen=T,
#         n_head=8,
#         n_layer=12,
#         num_nodes=N,
#         num_features=8,
#         location_features=3
#     )

#     edge_matrix = torch.randn(N, N)

#     print("----- FullTester -----")
#     month_ids = torch.empty(B, config.maxlen).random_(12).long()
#     day_ids = torch.empty(B, config.maxlen).random_(31).long()
#     hour_ids = torch.empty(B, config.maxlen).random_(24).long()

#     sample_data = {
#         "input": torch.randn(B, T, N, config.num_features),
#         "locations": torch.randn(B, T, N, config.location_features),
#         "edge_matrix": edge_matrix,
#         "month_ids": month_ids,
#         "day_ids": day_ids,
#         "hour_ids": hour_ids,
#         "attention_mask": None,
#     }
#     # print(sample_data)

#     model = HeirarchicalTransformer(config)
#     print(model.num_params)

#     out = model(**sample_data)
#     print("Logits:", out[0].size())
#     print("Mems:", [x.shape for x in out[1]])

#     out = model(**sample_data, get_loss=True)
#     print("Logits:", out[0].size())
#     print("Mems:", [x.shape for x in out[1]])
#     print("Loss:", out[2])

#     for i in range(T*2):
#         sample_data = {
#             "input": torch.randn(B, T, N, config.num_features),
#             "locations": torch.randn(B, T, N, config.location_features),
#             "edge_matrix": edge_matrix,
#             "month_ids": torch.empty(B, config.maxlen).random_(12).long(),
#             "day_ids": torch.empty(B, config.maxlen).random_(31).long(),
#             "hour_ids": torch.empty(B, config.maxlen).random_(24).long(),
#             "attention_mask": None,
#         }
#         mems = None
#         out = model(**sample_data, mems=mems, output_attentions=False)
#         logits, mems = out

#         print(f"------ {i} ------")
#         print(logits.size())
#         print(mems[0][-3:, :3, :3])
