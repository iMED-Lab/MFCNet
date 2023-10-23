import torch
from torchvision import models
import torch.nn as nn
from mmcv.cnn import ConvModule
from models.until import NestedTensor, build_position_encoding
from torch.nn import Linear, Dropout, Softmax, LayerNorm
import math
import copy
import models.configs as configs
from torch.nn.init import normal_
from einops import rearrange, reduce, repeat
from torch import Tensor
import torch.nn.functional as F
from torch.nn import init
from functools import partial

CONFIGS = {
    '4_layer': configs.get_4_config(),
    '3_layer': configs.get_3_config(),
    '2_layer': configs.get_2_config(),
    '1_layer': configs.get_1_config(),

}


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ResNet(nn.Module):
    def __init__(self, in_channels=3):
        super(ResNet, self).__init__()
        model = models.resnet34(pretrained=True)
        self.conv_input = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.line = nn.Linear(512, 6)

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        out = []
        y0 = self.conv_input(x)
        out.append(y0)
        y1 = self.layer1(self.maxpool(self.relu(self.bn1(y0))))
        out.append(y1)

        y2 = self.layer2(y1)
        out.append(y2)
        y3 = self.layer3(y2)
        out.append(y3)
        y4 = self.layer4(y3)
        out.append(y4)
        feature = self.gap(y4).view((y4.size(0), -1))
        # print(feature.shape)
        pre = self.line(feature)

        return out, pre


class MEAttention(nn.Module):
    def __init__(self, dim, configs_head=8):
        super(MEAttention, self).__init__()
        self.num_heads = configs_head
        self.coef = 4
        self.query_liner = nn.Linear(dim,
                                     dim * self.coef)
        self.num_heads = self.coef * self.num_heads
        self.k = 256 // self.coef  ## k=64
        self.linear_0 = nn.Linear(dim * self.coef // self.num_heads, self.k)
        self.key_liner = nn.Linear(dim, dim * self.coef)
        # self.value_liner = nn.Linear(self.k, dim * self.coef // self.num_heads)
        self.linear_1 = nn.Linear(self.k, dim * self.coef // self.num_heads)
        self.linear = nn.Linear(1920, 960)

        self.proj = nn.Linear(dim * self.coef, dim)

    def forward(self, src1, src2):
        B1, N1, C1 = src1.shape
        B2, N2, C2 = src2.shape
        query = self.query_liner(src1)

        key = self.key_liner(src2)

        merge = torch.cat([query, key], dim=2)
        attn = self.linear(merge)  # attn = attn.view(B1, N1, self.num_heads, -1).permute(0, 2, 1, 3)  #

        attn11 = attn.view(B1, N1, self.num_heads, -1).permute(0, 2, 1, 3)  #
        attn = self.linear_0(attn11)

        attn = attn.softmax(dim=-2)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))

        x = self.linear_1(attn).permute(0, 2, 1, 3).reshape(B1, N1, -1)

        x = self.proj(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, config, emb_size: int = 1029, num_heads: int = 7, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)

        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)

        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out, att


class EAM_Attention(nn.Module):
    def __init__(self, dim=256, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0
        self.coef = 4
        self.trans_dims = nn.Linear(dim, dim * self.coef)
        self.num_heads = self.num_heads * self.coef
        self.k = 256 // self.coef
        self.linear_0 = nn.Linear(dim * self.coef // self.num_heads, self.k)
        self.linear_1 = nn.Linear(self.k, dim * self.coef // self.num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim * self.coef, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        h = x
        x = x.permute(0, 2, 1)
        B, N, C = x.shape

        x = self.trans_dims(x)  # B, N, C
        x = x.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = self.linear_0(x)

        attn = attn.softmax(dim=-2)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))
        attn = self.attn_drop(attn)
        x = self.linear_1(attn).permute(0, 2, 1, 3).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 2, 1)
        return x, h


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)

        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        # print(hidden_states.shape)
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        # print(mixed_value_layer.shape)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = self.softmax(attention_scores)

        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Attention1(nn.Module):
    def __init__(self, config):
        super(Attention1, self).__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)

        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        # print(hidden_states.shape)
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        # print(mixed_value_layer.shape)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = self.softmax(attention_scores)

        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, mixed_query_layer


class Attention2(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ExternalAttention = ExternalAttention(config)
        self.liner1 = nn.Linear(config.hidden_size * 2, config.hidden_size * 2)
        self.liner2 = nn.Linear(config.hidden_size, 1029 * 2)
        self.glu = nn.GLU()

    def forward(self, x, qk):
        x = torch.cat([x, qk], dim=2)
        h = x
        x = self.ExternalAttention(x)
        x = x + h
        x = self.liner1(x)
        s = self.glu(x)

        return s


class ExternalAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.mk = nn.Linear(config.hidden_size * 2, 512, bias=False)
        self.mv = nn.Linear(512, config.hidden_size * 2, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        attn = self.mk(queries)  # bs,n,S
        attn = self.softmax(attn)  # bs,n,S
        attn = attn / torch.sum(attn, dim=2, keepdim=True)  # bs,n,S
        out = self.mv(attn)  # bs,n,d_model

        return out


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention1(config)
        # self.attn = EAM_Attention()
        self.Ea = Attention2(config)

    def forward(self, x):
        h = x

        x = self.attention_norm(x)

        x, q = self.attn(x)
        x = x + h
        h = x
        x = self.Ea(x, q)

        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, q


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        for _ in range(1):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))
        # self.part_select = Part_Attention()
        self.part_layer = Block(config)
        # self.part_norm = LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, hidden_states):
        attn_weights = []
        for layer in self.layer:
            hidden_states, weights = layer(hidden_states)
            attn_weights.append(weights)
        return hidden_states


class selfattention(nn.Module):
    def __init__(self,
                 in_channles=[64, 64, 128, 256, 512],
                 d_model=256,
                 n_levels=3,
                 proj_idxs=[1, 2, 3],
                 ):
        super(selfattention, self).__init__()
        self.proj_idxs = proj_idxs
        self.projs = nn.ModuleList()
        for idx in self.proj_idxs:
            self.projs.append(ConvModule(in_channles[idx],
                                         d_model,
                                         kernel_size=3,
                                         padding=1,
                                         conv_cfg=dict(type="Conv"),
                                         norm_cfg=dict(type='BN'),
                                         act_cfg=dict(type='ReLU')
                                         ))
        self.position_embedding = build_position_encoding(position_embedding="sine", hidden_dim=d_model)
        self.level_embed = nn.Parameter(torch.Tensor(n_levels, d_model))
        self._reset_parameters()
        if len(proj_idxs) == 1:
            self.config = CONFIGS['1_layer']
        elif len(proj_idxs) == 2:
            self.config = CONFIGS['2_layer']
        elif len(proj_idxs) == 3:
            self.config = CONFIGS['3_layer']
        elif len(proj_idxs) == 4:
            self.config = CONFIGS['4_layer']
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.attention = Encoder(self.config)
        self.Line = nn.Linear(self.config.Liner, 6)
        self.pose = nn.Parameter(torch.zeros(1, self.config.hidden_size, d_model))

    def _reset_parameters(self):
        normal_(self.level_embed)

    def projection(self, feats):

        pos = []
        masks = []
        cnn_feats = []
        tran_feats = []
        # print(len(feats))
        for idx, feats in enumerate(feats):

            if idx not in self.proj_idxs:
                cnn_feats.append(feats)
            else:
                n, c, h, w = feats.shape
                mask = torch.zeros((n, h, w)).to(torch.bool).to(feats.device)
                nested_feats = NestedTensor(feats, mask)
                # print(type(nested_feats))
                masks.append(mask)
                pos.append(self.position_embedding(nested_feats).to(nested_feats.tensors.dtype))
                tran_feats.append(feats)

        for idx, proj in enumerate(self.projs):
            tran_feats[idx] = proj(tran_feats[idx])

        return cnn_feats, tran_feats, pos, masks

    def forward(self, x):

        cnn_feats, trans_feats, pos_embs, masks = self.projection(x)

        features_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        feature_shapes = []
        spatial_shapes = []
        for lvl, (feature, mask, pos_embed) in enumerate(zip(trans_feats, masks, pos_embs)):
            bs, c, h, w = feature.shape
            spatial_shapes.append((h, w))
            feature_shapes.append(feature.shape)
            # print(feature.shape)
            feature = feature.flatten(2).transpose(1, 2)  ## feature:[bs,h*w,c]
            # print(feature.shape)
            mask = mask.flatten(1)  ##  mask :[bs,h*w]
            # print(mask.shape,'mask')
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  ## pos:[bs,h*w,c]

            # print(self.level_embed[lvl].view(1, 1, -1).shape, 1)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)

            lvl_pos_embed_flatten.append(lvl_pos_embed)

            features_flatten.append(feature)
            mask_flatten.append(mask)
        features_flatten = torch.cat(features_flatten, 1)
        # print(features_flatten.shape)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=features_flatten.device)
        # print(spatial_shapes,'2')
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # print(features_flatten,1)
        # print(lvl_pos_embed_flatten, 2)self.pose
        feature = (features_flatten + lvl_pos_embed_flatten).transpose(1, 2)
        # feature = (features_flatten +self.pose).transpose(1, 2)

        feature = self.attention(feature).transpose(1, 2)
        out = []
        features = feature.split(spatial_shapes.prod(1).tolist(), dim=1)
        # print(spatial_shapes.prod(1).tolist())
        for idx, (feats, ori_shape) in enumerate(
                zip(features, spatial_shapes)):
            abc = feats.transpose(1, 2).reshape(feature_shapes[idx])

            out.append(abc)

        for i in range(len(out)):
            # print(out[i].shape)
            x = self.avg(out[i])
            if i == 0:
                output = x
            else:
                output = torch.cat((x, output), 1)
        output = output.view(output.size(0), -1)
        # print(output.shape)
        output = self.Line(output)
        return output


class Mymodel(nn.Module):
    def __init__(self, in_channels=3):
        super(Mymodel, self).__init__()
        self.feature_model = ResNet()
        self.selfattention = selfattention()

    def forward(self, x):
        feature_list, pre = self.feature_model(x)
        x = self.selfattention(feature_list)

        return x


if __name__ == '__main__':
    image = torch.rand((32, 3, 224, 224))
    model = Mymodel()
    y = model(image)
    # print(y)
