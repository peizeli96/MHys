import json

import torch.nn as nn
import torch 
import math
from torch.nn.utils.weight_norm import weight_norm

BertLayerNorm = torch.nn.LayerNorm
from torch.nn import functional as F



class BertAttention(nn.Module):
    def __init__(self, attention_probs_dropout_prob,num_attention_heads, hidden_size, ctx_dim):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # visual_dim = 2048
        # if ctx_dim is None:
        #     ctx_dim =768
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # if attention_mask is not None:
        #     attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertAttOutput(nn.Module):
    def __init__(self,hidden_size,hidden_dropout_prob):
        super(BertAttOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class CrossattLayer(nn.Module):
    def __init__(self, attention_probs_dropout_prob=0.1,num_attention_heads=8, hidden_size=768,hidden_dropout_prob=0.1):
        super().__init__()
        self.att = BertAttention(attention_probs_dropout_prob=0.1,num_attention_heads=8, hidden_size=768, ctx_dim=768)
        self.output = BertAttOutput(hidden_size,hidden_dropout_prob=0.1)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output



class Attri_CrossattLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.att_cross_1 = CrossattLayer()
        self.att_cross_2 = CrossattLayer()
        self.att_cross_3 = CrossattLayer()
        self.att_cross_4 = CrossattLayer()
        self.att_cross_5 = CrossattLayer()


    def forward(self, input_tensor):
        input_tensor_1 = self.att_cross_1(input_tensor, input_tensor)
        input_tensor_2 = self.att_cross_2(input_tensor_1, input_tensor_1)
        input_tensor_3 = self.att_cross_3(input_tensor_2, input_tensor_2)
        input_tensor_4 = self.att_cross_4(input_tensor_3, input_tensor_3)
        out = self.att_cross_5(input_tensor_4, input_tensor_4)
        
        return out    
    
class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if ''!=act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if ''!=act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class BCNet(nn.Module):
    """Simple class for non-linear bilinear connect network
    """
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=[.2,.5], k=1):   # k = 3
        super(BCNet, self).__init__()
        
        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1]) # attention
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)
        
        if None == h_out:
            pass
        elif h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim*self.k, h_out), dim=None)

    def forward(self, v, q):
        # import pdb;pdb.set_trace()
        if None == self.h_out:
            v_ = self.v_net(v).transpose(1,2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1,2).unsqueeze(2)
            d_ = torch.matmul(v_, q_) # b x h_dim x v x q
            logits = d_.transpose(1,2).transpose(2,3) # b x v x q x h_dim
            return logits

        # broadcast Hadamard product, matrix-matrix production
        # fast computation but memory inefficient
        # epoch 1, time: 157.84
        elif self.h_out <= self.c:
            v_ = self.dropout(self.v_net(v)).unsqueeze(1)
            # import pdb;pdb.set_trace() 
            q_ = self.q_net(q)
            h_ = v_ * self.h_mat # broadcast, b x h_out x v x h_dim
            logits = torch.matmul(h_, q_.unsqueeze(1).transpose(2,3)) # b x h_out x v x q
            logits = logits + self.h_bias
            return logits # b x h_out x v x q

        # batch outer product, linear projection
        # memory efficient but slow computation
        # epoch 1, time: 304.87
        else: 
            v_ = self.dropout(self.v_net(v)).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            logits = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            return logits.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q

    def forward_with_weights(self, v, q, w):
        v_ = self.v_net(v).transpose(1,2).unsqueeze(2) # b x d x 1 x v
        q_ = self.q_net(q).transpose(1,2).unsqueeze(3) # b x d x q x 1
        logits = torch.matmul(torch.matmul(v_.float(), w.unsqueeze(1).float()), q_.float()).type_as(v_) # b x d x 1 x 1
        # logits = torch.matmul(torch.matmul(v_, w.unsqueeze(1)), q_)# b x d x 1 x 1
        logits = logits.squeeze(3).squeeze(2)
        if 1 < self.k:
            logits = logits.unsqueeze(1) # b x 1 x d
            logits = self.p_net(logits).squeeze(1) * self.k # sum-pooling
        return logits

    def forward_with_v_weights(self, v, q, w):
        v_ = self.v_net(v).transpose(1, 2)  # b x d x v
        q_ = self.q_net(q).transpose(1, 2)  # b x d x q
        logits = torch.cat([torch.matmul(q_, w.transpose(1, 2).float()), v_.float()], 1).type_as(v_)  # b x 2xd x v
        # logits = torch.matmul(torch.matmul(v_, w.unsqueeze(1)), q_)# b x d x 1 x 1
        logits = logits.transpose(1, 2)
        if 1 < self.k:
            logits = logits.unsqueeze(1)  # b x 1 x d
            logits = self.p_net(logits).squeeze(1) * self.k  # sum-pooling
        return logits

    def forward_with_q_weights(self, v, q, w):
        v_ = self.v_net(v).transpose(1, 2)  # b x d x v
        q_ = self.q_net(q).transpose(1, 2)  # b x d x q
        logits = torch.mul(torch.matmul(v_, w.float()), q_.float()).type_as(v_)  # b x 2xd x v
        # logits = torch.matmul(torch.matmul(v_, w.unsqueeze(1)), q_)# b x d x 1 x 1
        logits = logits.transpose(1, 2)
        if 1 < self.k:
            logits = logits.unsqueeze(1)  # b x 1 x d
            logits = self.p_net(logits).squeeze(1) * self.k  # sum-pooling
        return logits

class BiAttention(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=[.2,.5]):
        super(BiAttention, self).__init__()

        self.glimpse = glimpse
        self.logits = weight_norm(BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=1), \
            name='h_mat', dim=None)

    def forward(self, v, q, v_mask=True):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        p, logits = self.forward_all(v, q, v_mask)
        return p, logits

    def forward_all(self, v, q, v_mask=True):
        # import pdb;pdb.set_trace()
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v, q)  # b x g x v x q

        # if v_mask:
        #     mask = (0 == v.abs().sum(2)).unsqueeze(1).unsqueeze(3).expand(logits.size())
        #     logits.data.masked_fill_(mask.data, -float('inf'))

        p = nn.functional.softmax(logits.view(-1, self.glimpse, v_num * q_num), 2)
        return p.view(-1, self.glimpse, v_num, q_num), logits

class BanFusion(nn.Module):
    def __init__(self, dim_1, dim_2, gamma, omega):
        super(BanFusion, self).__init__()
        self.glimpse = gamma
        self.omega = omega
        self.v_att = BiAttention(dim_1, dim_2, dim_1, gamma)

        b_net = []
        q_prj = []

        for i in range(self.glimpse):
            b_net.append(BCNet(dim_1, dim_2, dim_1, None, k=1))
            q_prj.append(FCNet([dim_1, dim_1], '', .2))


        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)


    def forward(self, mod1, mod2):
        """Forward
        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]
        return: logits, not probs
        """

        b_emb = [0] * self.glimpse
        att, logits = self.v_att.forward_all(mod1, mod2)  # b x g x v x q

        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(mod1, self.omega * mod2, att[:, g, :, :])
            mod1 = self.omega * self.q_prj[g](b_emb[g].unsqueeze(1)) * mod1 + mod1

        return mod1
    


    

class Visual_Attention(nn.Module):
    def __init__(self, dim_image_feats, dim_att_lstm, nb_hidden):
        super(Visual_Attention,self).__init__()
        self.fc_image_feats = nn.Linear(dim_image_feats, nb_hidden, bias=False)
        self.fc_att_lstm = nn.Linear(dim_att_lstm, nb_hidden, bias=False)
        self.act_tan = nn.Tanh()
        self.fc_att = nn.Linear(nb_hidden, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, image_feats, h1):

        att_lstm_emb = self.fc_att_lstm(h1)
        image_feats_emb = self.fc_image_feats(image_feats)
        similarity = torch.matmul(image_feats_emb, att_lstm_emb.permute(0, 2, 1))
        sim,_ = torch.topk(similarity, dim=-1, k=1)
        att = F.softmax(sim, dim=1)
        weighted_feats = att * image_feats
        return weighted_feats.sum(dim=1)
    
class ReLUWithWeightNormFC(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ReLUWithWeightNormFC, self).__init__()

        layers = []
        layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
        layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class VisualGraphAggregator(nn.Module):
    def __init__(self, is_proj_dim=768, obj_dim=768, sem_dim=768):
        super(VisualGraphAggregator, self).__init__()
        self.im_sem_embed_dim = is_proj_dim
        self.im_embed_is = ReLUWithWeightNormFC(obj_dim, is_proj_dim)
        self.sem_embed_is = ReLUWithWeightNormFC(sem_dim, is_proj_dim)
        self.im_sem_combined = ReLUWithWeightNormFC(is_proj_dim * 2, obj_dim)


    def forward(self, vis, sem):

        # alignment, attention
        vis_proj = self.im_embed_is(vis)  # [batch, obj_num, im_sem_embed_dim]
        sem_proj = self.sem_embed_is(sem)  # [batch, ocr_num, im_sem_embed_dim]
        i_att = torch.matmul(vis_proj, sem_proj.permute(0, 2, 1))  # [batch, obj_num, im_sem_embed_dim]
        vis_sem = F.softmax(i_att, dim=2)
        att = torch.matmul(vis_sem, sem_proj)

        # aggregate the information, add aligned with aggregated
        i_new = self.im_sem_combined(torch.cat((vis_proj, att), 2))  # [batch, obj_num, obj_dim]
        return vis + i_new

class SemanticGraphAggregator(nn.Module):
    def __init__(self, im_sem_embed_dim=768, obj_dim=768, sem_dim=768):
        super(SemanticGraphAggregator, self).__init__()
        # for image_semantic_aggregator
        self.im_embed_is = ReLUWithWeightNormFC(obj_dim, im_sem_embed_dim)
        self.sem_embed_is = ReLUWithWeightNormFC(sem_dim, im_sem_embed_dim)
        self.im_sem_combined = ReLUWithWeightNormFC(im_sem_embed_dim * 2, sem_dim)


    def forward(self, vis, emb):
        # calculate the attention weights
        # import pdb; pdb.set_trace()
        vis_proj = self.im_embed_is(vis)  # [batch, obj_num, im_sem_embed_dim]
        emb_proj = self.sem_embed_is(emb)  # [batch, ocr_num, im_sem_embed_dim]

        # attention matrix
        similarity = torch.matmul(emb_proj, vis_proj.permute(0, 2, 1))
        att = F.softmax(similarity, dim=2)  # [batch, ocr_num, ocr_num]
        i_att = torch.matmul(att, vis_proj)  # [batch, ocr_num, im_sem_embed_dim]

        # aggregate the information, add aligned with aggregated
        combine = self.im_sem_combined(torch.cat((emb_proj, i_att), 2))  # [batch, ocr_num, sem_dim]
        return combine+emb
    
class cross_CrossattLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.att_cross_1 = CrossattLayer()
        self.att_cross_2 = CrossattLayer()
        self.att_cross_3 = CrossattLayer()
        # self.att_cross_4 = CrossattLayer()
        # self.att_cross_5 = CrossattLayer()
        # self.att_cross_6 = CrossattLayer()


    def forward(self, input_tensor_1, input_tensor_2):

        input_tensor_1 = self.att_cross_1(input_tensor_1, input_tensor_2)
        input_tensor_2 = self.att_cross_1(input_tensor_2, input_tensor_1)

        input_tensor_1 = self.att_cross_2(input_tensor_1, input_tensor_2)
        input_tensor_2 = self.att_cross_2(input_tensor_2, input_tensor_1)

        input_tensor_1 = self.att_cross_3(input_tensor_1, input_tensor_2)
        input_tensor_2 = self.att_cross_3(input_tensor_2, input_tensor_1)

        # input_tensor_1 = self.att_cross_4(input_tensor_1, input_tensor_2)
        # input_tensor_2 = self.att_cross_4(input_tensor_2, input_tensor_1)

        # input_tensor_1 = self.att_cross_5(input_tensor_1, input_tensor_2)
        # input_tensor_2 = self.att_cross_5(input_tensor_2, input_tensor_1)  
        
        # input_tensor_1 = self.att_cross_6(input_tensor_1, input_tensor_2)
        # input_tensor_2 = self.att_cross_6(input_tensor_2, input_tensor_1)
        
        return input_tensor_1   