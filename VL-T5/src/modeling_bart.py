
import math
import random
from dataclasses import dataclass

from transformers.models.bart.modeling_bart import (
    BartLearnedPositionalEmbedding,
    BartEncoderLayer, BartEncoder, BartDecoder,
    BartPretrainedModel, BartModel, BartForConditionalGeneration,
    BartConfig,
    ACT2FN,
    shift_tokens_right, _expand_mask
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import copy

from transformers.modeling_outputs import ModelOutput, BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, Seq2SeqModelOutput
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import logging
from transformers import BeamScorer, BeamSearchScorer

from language_model import WordEmbedding
from dictionary import Dictionary
from utils_old import tfidf_loading
from cross_attention import Attri_CrossattLayer, Visual_Attention, ReLUWithWeightNormFC, VisualGraphAggregator, SemanticGraphAggregator, BanFusion, cross_CrossattLayer
from contrastive_loss import ContrastiveLoss
criterion_graph = ContrastiveLoss(measure='dot', margin=0.01, max_violation=False)
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda")

logger = logging.get_logger(__name__)



class VisualEmbedding(nn.Module):
    def __init__(self, config, obj_order_embedding):
        super().__init__()
        self.config = config
        feat_dim = config.feat_dim
        pos_dim = config.pos_dim
        # n_objs = config.n_objs
        n_images = config.n_images

        if self.config.individual_vis_layer_norm:
            # Object feature encoding
            feat_embedding = [nn.Linear(feat_dim, config.d_model)]
            if self.config.use_vis_layer_norm:
                feat_embedding.append(nn.LayerNorm(config.d_model))
            self.feat_embedding = nn.Sequential(*feat_embedding)

            absolute_vis_pos_embedding = [nn.Linear(pos_dim + 1, config.d_model)]
            if self.config.use_vis_layer_norm:
                absolute_vis_pos_embedding.append(nn.LayerNorm(config.d_model))
            self.absolute_vis_pos_embedding = nn.Sequential(*absolute_vis_pos_embedding)

            if self.config.use_vis_order_embedding:
                self.obj_order_embedding = obj_order_embedding
                self.img_order_embedding = nn.Embedding(n_images, config.d_model)

                self.default_obj_order_ids = self.config.default_obj_order_ids
        else:
            # Object feature encoding
            feat_embedding = [nn.Linear(feat_dim, config.d_model)]
            self.feat_embedding = nn.Sequential(*feat_embedding)

            absolute_vis_pos_embedding = [nn.Linear(pos_dim + 1, config.d_model)]
            self.absolute_vis_pos_embedding = nn.Sequential(*absolute_vis_pos_embedding)

            if self.config.use_vis_order_embedding:
                self.obj_order_embedding = obj_order_embedding
                self.img_order_embedding = nn.Embedding(n_images, config.d_model)

                self.default_obj_order_ids = self.config.default_obj_order_ids

            if self.config.use_vis_layer_norm:
                self.layer_norm = nn.LayerNorm(config.d_model)

    def get_area(self, pos):
        """
        Args
            pos: [B, N, 4]
                (x1, x2, y1, y2)
        Return
            area : [B, N]
        """
        # [B, N]
        height = pos[:, :, 3] - pos[:, :, 2]
        width = pos[:, :, 1] - pos[:, :, 0]
        area = height * width
        return area

    def forward(self, feats, pos, img_order_ids=None, obj_order_ids=None):
        """
        Args
            feats: [B, N, feat_dim]
            pos: [B, N, 4]
                (x1, x2, y1, y2)
        Return
            relative_vis_pos_embedding: [B, N, N, n_heads]
            absolute_vis_pos_embedding: # [B, N, d_model]
        """

        B, N, _ = feats.size()
        assert pos.size() == (B, N, 4)

        feat_embedding = self.feat_embedding(feats)

        
        dtype = feats.dtype

        area = self.get_area(pos).unsqueeze(2)  # [B, N, 1]
        pos = torch.cat([pos, area], dim=2)  # [B, N, 5]

        # [B, N, d_model]
        absolute_vis_pos_embedding = self.absolute_vis_pos_embedding(pos)

        if self.config.use_vis_order_embedding:
            if img_order_ids is None:
                img_order_ids = torch.zeros(N, dtype=torch.long, device=device)
                img_order_ids = img_order_ids.unsqueeze(0)  # .expand(B, -1)
            img_order_embedding = self.img_order_embedding(img_order_ids)

            # if obj_order_ids is None:
            #     obj_order_ids = torch.tensor(self.default_obj_order_ids[:N], dtype=torch.long, device=device)
            #     obj_order_ids = obj_order_ids.unsqueeze(0)
            # else:
            #     obj_order_ids = self.obj_order_embedding.num_embeddings - obj_order_ids - 1
            # obj_order_embedding = self.obj_order_embedding(obj_order_ids)
            if obj_order_ids is None:
                obj_order_ids = torch.arange(N, dtype=torch.long, device=device)
                obj_order_ids = obj_order_ids.unsqueeze(0)
            # print('raw obj_order_ids', obj_order_ids)
            obj_order_ids = self.obj_order_embedding.num_embeddings - obj_order_ids - 1
            # print('re-indexed obj_order_ids', obj_order_ids)
            obj_order_embedding = self.obj_order_embedding(obj_order_ids)

            vis_embedding = feat_embedding + absolute_vis_pos_embedding + \
                img_order_embedding + obj_order_embedding

        else:
            vis_embedding = feat_embedding + absolute_vis_pos_embedding

        if not self.config.individual_vis_layer_norm:
            if self.config.use_vis_layer_norm:
                vis_embedding = self.layer_norm(vis_embedding)

        return vis_embedding


class JointEncoder(BartEncoder):
    """
    BartEncoder + visual embedding
    """
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)

        self.config = config

        self.visual_embedding = VisualEmbedding(config, self.embed_tokens)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,

        vis_inputs=None,
        vis_attention_mask=None,

        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        inputs_embeds = inputs_embeds + embed_pos

        B, L = inputs_embeds.size()[:-1]

        vis_feats = vis_inputs[0]
        boxes = vis_inputs[1]
        img_order_ids = None
        obj_order_ids = None
        if len(vis_inputs) >= 3:
            img_order_ids = vis_inputs[2]
        if len(vis_inputs) == 4:
            obj_order_ids = vis_inputs[3]




        vis_embeds = self.visual_embedding(vis_feats, boxes, img_order_ids, obj_order_ids)
        V_L = vis_embeds.size(1)

        if self.config.share_vis_lang_layer_norm:
            inputs_embeds = torch.cat([inputs_embeds, vis_embeds], dim=1)

            inputs_embeds = self.layernorm_embedding(inputs_embeds)
        else:
            inputs_embeds = self.layernorm_embedding(inputs_embeds)
            inputs_embeds = torch.cat([inputs_embeds, vis_embeds], dim=1)


        hidden_states = F.dropout(inputs_embeds, p=self.dropout, training=self.training)

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).to(dtype=inputs_embeds.dtype, device=device)

        if vis_attention_mask is None:
            vis_attention_mask = torch.ones(B, V_L, dtype=inputs_embeds.dtype, device=device)

        # print('attention_mask, ', attention_mask.size())
        # print('vis_attention_mask, ', vis_attention_mask.size())

        attention_mask = torch.cat([attention_mask, vis_attention_mask], dim=1)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        # print('ext_attention_mask, ', attention_mask.size())
        # print('attention_mask')
        # print(attention_mask.size())
        # print(attention_mask)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if getattr(self.config, "gradient_checkpointing", False):

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                    )
                else:
                    layer_outputs = encoder_layer(hidden_states, attention_mask, output_attentions=output_attentions)

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


# import clip
class VLBartModel(BartModel):
    def __init__(self, config: BartConfig):
        super(BartModel, self).__init__(config)

        # self.clip_model, _ = clip.load("ViT-B/32", device=device)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        #----- Modified-----#
        # self.encoder = BartEncoder(config, self.shared)
        self.encoder = JointEncoder(config, self.shared)
        #-------------------#
        self.decoder = BartDecoder(config, self.shared)

        dic = Dictionary.load_from_file('../datasets/vqa/dictionary.pkl')
        self.aug_ques_emb = WordEmbedding(dic.ntoken, 300, .0)
        self.aug_ques_emb = tfidf_loading(use_tfidf=True, w_emb =self.aug_ques_emb, dataroot= '../datasets/vqa/')

        self.augmented_ans_emb = WordEmbedding(dic.ntoken, 300, .0)
        self.augmented_ans_emb = tfidf_loading(use_tfidf=True, w_emb = self.augmented_ans_emb, dataroot= '../datasets/vqa/')        


        self.aug_vis_bart_emb = WordEmbedding(dic.ntoken, 300, .0)
        self.aug_vis_bart_emb = tfidf_loading(use_tfidf=True, w_emb = self.aug_vis_bart_emb, dataroot= '../datasets/vqa/')

        self.aug_vis_t5_emb = WordEmbedding(dic.ntoken, 300, .0)
        self.aug_vis_t5_emb = tfidf_loading(use_tfidf=True, w_emb = self.aug_vis_t5_emb, dataroot= '../datasets/vqa/')


        self.aug_vis_ques_mplug2_emb = WordEmbedding(dic.ntoken, 300, .0)
        self.aug_vis_ques_mplug2_emb = tfidf_loading(use_tfidf=True, w_emb = self.aug_vis_ques_mplug2_emb, dataroot= '../datasets/vqa/')


        self.l_att_proj = nn.Sequential(nn.Linear(768, 768), nn.ReLU())
        self.v_att_proj = nn.Sequential(nn.Linear(768, 768), nn.ReLU())
        self.v_bart_att_proj = nn.Sequential(nn.Linear(768, 768), nn.ReLU())
        self.v_t5_att_proj = nn.Sequential(nn.Linear(768, 768), nn.ReLU())
        self.v_vis_ques_att_proj = nn.Sequential(nn.Linear(768, 768), nn.ReLU())
        

        self.proj_aug_ques = nn.Sequential(nn.Linear(600, 768), nn.ReLU())
        self.proj_aug_vis = nn.Sequential(nn.Linear(600, 768), nn.ReLU())
        self.proj_aug_vis_bart = nn.Sequential(nn.Linear(600, 768), nn.ReLU())
        self.proj_aug_vis_t5 = nn.Sequential(nn.Linear(600, 768), nn.ReLU())
        self.proj_vis_ques =  nn.Sequential(nn.Linear(600, 768), nn.ReLU())


        self.self_ans_lan_atten = Attri_CrossattLayer()
        self.self_atten_dep = Attri_CrossattLayer()
        self.self_mplug_ans_atten =  Attri_CrossattLayer()
        self.self_mplug_ques_atten =  Attri_CrossattLayer()

        # self.cross_atten_dep  = cross_CrossattLayer()

        self.atten_1 = Visual_Attention(dim_image_feats=768, dim_att_lstm=768, nb_hidden=1024)
        self.atten_4 = Visual_Attention(dim_image_feats=768, dim_att_lstm=768, nb_hidden=1024)





        self.init_weights()

    def set_input_embeddings(self, value):
        #设置共享参数
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

        self.encoder.visual_embedding.obj_order_embedding = self.shared

    def forward(
        self,
        input_ids=None,
        attention_mask=None,

        vis_inputs=None,
        vis_attention_mask=None,

        aug_ques = None,
        aug_vis = None,
        aug_ques_clip = None,
        aug_vis_clip = None,
        aug_vis_bart = None,
        aug_vis_bart_clip = None,

        aug_vis_t5 = None,
        aug_vis_t5_clip = None,
        aug_vis_mplugques = None,
        aug_vis_mplugques_clip = None,

        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,

        **kwargs,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        ###########################################计算clip相关性
        aug_ques_feat_clip = torch.tensor(aug_ques_clip, dtype=torch.float).unsqueeze(1)   # B,1,1024
        aug_vis_feat_clip = torch.tensor(aug_vis_clip, dtype=torch.float)    # B,26,1024
        aug_vis_bart_feat_clip = torch.tensor(aug_vis_bart_clip, dtype=torch.float) 
        aug_vis_t5_feat_clip = torch.tensor(aug_vis_t5_clip, dtype=torch.float) 
        aug_vis_mplugques_clip = torch.tensor(aug_vis_mplugques_clip, dtype=torch.float) 



        sim_matrix_clip = torch.matmul(aug_vis_feat_clip, aug_ques_feat_clip.transpose(1,2))   # torch.Size([160, 26, 1])  
        sim_matrix_clip_log = F.log_softmax(sim_matrix_clip, dim=-1)

        sim_matrix_bart_clip = torch.matmul(aug_vis_bart_feat_clip, aug_ques_feat_clip.transpose(1,2))   # torch.Size([160, 26, 1])  
        sim_matrix_bart_clip_log = F.log_softmax(sim_matrix_bart_clip, dim=-1)

        sim_matrix_ques_mplug2_clip = torch.matmul(aug_vis_mplugques_clip, aug_ques_feat_clip.transpose(1,2))   # torch.Size([160, 26, 1])  
        sim_matrix_ques_mplug2_clip_log = F.log_softmax(sim_matrix_ques_mplug2_clip, dim=-1)

        sim_matrix_t5_clip = torch.matmul(aug_vis_t5_feat_clip, aug_ques_feat_clip.transpose(1,2))   # torch.Size([160, 26, 1])  
        sim_matrix_t5_clip_log = F.log_softmax(sim_matrix_t5_clip, dim=-1)

        ###################################################
        vis_feats_total = vis_inputs[0]
        boxes_total = vis_inputs[1]

        ################################问题
        aug_ques_emb = self.aug_ques_emb(aug_ques)   
        aug_ques_feat = self.proj_aug_ques(aug_ques_emb)
        aug_ques_feature = self.l_att_proj(aug_ques_feat)   # torch.Size([45, 14, 768])

        ############################描述
        aug_vis_emb = self.augmented_ans_emb(aug_vis)
        aug_vis_feat = self.proj_aug_vis(aug_vis_emb)   #torch.Size([45, 26, 40, 768])
        aug_vis_feature = self.v_att_proj(aug_vis_feat.sum(-2)) #torch.Size([45, 26, 768])

        #########################bart
        aug_vis_bart_emb = self.aug_vis_bart_emb(aug_vis_bart)
        aug_vis_bart_feat = self.proj_aug_vis_bart(aug_vis_bart_emb)
        aug_vis_bart_feature = self.v_bart_att_proj(aug_vis_bart_feat.sum(-2))


        ###############ques_mplug2
        aug_vis_ques_mplug2_emb = self.aug_vis_ques_mplug2_emb(aug_vis_mplugques)
        aug_vis_ques_mplug2_feat = self.proj_vis_ques(aug_vis_ques_mplug2_emb)
        aug_vis_ques_mplug2_feature = self.v_vis_ques_att_proj(aug_vis_ques_mplug2_feat.sum(-2))


        #########################t5
        aug_vis_t5_emb = self.aug_vis_t5_emb(aug_vis_t5)
        aug_vis_t5_feat = self.proj_aug_vis_t5(aug_vis_t5_emb)
        aug_vis_t5_feature = self.v_t5_att_proj(aug_vis_t5_feat.sum(-2))


        #######################################计算相关性

        ###################描述和问题的相关性并初始化视觉特征
        sim_matrix_v2l = torch.matmul(aug_vis_feature, aug_ques_feature.transpose(1,2))   # torch.Size([45, 26, 14])   #torch.Size([B, 19, 1])
        kg_output, _ = torch.topk(sim_matrix_v2l, dim=-1, k=1)
        sim = kg_output + sim_matrix_clip_log
        _,idx = torch.sort(sim.squeeze(), descending=True, dim=-1)   #torch.Size([45, 26])  

        image_feat_1 = torch.ones([int(len(aug_vis_feat)), 36, 2048],dtype =torch.float).to(aug_vis_feat.device)
        image_feat_2 = torch.ones([int(len(aug_vis_feat)), 36, 2048],dtype =torch.float).to(aug_vis_feat.device)
        image_feat_3 = torch.ones([int(len(aug_vis_feat)), 36, 2048],dtype =torch.float).to(aug_vis_feat.device)
        image_feat_4 = torch.ones([int(len(aug_vis_feat)), 36, 2048],dtype =torch.float).to(aug_vis_feat.device) 
        image_feat_5 = torch.ones([int(len(aug_vis_feat)), 36, 2048],dtype =torch.float).to(aug_vis_feat.device)   
        box_feat_1 = torch.ones([int(len(aug_vis_feat)), 36, 4],dtype =torch.float).to(aug_vis_feat.device)
        box_feat_2 = torch.ones([int(len(aug_vis_feat)), 36, 4],dtype =torch.float).to(aug_vis_feat.device)
        box_feat_3 = torch.ones([int(len(aug_vis_feat)), 36, 4],dtype =torch.float).to(aug_vis_feat.device)
        box_feat_4 = torch.ones([int(len(aug_vis_feat)), 36, 4],dtype =torch.float).to(aug_vis_feat.device)  
        box_feat_5 = torch.ones([int(len(aug_vis_feat)), 36, 4],dtype =torch.float).to(aug_vis_feat.device) 
        aug_vis_feat_1 = torch.ones([int(len(aug_ques_feat)), 1, 40, 768],dtype =torch.float).to(aug_ques_feat.device)        

        for batch_num in range(0,int(len(aug_vis_feat))):

            image_feat_1[int(batch_num),:,:] = torch.index_select(vis_feats_total[int(batch_num)], dim = 0, index = idx[int(batch_num),0].squeeze())
            image_feat_2[int(batch_num),:,:] = torch.index_select(vis_feats_total[int(batch_num)], dim = 0, index = idx[int(batch_num),1].squeeze())
            image_feat_3[int(batch_num),:,:] = torch.index_select(vis_feats_total[int(batch_num)], dim = 0, index = idx[int(batch_num),2].squeeze())
            image_feat_4[int(batch_num),:,:] = torch.index_select(vis_feats_total[int(batch_num)], dim = 0, index = idx[int(batch_num),3].squeeze())            
            image_feat_5[int(batch_num),:,:] = torch.index_select(vis_feats_total[int(batch_num)], dim = 0, index = idx[int(batch_num),4].squeeze())            

            box_feat_1[int(batch_num),:,:] = torch.index_select(boxes_total[int(batch_num)], dim = 0, index = idx[int(batch_num),0].squeeze())
            box_feat_2[int(batch_num),:,:] = torch.index_select(boxes_total[int(batch_num)], dim = 0, index = idx[int(batch_num),1].squeeze())
            box_feat_3[int(batch_num),:,:] = torch.index_select(boxes_total[int(batch_num)], dim = 0, index = idx[int(batch_num),2].squeeze())
            box_feat_4[int(batch_num),:,:] = torch.index_select(boxes_total[int(batch_num)], dim = 0, index = idx[int(batch_num),3].squeeze())  
            box_feat_5[int(batch_num),:,:] = torch.index_select(boxes_total[int(batch_num)], dim = 0, index = idx[int(batch_num),4].squeeze())  
                                
    
            aug_vis_feat_1[int(batch_num),:,:,:]  = torch.index_select(aug_vis_feat[int(batch_num)], dim = 0, index = idx[int(batch_num),0].squeeze())





       ###################bart和问题的相关性并初始化视觉特征
        sim_matrix_b2l = torch.matmul(aug_vis_bart_feature, aug_ques_feature.transpose(1,2))   # torch.Size([45, 26, 14])   #torch.Size([B, 19, 1])
        bg_output, _ = torch.topk(sim_matrix_b2l, dim=-1, k=1)
        sim_bart = bg_output + sim_matrix_bart_clip_log
        _,idx_bart = torch.sort(sim_bart.squeeze(), descending=False, dim=-1)   #torch.Size([45, 26])  

        image_feat_bart_1 = torch.ones([int(len(aug_vis_feat)), 36, 2048],dtype =torch.float).to(aug_vis_feat.device)
        image_feat_bart_2 = torch.ones([int(len(aug_vis_feat)), 36, 2048],dtype =torch.float).to(aug_vis_feat.device)
        image_feat_bart_3 = torch.ones([int(len(aug_vis_feat)), 36, 2048],dtype =torch.float).to(aug_vis_feat.device)
        image_feat_bart_4 = torch.ones([int(len(aug_vis_feat)), 36, 2048],dtype =torch.float).to(aug_vis_feat.device)    
        image_feat_bart_5 = torch.ones([int(len(aug_vis_feat)), 36, 2048],dtype =torch.float).to(aug_vis_feat.device)    


        box_feat_bart_1 = torch.ones([int(len(aug_vis_feat)), 36, 4],dtype =torch.float).to(aug_vis_feat.device)
        box_feat_bart_2 = torch.ones([int(len(aug_vis_feat)), 36, 4],dtype =torch.float).to(aug_vis_feat.device)
        box_feat_bart_3 = torch.ones([int(len(aug_vis_feat)), 36, 4],dtype =torch.float).to(aug_vis_feat.device)
        box_feat_bart_4 = torch.ones([int(len(aug_vis_feat)), 36, 4],dtype =torch.float).to(aug_vis_feat.device)
        box_feat_bart_5 = torch.ones([int(len(aug_vis_feat)), 36, 4],dtype =torch.float).to(aug_vis_feat.device)
        
        aug_vis_bart_feat_2 = torch.ones([int(len(aug_ques_feat)), 1, 14, 768],dtype =torch.float).to(aug_ques_feat.device)        

        for batch_num in range(0,int(len(aug_vis_feat))):

            image_feat_bart_1[int(batch_num),:,:] = torch.index_select(vis_feats_total[int(batch_num)], dim = 0, index = idx_bart[int(batch_num),0].squeeze())
            image_feat_bart_2[int(batch_num),:,:] = torch.index_select(vis_feats_total[int(batch_num)], dim = 0, index = idx_bart[int(batch_num),1].squeeze())
            image_feat_bart_3[int(batch_num),:,:] = torch.index_select(vis_feats_total[int(batch_num)], dim = 0, index = idx_bart[int(batch_num),2].squeeze())
            image_feat_bart_4[int(batch_num),:,:] = torch.index_select(vis_feats_total[int(batch_num)], dim = 0, index = idx_bart[int(batch_num),3].squeeze())
            image_feat_bart_5[int(batch_num),:,:] = torch.index_select(vis_feats_total[int(batch_num)], dim = 0, index = idx_bart[int(batch_num),4].squeeze())
            
            box_feat_bart_1[int(batch_num),:,:] = torch.index_select(boxes_total[int(batch_num)], dim = 0, index = idx_bart[int(batch_num),0].squeeze())
            box_feat_bart_2[int(batch_num),:,:] = torch.index_select(boxes_total[int(batch_num)], dim = 0, index = idx_bart[int(batch_num),1].squeeze())
            box_feat_bart_3[int(batch_num),:,:] = torch.index_select(boxes_total[int(batch_num)], dim = 0, index = idx_bart[int(batch_num),2].squeeze())
            box_feat_bart_4[int(batch_num),:,:] = torch.index_select(boxes_total[int(batch_num)], dim = 0, index = idx_bart[int(batch_num),3].squeeze())            
            box_feat_bart_5[int(batch_num),:,:] = torch.index_select(boxes_total[int(batch_num)], dim = 0, index = idx_bart[int(batch_num),4].squeeze())            
            
            aug_vis_bart_feat_2[int(batch_num),:,:,:]  = torch.index_select(aug_vis_bart_feat[int(batch_num)], dim = 0, index = idx_bart[int(batch_num),0].squeeze())

        ###################mplug_ques和问题的相关性并初始化视觉特征

        sim_matrix_ques_mplug2l = torch.matmul(aug_vis_ques_mplug2_feature, aug_ques_feature.transpose(1,2))   # torch.Size([45, 26, 14])   #torch.Size([B, 19, 1])
        ques_mplug2l_output, _ = torch.topk(sim_matrix_ques_mplug2l, dim=-1, k=1)
        sim_mplug2_ques = ques_mplug2l_output + sim_matrix_ques_mplug2_clip_log
        _,idx_mplug2_ques = torch.sort(sim_mplug2_ques.squeeze(), descending=False, dim=-1)   #torch.Size([45, 26])  

        image_feat_mpluhques_1 = torch.ones([int(len(aug_vis_feat)), 36, 2048],dtype =torch.float).to(aug_vis_feat.device)
        image_feat_mpluhques_2 = torch.ones([int(len(aug_vis_feat)), 36, 2048],dtype =torch.float).to(aug_vis_feat.device)
        image_feat_mpluhques_3 = torch.ones([int(len(aug_vis_feat)), 36, 2048],dtype =torch.float).to(aug_vis_feat.device)
        image_feat_mpluhques_4 = torch.ones([int(len(aug_vis_feat)), 36, 2048],dtype =torch.float).to(aug_vis_feat.device)        
        image_feat_mpluhques_5 = torch.ones([int(len(aug_vis_feat)), 36, 2048],dtype =torch.float).to(aug_vis_feat.device)        
        
        box_feat_mpluhques_1 = torch.ones([int(len(aug_vis_feat)), 36, 4],dtype =torch.float).to(aug_vis_feat.device)
        box_feat_mpluhques_2 = torch.ones([int(len(aug_vis_feat)), 36, 4],dtype =torch.float).to(aug_vis_feat.device)
        box_feat_mpluhques_3 = torch.ones([int(len(aug_vis_feat)), 36, 4],dtype =torch.float).to(aug_vis_feat.device)
        box_feat_mpluhques_4 = torch.ones([int(len(aug_vis_feat)), 36, 4],dtype =torch.float).to(aug_vis_feat.device)
        box_feat_mpluhques_5 = torch.ones([int(len(aug_vis_feat)), 36, 4],dtype =torch.float).to(aug_vis_feat.device)

        aug_vis_mplug2_ques_feat_3 = torch.ones([int(len(aug_ques_feat)), 1, 14, 768],dtype =torch.float).to(aug_ques_feat.device)        

        for batch_num in range(0,int(len(aug_vis_feat))):

            image_feat_mpluhques_1[int(batch_num),:,:] = torch.index_select(vis_feats_total[int(batch_num)], dim = 0, index = idx_mplug2_ques[int(batch_num),0].squeeze())
            image_feat_mpluhques_2[int(batch_num),:,:] = torch.index_select(vis_feats_total[int(batch_num)], dim = 0, index = idx_mplug2_ques[int(batch_num),1].squeeze())
            image_feat_mpluhques_3[int(batch_num),:,:] = torch.index_select(vis_feats_total[int(batch_num)], dim = 0, index = idx_mplug2_ques[int(batch_num),2].squeeze())
            image_feat_mpluhques_4[int(batch_num),:,:] = torch.index_select(vis_feats_total[int(batch_num)], dim = 0, index = idx_mplug2_ques[int(batch_num),3].squeeze())            
            image_feat_mpluhques_5[int(batch_num),:,:] = torch.index_select(vis_feats_total[int(batch_num)], dim = 0, index = idx_mplug2_ques[int(batch_num),4].squeeze())            


            box_feat_mpluhques_1[int(batch_num),:,:] = torch.index_select(boxes_total[int(batch_num)], dim = 0, index = idx_mplug2_ques[int(batch_num),0].squeeze())
            box_feat_mpluhques_2[int(batch_num),:,:] = torch.index_select(boxes_total[int(batch_num)], dim = 0, index = idx_mplug2_ques[int(batch_num),1].squeeze())
            box_feat_mpluhques_3[int(batch_num),:,:] = torch.index_select(boxes_total[int(batch_num)], dim = 0, index = idx_mplug2_ques[int(batch_num),2].squeeze())
            box_feat_mpluhques_4[int(batch_num),:,:] = torch.index_select(boxes_total[int(batch_num)], dim = 0, index = idx_mplug2_ques[int(batch_num),3].squeeze())
            box_feat_mpluhques_5[int(batch_num),:,:] = torch.index_select(boxes_total[int(batch_num)], dim = 0, index = idx_mplug2_ques[int(batch_num),4].squeeze())

            aug_vis_mplug2_ques_feat_3[int(batch_num),:,:,:]  = torch.index_select(aug_vis_ques_mplug2_feat[int(batch_num)], dim = 0, index = idx_mplug2_ques[int(batch_num),0].squeeze())

        # ###################t5和问题的相关性并初始化视觉特征



       ###################bart和问题的相关性并初始化视觉特征
        sim_matrix_t2l = torch.matmul(aug_vis_t5_feature, aug_ques_feature.transpose(1,2))   # torch.Size([45, 26, 14])   #torch.Size([B, 19, 1])
        tg_output, _ = torch.topk(sim_matrix_t2l, dim=-1, k=1)
        sim_t5 = tg_output + sim_matrix_t5_clip_log
        _,idx_t5 = torch.sort(sim_t5.squeeze(), descending=False, dim=-1)   #torch.Size([45, 26])  

        image_feat_t5_1 = torch.ones([int(len(aug_vis_feat)), 36, 2048],dtype =torch.float).to(aug_vis_feat.device)
        image_feat_t5_2 = torch.ones([int(len(aug_vis_feat)), 36, 2048],dtype =torch.float).to(aug_vis_feat.device)
        image_feat_t5_3 = torch.ones([int(len(aug_vis_feat)), 36, 2048],dtype =torch.float).to(aug_vis_feat.device)
        image_feat_t5_4 = torch.ones([int(len(aug_vis_feat)), 36, 2048],dtype =torch.float).to(aug_vis_feat.device)        
        image_feat_t5_5 = torch.ones([int(len(aug_vis_feat)), 36, 2048],dtype =torch.float).to(aug_vis_feat.device)        
        
        box_feat_t5_1 = torch.ones([int(len(aug_vis_feat)), 36, 4],dtype =torch.float).to(aug_vis_feat.device)
        box_feat_t5_2 = torch.ones([int(len(aug_vis_feat)), 36, 4],dtype =torch.float).to(aug_vis_feat.device)
        box_feat_t5_3 = torch.ones([int(len(aug_vis_feat)), 36, 4],dtype =torch.float).to(aug_vis_feat.device)
        box_feat_t5_4 = torch.ones([int(len(aug_vis_feat)), 36, 4],dtype =torch.float).to(aug_vis_feat.device)    
        box_feat_t5_5 = torch.ones([int(len(aug_vis_feat)), 36, 4],dtype =torch.float).to(aug_vis_feat.device)    

        
        aug_vis_t5_feat_4 = torch.ones([int(len(aug_ques_feat)), 1, 14, 768],dtype =torch.float).to(aug_ques_feat.device)        

        for batch_num in range(0,int(len(aug_vis_feat))):

            image_feat_t5_1[int(batch_num),:,:] = torch.index_select(vis_feats_total[int(batch_num)], dim = 0, index = idx_t5[int(batch_num),0].squeeze())
            image_feat_t5_2[int(batch_num),:,:] = torch.index_select(vis_feats_total[int(batch_num)], dim = 0, index = idx_t5[int(batch_num),1].squeeze())
            image_feat_t5_3[int(batch_num),:,:] = torch.index_select(vis_feats_total[int(batch_num)], dim = 0, index = idx_t5[int(batch_num),2].squeeze())
            image_feat_t5_4[int(batch_num),:,:] = torch.index_select(vis_feats_total[int(batch_num)], dim = 0, index = idx_t5[int(batch_num),3].squeeze())            
            image_feat_t5_5[int(batch_num),:,:] = torch.index_select(vis_feats_total[int(batch_num)], dim = 0, index = idx_t5[int(batch_num),4].squeeze())            


            
            box_feat_t5_1[int(batch_num),:,:] = torch.index_select(boxes_total[int(batch_num)], dim = 0, index = idx_t5[int(batch_num),0].squeeze())
            box_feat_t5_2[int(batch_num),:,:] = torch.index_select(boxes_total[int(batch_num)], dim = 0, index = idx_t5[int(batch_num),1].squeeze())
            box_feat_t5_3[int(batch_num),:,:] = torch.index_select(boxes_total[int(batch_num)], dim = 0, index = idx_t5[int(batch_num),2].squeeze())
            box_feat_t5_4[int(batch_num),:,:] = torch.index_select(boxes_total[int(batch_num)], dim = 0, index = idx_t5[int(batch_num),3].squeeze())
            box_feat_t5_5[int(batch_num),:,:] = torch.index_select(boxes_total[int(batch_num)], dim = 0, index = idx_t5[int(batch_num),4].squeeze())            
            
            
            aug_vis_t5_feat_4[int(batch_num),:,:,:]  = torch.index_select(aug_vis_bart_feat[int(batch_num)], dim = 0, index = idx_t5[int(batch_num),0].squeeze())


        vis_feat = torch.cat((image_feat_1, image_feat_2, image_feat_3, image_feat_4, image_feat_5, image_feat_bart_1, image_feat_bart_2,image_feat_bart_3, image_feat_bart_4, image_feat_bart_5, image_feat_t5_1, image_feat_t5_2,image_feat_t5_3, image_feat_t5_4,image_feat_t5_5, image_feat_mpluhques_1, image_feat_mpluhques_2, image_feat_mpluhques_3, image_feat_mpluhques_4, image_feat_mpluhques_5),1)
        box_feat = torch.cat((box_feat_1, box_feat_2,box_feat_3, box_feat_4, box_feat_5, box_feat_bart_1, box_feat_bart_2, box_feat_bart_3, box_feat_bart_4, box_feat_bart_5, box_feat_t5_1, box_feat_t5_2, box_feat_t5_3, box_feat_t5_4, box_feat_t5_5, box_feat_mpluhques_1, box_feat_mpluhques_2, box_feat_mpluhques_3, box_feat_mpluhques_4, box_feat_mpluhques_5),1)


        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,

                vis_inputs=(vis_feat, box_feat),
                # vis_inputs=(image_feat, box_feat),
                vis_attention_mask=vis_attention_mask,

                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).to(dtype=torch.float, device=device)
        if vis_attention_mask is None:
            B, L = attention_mask.size()
            V_L = encoder_outputs[0].size(1) - L
            vis_attention_mask = attention_mask.new_ones(B, V_L)
        encoder_attention_mask = torch.cat([attention_mask, vis_attention_mask], dim=1)


        language_feature = encoder_outputs[0][:,:int(L),:]
        vis_feature = encoder_outputs[0][:,int(L):,:]

        # import pdb; pdb.set_trace()
        dep_vis_feature = self.self_atten_dep(torch.cat((aug_vis_feat_1.squeeze(), language_feature),1))[:,:40,:]
        lan_feat = self.self_ans_lan_atten(torch.cat((aug_ques_feature, language_feature),1))[:,:14,:]

        mplug_ques_feature = self.self_mplug_ques_atten(torch.cat((aug_vis_mplug2_ques_feat_3.squeeze(),language_feature),1))[:,:14,:]
        mplug_ans_feature = self.self_mplug_ans_atten(torch.cat((aug_vis_t5_feat_4.squeeze(),language_feature),1))[:,:14,:]


        # import pdb; pdb.set_trace()

        relation_1 = self.atten_1(dep_vis_feature, language_feature)
        relation_4 = self.atten_4(lan_feat,language_feature)



        norm_cus_1 = F.normalize(relation_1, dim=-1)
        norm_cus_4 = F.normalize(relation_4, dim=-1)
      
        # import pdb; pdb.set_trace()
        ans_loss = criterion_graph(norm_cus_1, norm_cus_4) 


        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            # encoder_attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )




        return ans_loss, decoder_outputs.last_hidden_state, decoder_outputs.past_key_values, decoder_outputs.hidden_states, decoder_outputs.attentions, decoder_outputs.cross_attentions, encoder_outputs.last_hidden_state, encoder_outputs.hidden_states, encoder_outputs.attentions
    ###############0,                     1                                2                                 3                              4                         5                                    6                                7                             8                                  



        


class VLBart(BartForConditionalGeneration):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    def __init__(self, config: BartConfig):
        super(BartForConditionalGeneration, self).__init__(config)
        self.model = VLBartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,

        vis_inputs=None,
        vis_attention_mask=None,
        
        aug_ques = None,
        aug_vis = None,
        aug_ques_clip = None,
        aug_vis_clip = None,
        aug_vis_bart = None,
        aug_vis_bart_clip = None,

        aug_vis_t5 = None,
        aug_vis_t5_clip = None,
        aug_vis_mplugques = None,
        aug_vis_mplugques_clip = None,

        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,

        reduce_loss=False,

        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,

            vis_inputs=vis_inputs,
            vis_attention_mask=vis_attention_mask,

            aug_ques = aug_ques,
            aug_vis = aug_vis,

            aug_ques_clip = aug_ques_clip,
            aug_vis_clip = aug_vis_clip,


            aug_vis_bart = aug_vis_bart,
            aug_vis_bart_clip = aug_vis_bart_clip,

            aug_vis_t5 = aug_vis_t5,
            aug_vis_t5_clip = aug_vis_t5_clip,
            aug_vis_mplugques = aug_vis_mplugques,
            aug_vis_mplugques_clip = aug_vis_mplugques_clip,

            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        # masked_lm_loss = None
        # if labels is not None:
        #     # loss_fct = CrossEntropyLoss()
        #     if reduce_loss:
        #         loss_fct = CrossEntropyLoss(ignore_index=-100)
        #     else:
        #         loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
        #     masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        # if not return_dict:
        #     output = (lm_logits,) + outputs[1:]
        #     return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output



        

        # return Seq2SeqLMOutput(
        #     loss=outputs.loss,
        #     past_key_values=outputs.past_key_values,
        #     decoder_hidden_states=outputs.decoder_hidden_states,  decoder_hidden_states
        #     decoder_attentions=outputs.decoder_attentions,
        #     cross_attentions=outputs.cross_attentions,
        #     encoder_last_hidden_state=outputs.encoder_last_hidden_state,
        #     encoder_hidden_states=outputs.encoder_hidden_states,
        #     encoder_attentions=outputs.encoder_attentions,
        #     decoder_hidden_states_add=outputs.decoder_hidden_states_add,
        # )
        return outputs[0], outputs[3]
    


    def prepare_inputs_for_generation(
        self, decoder_input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        output = {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

        if "vis_attention_mask" in kwargs:
            output["vis_attention_mask"] = kwargs['vis_attention_mask']

        return output

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        **model_kwargs
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1,
                                                                expand_size).view(-1).to(device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(
                0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(
                0, expanded_return_idx)

        if model_kwargs.get("vis_attention_mask", None) is not None:
            model_kwargs['vis_attention_mask'] = model_kwargs['vis_attention_mask'].index_select(
                0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx
            )
            model_kwargs["encoder_outputs"] = encoder_outputs

        return input_ids, model_kwargs
    
