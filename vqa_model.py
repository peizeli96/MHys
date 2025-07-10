from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'



def instance_bce(logits, labels):
    assert logits.dim() == 2
    cross_entropy_loss = nn.CrossEntropyLoss()

    prediction_ans_k, top_ans_ind = torch.topk(F.softmax(labels, dim=-1), k=1, dim=-1, sorted=False)
    ce_loss = cross_entropy_loss(logits, top_ans_ind.squeeze(-1))

    return ce_loss



from modeling_t5 import VLT5
class VLT5VQA(VLT5):
    def __init__(self, config, num_answers=None, label2ans=None):
        super().__init__(config)

        if config.classifier:
            self.answer_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model * 2),
                nn.GELU(),
                nn.LayerNorm(config.d_model * 2),
                nn.Linear(config.d_model * 2, num_answers)
            )

        self.num_answers = num_answers
        self.label2ans = label2ans
        self.bce_loss = nn.BCEWithLogitsLoss()

    def train_step(self, batch):

        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        if self.config.classifier:
            B = len(input_ids)

            decoder_input_ids = torch.ones(
                B, 1, dtype=torch.long, device=device) * self.config.decoder_start_token_id

            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                return_dict=True
            )
            target = batch['targets'].to(device)

            last_layer_hidden_state = output.decoder_hidden_states[-1]
            last_hidden_state = last_layer_hidden_state.view(B, -1, self.config.d_model)[:, -1]

            # [B, num_answers]
            logit = self.answer_head(last_hidden_state)

            loss = self.bce_loss(logit, target)

        else:
            lm_labels = batch["target_ids"].to(device)
            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                labels=lm_labels,
                return_dict=True
            )
            # import pdb; pdb.set_trace()
            assert 'loss' in output

            lm_mask = (lm_labels != -100).float()
            B, L = lm_labels.size()

            loss = output['loss']

            loss = loss.view(B, L) * lm_mask

            loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)  # B

            loss = loss * batch['scores'].to(device=device)

            loss = loss.mean()

        result = {
            'loss': loss
        }

        return result

    @torch.no_grad()
    def test_step(self, batch, **kwargs):
        self.eval()
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        result = {}
        if self.config.classifier:
            B = len(input_ids)

            decoder_input_ids = torch.ones(
                B, 1, dtype=torch.long, device=device) * self.config.decoder_start_token_id

            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                return_dict=True
            )

            last_layer_hidden_state = output.decoder_hidden_states[-1]
            last_hidden_state = last_layer_hidden_state.view(B, -1, self.config.d_model)[:, -1]

            # [B, num_answers]
            logit = self.answer_head(last_hidden_state)

            score, pred_ans_id = logit.max(1)
            pred_ans_id = pred_ans_id.cpu().numpy()
            pred_ans = [self.label2ans[ans_id] for ans_id in pred_ans_id]

            result['pred_ans'] = pred_ans

        else:
            output = self.generate(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                **kwargs
            )
            generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            result['token_ids'] = output
            result['pred_ans'] = generated_sents

        return result

from modeling_bart import VLBart
import json
project_dir = Path(__file__).resolve().parent.parent  # VLT5
workspace_dir = project_dir.parent
dataset_dir = workspace_dir.joinpath('datasets/').resolve()
vqa_dir = dataset_dir.joinpath('vqa')
label2ans = json.load(open(vqa_dir.joinpath("trainval_label2ans.json")))



class VLBartVQA(VLBart):
    def __init__(self, config, num_answers=len(label2ans), label2ans=label2ans):
        super().__init__(config)


        # if config.classifier:
        self.answer_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model * 2),
                nn.GELU(),
                nn.LayerNorm(config.d_model * 2),
                nn.Linear(config.d_model * 2, num_answers)
            )

        self.num_answers = num_answers
        self.label2ans = label2ans
        # self.bce_loss = nn.BCEWithLogitsLoss()

        self.answer_head_add = nn.Sequential(
                nn.Linear(config.d_model, config.d_model * 2),
                nn.GELU(),
                nn.LayerNorm(config.d_model * 2),
                nn.Linear(config.d_model * 2, num_answers)
            )


    def train_step(self, batch):

        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        # import pdb; pdb.set_trace()

        # clip_feat_vis = torch.from_numpy(np.array(batch["clip_feat_vis"])).squeeze().to(device) 
        # clip_feat_ques = (batch['clip_feat_ques']).to(device) 

        # import pdb; pdb.set_trace()
        aug_ques = torch.from_numpy(np.array(batch['aug_ques'])).squeeze().to(device)   
        aug_vis = torch.from_numpy(np.array(batch['aug_vis'])).squeeze().to(device) 
        aug_ques_clip = torch.from_numpy(np.array(batch['aug_ques_clip'])).squeeze().to(device)   
        aug_vis_clip = torch.from_numpy(np.array(batch['aug_vis_clip'])).squeeze().to(device) 

        aug_vis_bart = torch.from_numpy(np.array(batch['aug_vis_bart'])).squeeze().to(device)
        aug_vis_bart_clip = torch.from_numpy(np.array(batch['aug_vis_bart_clip'])).squeeze().to(device)  
        # description = torch.from_numpy(np.array(batch['description'])).squeeze().to(device)
        aug_vis_t5 = torch.from_numpy(np.array(batch['aug_vis_t5'])).squeeze().to(device)
        aug_vis_t5_clip = torch.from_numpy(np.array(batch['aug_vis_t5_clip'])).squeeze().to(device)         

        aug_vis_mplugques = torch.from_numpy(np.array(batch['aug_vis_mplugques'])).squeeze().to(device)
        aug_vis_mplugques_clip = torch.from_numpy(np.array(batch['aug_vis_mplugques_clip'])).squeeze().to(device)         

        # if self.config.classifier:
        B = len(input_ids)

        decoder_input_ids = torch.tensor(
                [self.config.decoder_start_token_id, self.config.bos_token_id],
                dtype=torch.long, device=device).unsqueeze(0).expand(B, 2)

        ans_loss, last_layer_hidden_state  = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                decoder_input_ids=decoder_input_ids,
                aug_ques = aug_ques,
                aug_vis = aug_vis,
                aug_vis_bart = aug_vis_bart,
                aug_vis_bart_clip = aug_vis_bart_clip,
                aug_ques_clip = aug_ques_clip,
                aug_vis_clip = aug_vis_clip,

                aug_vis_t5 = aug_vis_t5,
                aug_vis_t5_clip = aug_vis_t5_clip,
                aug_vis_mplugques = aug_vis_mplugques,
                aug_vis_mplugques_clip = aug_vis_mplugques_clip,

                output_hidden_states=True,
                return_dict=True
            )

        # import pdb; pdb.set_trace()
        target = batch['targets'].to(device)

        # last_layer_hidden_state = output.decoder_hidden_states[-1]
        # import pdb; pdb.set_trace()

        last_hidden_state = last_layer_hidden_state[-1].view(B, -1, self.config.d_model)[:, -1]

        # [B, num_answers]
        logit = self.answer_head(last_hidden_state)

        loss = instance_bce(logit, target)



        # last_hidden_state_add = last_layer_hidden_state_add[-1].view(B, -1, self.config.d_model)[:, -1]

        # logit_add = self.answer_head_add(last_hidden_state_add)


        # loss_add = instance_bce(logit_add, target)

        # loss_sum = instance_bce(logit_add+logit, target)



        # print("loss_add",loss_add)
        print("loss", loss)
        print("ans_loss", ans_loss)
        # print("loss_sum",loss_sum)
            

        loss = loss + ans_loss


        # else:


            
        #     lm_labels = batch["target_ids"].to(device)

        #     output = self(
        #         input_ids=input_ids,
        #         vis_inputs=(vis_feats, vis_pos),
        #         labels=lm_labels,
        #         return_dict=True
        #     )



            # output.keys()   odict_keys(['loss', 'logits', 'past_key_values', 'encoder_last_hidden_state'])
            # output['loss'].size() torch.Size([1600])
            # output['logits'].size() torch.Size([200, 8, 50465])
            # output['encoder_last_hidden_state'].size() torch.Size([200, 92, 768])

            # # import pdb; pdb.set_trace()

            # assert 'loss' in output

            # lm_mask = (lm_labels != -100).float()
            # B, L = lm_labels.size()

            # loss = output['loss']

            # loss = loss.view(B, L) * lm_mask

            # loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)  # B

            # loss = loss * batch['scores'].to(device=device)

            # loss = loss.mean() # tensor(4.2841

            # print("loss.mean", loss)
            # print("ans_loss", ans_loss)
            

            # loss = loss + ans_loss 

        result = {
            'loss': loss
        }

        return result


    @torch.no_grad()
    def test_step(self, batch, **kwargs):
        self.eval()
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        aug_ques = torch.from_numpy(np.array(batch['aug_ques'])).squeeze().to(device)   
        aug_vis = torch.from_numpy(np.array(batch['aug_vis'])).squeeze().to(device)   
        aug_ques_clip = torch.from_numpy(np.array(batch['aug_ques_clip'])).squeeze().to(device)   
        aug_vis_clip = torch.from_numpy(np.array(batch['aug_vis_clip'])).squeeze().to(device)   
        aug_vis_bart = torch.from_numpy(np.array(batch['aug_vis_bart'])).squeeze().to(device)
        aug_vis_bart_clip = torch.from_numpy(np.array(batch['aug_vis_bart_clip'])).squeeze().to(device)  
        # description = torch.from_numpy(np.array(batch['description'])).squeeze().to(device)
        aug_vis_t5 = torch.from_numpy(np.array(batch['aug_vis_t5'])).squeeze().to(device)
        aug_vis_t5_clip = torch.from_numpy(np.array(batch['aug_vis_t5_clip'])).squeeze().to(device)         

        aug_vis_mplugques = torch.from_numpy(np.array(batch['aug_vis_mplugques'])).squeeze().to(device)
        aug_vis_mplugques_clip = torch.from_numpy(np.array(batch['aug_vis_mplugques_clip'])).squeeze().to(device)         

        result = {}
        # if self.config.classifier:
        B = len(input_ids)

        decoder_input_ids = torch.tensor(
                [self.config.decoder_start_token_id, self.config.bos_token_id],
                dtype=torch.long, device=device).unsqueeze(0).expand(B, 2)

        _, last_layer_hidden_state = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                decoder_input_ids=decoder_input_ids,
                aug_ques = aug_ques,
                aug_vis = aug_vis,
                aug_vis_bart = aug_vis_bart,
                aug_vis_bart_clip = aug_vis_bart_clip,                
                aug_ques_clip = aug_ques_clip,
                aug_vis_clip = aug_vis_clip,     
                aug_vis_t5 = aug_vis_t5,
                aug_vis_t5_clip = aug_vis_t5_clip,
                aug_vis_mplugques = aug_vis_mplugques,
                aug_vis_mplugques_clip = aug_vis_mplugques_clip,                           
                output_hidden_states=True,
                return_dict=True
            )

        # last_layer_hidden_state = output.decoder_hidden_states[-1]
        last_hidden_state = last_layer_hidden_state[-1].view(B, -1, self.config.d_model)[:, -1]

        #last_hidden_state_add = last_layer_hidden_state_add[-1].view(B, -1, self.config.d_model)[:, -1]



        #####################

        # [B, num_answers]
        logit = self.answer_head(last_hidden_state)

        _, pred_ans_id = logit.max(1)
        pred_ans_id = pred_ans_id.cpu().numpy()
        pred_ans = [self.label2ans[ans_id] for ans_id in pred_ans_id]

        result['pred_ans'] = pred_ans



        # # [B, num_answers]
        # logit_add = self.answer_head_add(last_hidden_state_add)

        # _, pred_ans_id_add = logit_add.max(1)
        # pred_ans_id_add = pred_ans_id_add.cpu().numpy()
        # pred_ans_add = [self.label2ans[ans_id] for ans_id in pred_ans_id_add]

        # result['pred_ans_add'] = pred_ans_add


        

        # logit_sum = logit_add + logit

        # _, pred_ans_id_sum = logit_sum.max(1)
        # pred_ans_id_sum = pred_ans_id_sum.cpu().numpy()
        # pred_ans_sum = [self.label2ans[ans_id] for ans_id in pred_ans_id_sum]


        # result['pred_ans_sum'] = pred_ans_sum


        # logit_sum_sum = logit_add + logit + logit_sum

        # _, pred_ans_id_sum_sum = logit_sum_sum.max(1)
        # pred_ans_id_sum_sum = pred_ans_id_sum_sum.cpu().numpy()
        # pred_ans_sum_sum_sum = [self.label2ans[ans_id] for ans_id in pred_ans_id_sum_sum]


        # result['pred_ans_sum_sum'] = pred_ans_sum_sum_sum




        # else:
        #     output = self.generate(
        #         input_ids=input_ids,
        #         vis_inputs=(vis_feats, vis_pos),
        #         **kwargs
        #     )

            # output, _ = self(
            #     input_ids=input_ids,
            #     vis_inputs=(vis_feats, vis_pos),
            #     aug_ques = aug_ques,
            #     **kwargs
            # )
            # import pdb; pdb.set_trace()



            # generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            # result['token_ids'] = output
            # result['pred_ans'] = generated_sents

        return result