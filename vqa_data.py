from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import random
from multiprocessing import Pool
import h5py
import pickle
import math
from tqdm import tqdm
import torch
import numpy as np
from copy import deepcopy
import re
import pandas as pd
from PIL import Image
from torch.utils.data.distributed import DistributedSampler

from transformers import T5TokenizerFast, BartTokenizer
from tokenization import VLT5TokenizerFast
# import clip

project_dir = Path(__file__).resolve().parent.parent  # VLT5
workspace_dir = project_dir.parent
dataset_dir = workspace_dir.joinpath('datasets/').resolve()
coco_dir = dataset_dir.joinpath('COCO')
vg_dir = dataset_dir.joinpath('VG')
coco_img_dir = coco_dir.joinpath('images/')
coco_feature_dir = coco_dir.joinpath('features')
vqa_dir = dataset_dir.joinpath('vqa')
# import pdb; pdb.set_trace()

class VQAFineTuneDataset(Dataset):
    def __init__(self, split='train', raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None, mode='train'):
        super().__init__()

        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.args = args

        self.mode = mode
        # _, self.preprocess = clip.load('ViT-B/32', device="cuda")

        # Loading datasets to data
        self.sources = split.split(',')
        if self.verbose:
            print('Data sources: ', self.sources)

        if 't5' in self.args.backbone:
            if self.args.use_vision:
                self.tokenizer = VLT5TokenizerFast.from_pretrained(
                    args.backbone,
                    max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case)
            else:
                self.tokenizer = T5TokenizerFast.from_pretrained(
                    args.backbone,
                    max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case)

        elif 'bart' in self.args.backbone:
            self.tokenizer = BartTokenizer.from_pretrained(
                args.backbone,
                # max_length=self.args.max_text_length,
                do_lower_case=self.args.do_lower_case)

            if args.use_vis_order_embedding:
                additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                        [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
                special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
                num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

        self.answer_normalizer = VQAEvaluator()

        self.img_ids_to_source = {}
        data_info_dicts = []
        for source in self.sources:
            data_info_path = dataset_dir.joinpath(f'vqa/{source}.json')
            with open(data_info_path) as f:
                _data_info_dicts = json.load(f)
                for _d in _data_info_dicts:
                    for img_id in _d['img_id']:
                        self.img_ids_to_source[str(img_id)] = 'vg'

                        # import pdb; pdb.set_trace()


                data_info_dicts.extend(_data_info_dicts)
            if self.verbose:
                print(f"Loaded {len(_data_info_dicts)} data from", source)

        data = data_info_dicts

        self.n_gpus = torch.cuda.device_count()

        self.rank = rank

        if self.topk > 0:
            data = data[:self.topk]
            if self.verbose:
                print(f"Use only {self.topk} data")

        self.data = data

        if self.verbose:
            print("# all sentences:", len(self.data))

        self.n_boxes = args.n_boxes
        self.source_to_h5 = {
            'train': coco_feature_dir.joinpath(f'train2014_obj36.h5'),
            'minival': coco_feature_dir.joinpath(f'val2014_obj36.h5'),
            'nominival': coco_feature_dir.joinpath(f'val2014_obj36.h5'),
            'test': coco_feature_dir.joinpath(f'test2015_obj36.h5'),

            'vg': coco_feature_dir.joinpath('vg_gqa_obj36.h5'),

            'train2014': coco_feature_dir.joinpath(f'train2014_obj36.h5'),
            'val2014': coco_feature_dir.joinpath(f'val2014_obj36.h5'),
        }


        # if self.mode == "train":
        with open(dataset_dir.joinpath(f'vqa/clip_ques_feat_RN5064.pickle'),"rb") as f:
            self.aug_ques_clip = pickle.load(f)

        with open(dataset_dir.joinpath(f'vqa/clip_description_feat_RN5064.pickle'),"rb") as f:
            self.aug_vis_clip = pickle.load(f)

        with open(dataset_dir.joinpath(f'vqa/clip_single_image_ques_bart_feat_RN5064.pickle'),"rb") as f:
            self.aug_vis_bart_clip = pickle.load(f)         

        with open(dataset_dir.joinpath(f'vqa/clip_single_image_ques_t5_feat_RN5064.pickle'),"rb") as f:
            self.aug_vis_t5_clip = pickle.load(f)                

        with open(dataset_dir.joinpath(f'vqa/clip_qa_ques_mplug2_new_RN5064.pickle'),"rb") as f:
            self.aug_vis_mplugques_clip = pickle.load(f)              


        with open(dataset_dir.joinpath(f'vqa/retvqa_release_v1_ques_14.pickle'),"rb") as f:
            self.aug_ques = pickle.load(f)

        with open(dataset_dir.joinpath(f'vqa/image_mplug_owl_40.pickle'),"rb") as f:
            self.aug_vis = pickle.load(f)

        with open(dataset_dir.joinpath(f'vqa/single_image_ques_bart_14.pickle'),"rb") as f:
            self.aug_vis_bart = pickle.load(f)

        with open(dataset_dir.joinpath(f'vqa/single_image_ques_t5_14.pickle'),"rb") as f:
            self.aug_vis_t5 = pickle.load(f)

        with open(dataset_dir.joinpath(f'vqa/test_qa_ques_mplug2_new_14.pickle'),"rb") as f:
            self.aug_vis_mplugques = pickle.load(f)


        # import pdb; pdb.set_trace()


        # vis_feats = []
        # out_dict_boxes = []
        # aug_vis = []
        # for img_id in self.data[0]['img_id']:
        #     # img_id = datum['img_id']

        #         source = self.img_ids_to_source[str(img_id)]

        #         f = self.source_to_h5[source]

        #         if isinstance(f, Path):
        #             # path = self.data_source_to_h5_path[source]
        #             f = h5py.File(f, 'r')
        #             # self.split_to_h5_features[split_i] = f
        #             self.source_to_h5[source] = f

        #         feats = np.zeros(shape=(self.n_boxes, 2048), dtype=np.float32)
        #         try:
        #             f[f'{img_id}/features'].read_direct(feats)
        #         except KeyError:
        #             print('img_id', img_id)
        #             # print(datum)
        #             exit()

        #         feats = torch.from_numpy(feats)
        #         vis_feats.append(feats)

        #         # Normalize the boxes (to 0 ~ 1)
        #         img_h = f[f'{img_id}/img_h'][()]
        #         img_w = f[f'{img_id}/img_w'][()]
        #         boxes = f[f'{img_id}/boxes'][()]  # (x1, y1, x2, y2)
        #         boxes[:, (0, 2)] /= img_w
        #         boxes[:, (1, 3)] /= img_h
        #         np.testing.assert_array_less(boxes, 1+1e-5)
        #         # np.testing.assert_array_less(boxes, 1+5e-2)
        #         np.testing.assert_array_less(-boxes, 0+1e-5)
        #         boxes = torch.from_numpy(boxes)

        #         boxes.clamp_(min=0.0, max=1.0)

        #         out_dict_boxes.append(boxes)

        #         aug_vis.append(self.aug_vis[str(img_id)])

        # import pdb; pdb.set_trace()
        # torch.cat((vis_feats))
        # torch.cat((out_dict_boxes))
        # torch.cat((aug_vis))



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]

        ###### Image ######
        # if self.args.use_vision:
        #     img_id = datum['img_id']
        #     out_dict['img_id'] = img_id

        #     source = self.img_ids_to_source[img_id]

        #     f = self.source_to_h5[source]

        #     if isinstance(f, Path):
        #         # path = self.data_source_to_h5_path[source]
        #         f = h5py.File(f, 'r')
        #         # self.split_to_h5_features[split_i] = f
        #         self.source_to_h5[source] = f

        #     feats = np.zeros(shape=(self.n_boxes, 2048), dtype=np.float32)
        #     try:
        #         f[f'{img_id}/features'].read_direct(feats)
        #     except KeyError:
        #         print('img_id', img_id)
        #         print(datum)
        #         exit()

        #     feats = torch.from_numpy(feats)
        #     out_dict['vis_feats'] = feats

        #     # Normalize the boxes (to 0 ~ 1)
        #     img_h = f[f'{img_id}/img_h'][()]
        #     img_w = f[f'{img_id}/img_w'][()]
        #     boxes = f[f'{img_id}/boxes'][()]  # (x1, y1, x2, y2)
        #     boxes[:, (0, 2)] /= img_w
        #     boxes[:, (1, 3)] /= img_h
        #     np.testing.assert_array_less(boxes, 1+1e-5)
        #     # np.testing.assert_array_less(boxes, 1+5e-2)
        #     np.testing.assert_array_less(-boxes, 0+1e-5)
        #     boxes = torch.from_numpy(boxes)

        #     boxes.clamp_(min=0.0, max=1.0)

        #     out_dict['boxes'] = boxes

        if self.args.use_vision:
            out_dict['img_id'] = []
            vis_feats = []
            out_dict_boxes = []
            aug_vis = []
            aug_vis_clip = []
            aug_vis_bart_clip = []
            aug_vis_bart = []
            aug_vis_t5_clip = []
            aug_vis_t5 = []
            aug_vis_mplugques= []
            aug_vis_mplugques_clip = []

            for img_id in datum['img_id']:
            # img_id = datum['img_id']
                
                out_dict['img_id'].append(img_id) 
    
                source = self.img_ids_to_source[str(img_id)]

                f = self.source_to_h5[source]

                if isinstance(f, Path):
                    # path = self.data_source_to_h5_path[source]
                    f = h5py.File(f, 'r')
                    # self.split_to_h5_features[split_i] = f
                    self.source_to_h5[source] = f

                feats = np.zeros(shape=(self.n_boxes, 2048), dtype=np.float32)
                try:
                    f[f'{img_id}/features'].read_direct(feats)
                except KeyError:
                    print('img_id', img_id)
                    print(datum)
                    exit()

                feats = torch.from_numpy(feats)
                vis_feats.append(feats.unsqueeze(0))

                # Normalize the boxes (to 0 ~ 1)
                img_h = f[f'{img_id}/img_h'][()]
                img_w = f[f'{img_id}/img_w'][()]
                boxes = f[f'{img_id}/boxes'][()]  # (x1, y1, x2, y2)
                boxes[:, (0, 2)] /= img_w
                boxes[:, (1, 3)] /= img_h
                np.testing.assert_array_less(boxes, 1+1e-5)
                # np.testing.assert_array_less(boxes, 1+5e-2)
                np.testing.assert_array_less(-boxes, 0+1e-5)
                boxes = torch.from_numpy(boxes)

                boxes.clamp_(min=0.0, max=1.0)

                out_dict_boxes.append(boxes.unsqueeze(0))

                aug_vis.append(self.aug_vis[str(img_id)])
                aug_vis_clip.append(self.aug_vis_clip[str(img_id)])

                aug_vis_bart_clip.append(self.aug_vis_bart_clip[str(img_id)])
                aug_vis_bart.append(self.aug_vis_bart[str(img_id)])

                aug_vis_t5_clip.append(self.aug_vis_t5_clip[str(img_id)])
                aug_vis_t5.append(self.aug_vis_t5[str(img_id)])     

                aug_vis_mplugques.append(self.aug_vis_mplugques[str(img_id)])   
                aug_vis_mplugques_clip.append(self.aug_vis_mplugques_clip[str(img_id)])         

            # out_dict['vis_feats'] = np.concatenate((c, vis_feats[1]),0)
            # out_dict['boxes'] = torch.cat((out_dict_boxes[0],out_dict_boxes[1]),0)
            out_dict['vis_feats'] = np.concatenate((vis_feats),0)
            out_dict['boxes'] =   np.concatenate((out_dict_boxes),0)
            out_dict['aug_vis'] = np.concatenate((aug_vis),0)
            out_dict['aug_vis_bart'] = np.concatenate((aug_vis_bart),0)

            out_dict['aug_vis_clip'] = np.concatenate((aug_vis_clip),0)
            out_dict['aug_vis_bart_clip'] = np.concatenate((aug_vis_bart_clip),0)

            out_dict['aug_vis_t5'] = np.concatenate((aug_vis_t5),0)
            out_dict['aug_vis_t5_clip'] = np.concatenate((aug_vis_t5_clip),0)

            out_dict['aug_vis_mplugques'] = np.concatenate((aug_vis_mplugques),0)
            out_dict['aug_vis_mplugques_clip'] = np.concatenate((aug_vis_mplugques_clip),0)



        ###### Text #####
            


        # caption = datum['caption']
        if 'sent' in datum:
            sent = datum['sent']
        elif 'question' in datum:
            sent = datum['question']

        input_ids = self.tokenizer.encode(f'vqa: {sent}', max_length=20, truncation=True)

        question_id = datum['question_id']
        out_dict['question_id'] = question_id

        out_dict['aug_ques'] = self.aug_ques[str(question_id)]
        out_dict['aug_ques_clip'] = self.aug_ques_clip[str(question_id)]


        # if len(out_dict['clip_feat_ques']) >14:
        #     out_dict['clip_feat_ques'] = out_dict['clip_feat_ques'][:14]
        # else:
        #     out_dict['clip_feat_ques'] = out_dict['clip_feat_ques'] + (14-len(out_dict['clip_feat_ques'])) * clip.tokenize('<|endoftext|>')
        
        
        # out_dict['description'] = self.description[str(question_id)]


        out_dict['sent'] = sent
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)

        # out_dict['target_ids'] = torch.LongTensor(target_ids)
        # out_dict['target_length'] = len(target_ids)

        if 'is_topk_optimal' in datum:
            out_dict['is_topk_optimal'] = datum['is_topk_optimal']

        if 'label' in datum:
            label = datum['label']
            out_dict['label'] = label

            # 3129 topk answers
            # if self.args.classifier:
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                    try:
                        target[self.raw_dataset.ans2label[ans]] = score
                    except:
                        target[self.raw_dataset.ans2label[""]] = score
            out_dict['target'] = target



        return out_dict


    def collate_fn(self, batch):
        batch_entry = {}

        args = batch[0]['args']

        B = len(batch)

        S_W_L = max(entry['input_length'] for entry in batch)
        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        # clip_feat_ques = torch.ones(B, 77, dtype=torch.long)

        if args.use_vision:
            # V_L = len(batch[0]['boxes'])
            V_L = 36
            feat_dim = batch[0]['vis_feats'].shape[-1]

            boxes = torch.zeros(B, 26, V_L, 4, dtype=torch.float)
            vis_feats = torch.zeros(B, 26, V_L, feat_dim, dtype=torch.float)

        if 'target' in batch[0]:
            # targets = []
            targets = torch.zeros(B, len(batch[0]['target']), dtype=torch.float)
        if 'target_ids' in batch[0]:
            T_W_L = max(entry['target_length'] for entry in batch)
            target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        sentences = []
        aug_ques = []
        aug_ques_clip = []
        aug_vis = []
        aug_vis_bart = []
        aug_vis_bart_clip = []
        aug_vis_t5 = []
        aug_vis_t5_clip = []
        aug_vis_mplugques = []
        aug_vis_mplugques_clip = []
        aug_vis_clip = []
        question_ids = []
        answers = []
        all_answers = []
        img_ids = []
        img_paths = []
        labels = []
        scores = []
        is_topk_optimal = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            # clip_feat_ques[i, :77] = entry['clip_feat_ques'].squeeze()

            if args.use_vision:
                boxes[i] += entry['boxes']
                vis_feats[i] += entry['vis_feats']
                # img_ids.append(entry['img_id'])
                # img_paths.append(entry['img_path'])

            if 'target_ids' in entry:
                target_ids[i, :entry['target_length']] = entry['target_ids']

            if 'target' in entry:
                targets[i] += entry['target']
                # targets.append(entry['target'])

            sentences.append(entry['sent'])
            aug_ques.append(entry['aug_ques'])
            aug_ques_clip.append(entry['aug_ques_clip'])
            aug_vis.append(entry['aug_vis'])
            aug_vis_bart.append(entry['aug_vis_bart'])
            aug_vis_clip.append(entry['aug_vis_clip'])
            aug_vis_bart_clip.append(entry['aug_vis_bart_clip'])

            aug_vis_t5.append(entry['aug_vis_t5'])
            aug_vis_t5_clip.append(entry['aug_vis_t5_clip'])

            aug_vis_mplugques.append(entry['aug_vis_mplugques'])
            aug_vis_mplugques_clip.append(entry['aug_vis_mplugques_clip'])

            # clip_feat_vis.append(entry['clip_feat_vis'])



            # description.append(entry['description'])
            question_ids.append(entry['question_id'])
            if 'answer' in entry:
                answers.append(entry['answer'])
            if 'all_answers' in entry:
                all_answers.append(entry['all_answers'])
            if 'score' in entry:
                scores.append(entry['score'])

            if 'label' in entry:
                labels.append(entry['label'])

            if 'is_topk_optimal' in entry:
                is_topk_optimal.append(entry['is_topk_optimal'])

        batch_entry['input_ids'] = input_ids
        if 'target_ids' in batch[0]:
            word_mask = target_ids != self.tokenizer.pad_token_id
            target_ids[~word_mask] = -100
            batch_entry['target_ids'] = target_ids
        if 'target' in batch[0]:
            # targets = torch.stack(targets, dim=0)
            batch_entry['targets'] = targets

        if args.use_vision:
            batch_entry['boxes'] = boxes
            batch_entry['vis_feats'] = vis_feats
            # batch_entry['img_id'] = img_ids
            # batch_entry['img_paths'] = img_paths

        batch_entry['sent'] = sentences
        batch_entry['question_ids'] = question_ids
        batch_entry['answers'] = answers
        batch_entry['all_answers'] = all_answers
        batch_entry['scores'] = torch.FloatTensor(scores)
        batch_entry['labels'] = labels

        batch_entry['aug_ques'] = aug_ques
        batch_entry['aug_vis'] = aug_vis
        batch_entry['aug_vis_bart'] = aug_vis_bart
        batch_entry['aug_vis_bart_clip'] = aug_vis_bart_clip

        batch_entry['aug_vis_t5'] = aug_vis_t5
        batch_entry['aug_vis_t5_clip'] = aug_vis_t5_clip 

        batch_entry['aug_vis_mplugques'] = aug_vis_mplugques
        batch_entry['aug_vis_mplugques_clip'] = aug_vis_mplugques_clip              

        batch_entry['aug_vis_clip'] = aug_vis_clip
        batch_entry['aug_ques_clip'] = aug_ques_clip


        # batch_entry['clip_feat_ques'] = clip_feat_ques
        # batch_entry['clip_feat_vis'] = clip_feat_vis

        # batch_entry['description'] = description

        batch_entry['args'] = args
        batch_entry['task'] = 'vqa'

        return batch_entry


def get_loader(args, split='karpathy_train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0, topk=-1):

    verbose = True

    _dset = VQADataset(split, verbose)
    
    dataset = VQAFineTuneDataset(
        split,
        raw_dataset=_dset,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        args=args,
        mode=mode)

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=workers, pin_memory=True, sampler=sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers, pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False)

    if verbose:
        loader.evaluator = VQAEvaluator(_dset)

    loader.task = 'vqa'

    return loader


class VQADataset:
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }
    """

    def __init__(self, splits: str, verbose=True):
        self.name = splits
        self.splits = splits.split(',')

        with open(dataset_dir.joinpath(f'vqa/retvqa_release_v1_trainval_vlt5.json')) as f:
            train2014_data = json.load(f)
        with open(dataset_dir.joinpath(f'vqa/retvqa_release_v1_test_vlt5.json')) as f:
            val2014_data = json.load(f)
        train2014_id2datum = {}
        for datum in train2014_data:
            qid = datum['question_id']
            train2014_id2datum[qid] = datum
        val2014_id2datum = {}
        for datum in val2014_data:
            qid = datum['question_id']
            val2014_id2datum[qid] = datum
        self.id2datum_gt = {**train2014_id2datum, **val2014_id2datum}

        # Loading datasets
        self.data = []
        # import pdb; pdb.set_trace()
        for split in self.splits:
            self.data.extend(
                json.load(open(vqa_dir.joinpath("%s.json" % split))))

        if verbose:
            print("Load %d data from split(s) %s." %
                  (len(self.data), self.name))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Topk Answers
        self.ans2label = json.load(
            open(vqa_dir.joinpath("trainval_ans2label.json")))
        self.label2ans = json.load(
            open(vqa_dir.joinpath("trainval_label2ans.json")))
        assert len(self.ans2label) == len(self.label2ans)

        if verbose:
            print('# Answers:', len(self.ans2label))

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


class VQAEvaluator:
    def __init__(self, dataset: VQADataset = None):
        self.dataset = dataset

        """https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py"""

        self.contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
							 "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
							 "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
							 "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
							 "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
							 "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
							 "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
							 "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
							 "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
							 "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
							 "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
							 "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
							 "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
							 "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
							 "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
							 "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
							 "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
							 "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
							 "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
							 "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
							 "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
							 "youll": "you'll", "youre": "you're", "youve": "you've"}

        self.manualMap    = { 'none': '0',
							  'zero': '0',
							  'one': '1',
							  'two': '2',
							  'three': '3',
							  'four': '4',
							  'five': '5',
							  'six': '6',
							  'seven': '7',
							  'eight': '8',
							  'nine': '9',
							  'ten': '10'
							}

        self.articles     = ['a',
							 'an',
							 'the'
							]

        self.periodStrip  = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip   = re.compile("(\d)(\,)(\d)")
        self.punct        = [';', r"/", '[', ']', '"', '{', '}',
							 '(', ')', '=', '+', '\\', '_', '-',
							 '>', '<', '@', '`', ',', '?', '!']

        self.n = 2

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }
        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)



    def normalize_answer(self, resAns):
        resAns      = resAns.replace('\n', ' ')
        resAns      = resAns.replace('\t', ' ')
        resAns      = resAns.strip()
        resAns      = self.processPunctuation(resAns)
        resAns      = self.processDigitArticle(resAns)
        resAns = resAns.replace(',', '')
        return resAns

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) != None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = self.periodStrip.sub("",
                                        outText,
                                        re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = ' '.join(outText)
        return outText

    def setEvalQA(self, quesId, acc):
        self.evalQA[quesId] = round(100*acc, self.n)

    def setEvalQuesType(self, quesId, quesType, acc):
        if quesType not in self.evalQuesType:
            self.evalQuesType[quesType] = {}
        self.evalQuesType[quesType][quesId] = round(100*acc, self.n)

    def setEvalAnsType(self, quesId, ansType, acc):
        if ansType not in self.evalAnsType:
            self.evalAnsType[ansType] = {}
        self.evalAnsType[ansType][quesId] = round(100*acc, self.n)

    def setAccuracy(self, accQA):
        self.accuracy['overall'] = round(100*float(sum(accQA))/len(accQA), self.n)


