import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
from sklearn.metrics import jaccard_score
from tqdm import tqdm
import csv
import pickle
import open3d as o3d
from .D3CTTA import D3Ctta
from sklearn.metrics import davies_bouldin_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from utils.losses import CELoss, SoftCELoss, DICELoss, SoftDICELoss, HLoss, SCELoss
from utils.collation import CollateSeparated, CollateStream
from utils.sampler import SequentialSampler
from utils.dataset_online import PairedOnlineDataset, FrameOnlineDataset
from models import MinkUNet18_HEADS, MinkUNet18_SSL, MinkUNet18_MCMC

import sys
sys.path.append('./utils/')
import open3d as o3d

import pickle
torch.cuda.set_device(1)


class OneDomainAdaptation(object):
    r"""
    Segmentation Module for MinkowskiEngine for training on one domain.
    """

    def __init__(self,
                 model,
                 eval_dataset,
                 adapt_dataset,
                 source_model=None,
                 epsilon=0.,
                 seg_beta=1.0,
                 temperature=0.5,
                 stream_batch_size=1,
                 adaptation_batch_size=2,
                 weight_decay=1e-5,
                 momentum=0.8,
                 val_batch_size=6,
                 train_num_workers=10,
                 val_num_workers=10,
                 num_classes=7,
                 clear_cache_int=2,
                 use_random_wdw=False,
                 freeze_list=None,
                 num_mc_iterations=10):

        super().__init__()

        for name, value in list(vars().items()):
            if name != "self":
                setattr(self, name, value)

        self.ignore_label = self.eval_dataset.ignore_label
        
        hparams = {'alpha':0.0001, 'beta':0.9, 'gamma':1, 'wdecay':1e-5}
        self.adapt_method = D3Ctta(num_classes, 5, model, source_model).cuda()

        self.global_step = 0

        self.device = None
        self.max_time_wdw = self.eval_dataset.max_time_wdw
        
        self.topk_matches = 0

        self.dataset_name = self.adapt_dataset.name

        self.total_iou=0




    def freeze(self):
        # here we freeze parts that have to be frozen forever
        if self.freeze_list is not None:
            for name, p in self.model.named_parameters():
                for pf in self.freeze_list:
                    if pf in name:
                        p.requires_grad = False

    def delayed_freeze(self, frame):
        # here we freeze parts that have to be frozen only for a certain period
        if self.delayed_freeze_list is not None:
            for name, p in self.model.named_parameters():
                for pf, frame_act in self.delayed_freeze_list.items():
                    if pf in name and frame <= frame_act:
                        p.requires_grad = False

    
    def adaptation_single_step(self, batch, frame=0, adapt_method=None):

        self.model.eval()

        stensor = ME.SparseTensor(coordinates=batch['coordinates'].int().to(self.device),
                                   features=batch["features"].to(self.device),
                                   quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
        torch.cuda.empty_cache()
            
        labels = batch['labels'].long()
        with torch.no_grad():
            pred_seg = self.adapt_method(stensor, labels)
            # pred_seg = self.model(stensor, is_seg=True).F.detach()
            # pred_seg = pred_seg.argmax(1)

        total_loss = 0


        iou_tmp = jaccard_score(pred_seg.cpu().numpy(), labels.cpu().numpy(), average=None,
                                labels=np.arange(0, self.num_classes),
                                zero_division=-0.1)
        present_labels = np.arange(0, self.num_classes)
        iou_tmp[iou_tmp==1] = 0
        names = [os.path.join('training', n + '_iou') for n in self.adapt_dataset.class2names[present_labels]]
        results_dict = dict(zip(names, iou_tmp[present_labels].tolist()))
        results_dict['training/seg_loss'] = total_loss
        results_dict['training/iou'] = np.mean(iou_tmp[present_labels])

        return results_dict


    def get_online_dataloader(self, dataset, is_adapt=False):
        if is_adapt:
            collate = CollateSeparated(torch.device('cpu'))
            sampler = SequentialSampler(dataset, is_adapt=True, adapt_batchsize=self.adaptation_batch_size,
                                        max_time_wdw=self.max_time_wdw)
            dataloader = DataLoader(dataset,
                                    collate_fn=collate,
                                    sampler=sampler,
                                    pin_memory=False,
                                    num_workers=self.train_num_workers)
        else:
            # collate = CollateFN(torch.device('cpu'))
            collate = CollateStream(torch.device('cpu'))
            sampler = SequentialSampler(dataset, is_adapt=False, adapt_batchsize=self.stream_batch_size)
            dataloader = DataLoader(dataset,
                                    collate_fn=collate,
                                    sampler=sampler,
                                    pin_memory=False,
                                    num_workers=self.train_num_workers)
        return dataloader

    def save_pcd(self, batch, preds, labels, save_path, frame, is_global=False):
        pcd = o3d.geometry.PointCloud()

        if not is_global:
            pts = batch['coordinates']
            pcd.points = o3d.utility.Vector3dVector(pts[:, 1:])
        else:
            pts = batch['global_points'][0]
            pcd.points = o3d.utility.Vector3dVector(pts)
        if self.num_classes == 7 or self.num_classes == 2:
            pcd.colors = o3d.utility.Vector3dVector(self.eval_dataset.color_map[labels+1])
        else:
            pcd.colors = o3d.utility.Vector3dVector(self.eval_dataset.color_map[labels])

        os.makedirs(os.path.join(save_path, 'gt'), exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(save_path, 'gt', str(frame)+'.ply'), pcd)
        if self.num_classes == 7 or self.num_classes == 2:
            pcd.colors = o3d.utility.Vector3dVector(self.eval_dataset.color_map[preds+1])
        else:
            pcd.colors = o3d.utility.Vector3dVector(self.eval_dataset.color_map[preds])

        os.makedirs(os.path.join(save_path, 'preds'), exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(save_path, 'preds', str(frame)+'.ply'), pcd)


class OnlineTrainer(object):

    def __init__(self,
                 pipeline,
                 collate_fn_eval=None,
                 collate_fn_adapt=None,
                 device='cpu',
                 default_root_dir=None,
                 weights_save_path=None,
                 loggers=None,
                 save_checkpoint_every=2,
                 source_checkpoint=None,
                 student_checkpoint=None,
                 boost=True,
                 save_predictions=False,
      #           is_double=True,
      #           is_pseudo=True,
                 use_mcmc=True,
                 sub_epochs=0,
                 corrupt_type=[],
                 level='light'):

        super().__init__()

        if device is not None:
            self.device = torch.device(f'cuda:{device}')
        else:
            self.device = torch.device('cpu')

        self.default_root_dir = default_root_dir
        self.weights_save_path = weights_save_path
        self.loggers = loggers
        self.save_checkpoint_every = save_checkpoint_every
        self.source_checkpoint = source_checkpoint
        self.student_checkpoint = student_checkpoint

        self.pipeline = pipeline
        self.pipeline.device = self.device

   #     self.is_double = is_double
        self.use_mcmc = use_mcmc
        self.model = self.pipeline.model
        self.corrupt_type = corrupt_type
        self.num_domains = len(self.corrupt_type)


        self.eval_dataset = self.pipeline.eval_dataset
        self.adapt_dataset = self.pipeline.adapt_dataset

        self.max_time_wdw = self.eval_dataset.max_time_wdw

        self.eval_dataset.eval()
        self.adapt_dataset.train()

        self.online_sequences = np.arange(self.adapt_dataset.num_sequences())   #scene的个数，即序列个数 int
        self.seq_domains = len(self.online_sequences) // self.num_domains
        self.num_frames = len(self.eval_dataset)
        # self.seq_domains = len(self.online_sequences) // 7


        self.collate_fn_eval = collate_fn_eval
        self.collate_fn_adapt = collate_fn_adapt
        self.collate_fn_eval.device = self.device
        self.collate_fn_adapt.device = self.device

        self.sequence = -1

        self.adaptation_results_dict = {s: [] for s in np.arange(self.seq_domains)}
        self.source_results_dict = {s: [] for s in np.arange(self.seq_domains)}

        self.eval_dataloader = None
        self.adapt_dataloader = None
        self.level = level
        self.eval_dataset.level = level
        self.adapt_dataset.level = level
        self.corrupt = None

        self.boost = boost

        self.save_predictions = save_predictions

        self.sub_epochs = sub_epochs
        self.num_classes = self.pipeline.num_classes

        self.dataset_name = self.pipeline.dataset_name
        

    def adapt_double(self):

        self.load_source_model()
        self.pipeline.prompt_t3a.copy_model()

        for i, corrupt in enumerate(self.corrupt_type):
            self.corrupt = corrupt
            self.eval_dataset.corrupt = corrupt
            self.adapt_dataset.corrupt = corrupt

            for sequence in tqdm(np.arange(self.seq_domains), desc='Online Adaptation'):
                sequence_glob = i * self.seq_domains + sequence
                self.set_sequence(corrupt, sequence_glob)
            # adapt on sequence
                sequence_dict = self.online_adaptation_routine()

                self.adaptation_results_dict[sequence] = sequence_dict
            self.save_final_results(i)



    def eval(self, is_adapt=False):
        # load model only once
#         self.reload_model(is_adapt=False)

        for i, corrupt in enumerate(self.corrupt_type):
            self.corrupt = corrupt
            self.eval_dataset.corrupt = corrupt
#             self.adapt_dataset.corrupt = corrupt
            for sequence in tqdm(np.arange(self.seq_domains), desc='Online Evaluation', leave=True):
                # set sequence
                sequence_glob = i * self.seq_domains + sequence
                self.set_sequence(corrupt, sequence_glob)
                # evaluate
                sequence_dict = self.online_evaluation_routine()
                # store dict
                self.source_results_dict[sequence] = sequence_dict
        # if not is_adapt:
            self.save_eval_results(i)

    def check_frame(self, fr):
        return (fr+1) >= self.pipeline.adaptation_batch_size and fr >= self.max_time_wdw

    def online_adaptation_routine(self):
        # move to device
        self.model.to(self.device)
        adaptation_results = []

        if self.save_predictions:
            save_path = os.path.join(self.weights_save_path, 'pcd')
        else:
            save_path = None

        for f in tqdm(range(len(self.eval_dataset)), desc=f'Seq: {self.sequence}', leave=True):
              
            train_dict = {}
            batch = self.get_adaptation_batch(f)

                #利用当前帧进行adaptation
                
            for _ in range(self.sub_epochs):

                train_dict = self.pipeline.adaptation_single_step(batch, f) 

                train_dict['validation/frame'] = f

                if train_dict is not None:
                    train_dict.update(train_dict)
            adaptation_results.append(train_dict)

        return adaptation_results

    def online_evaluation_routine(self):
        # move model to device
        self.model.to(self.device)
        # for store
        source_results = []

        if self.save_predictions:
            save_path = os.path.join(self.weights_save_path, 'pcd')
        else:
            save_path = None

        with torch.no_grad():
            for f in tqdm(range(len(self.eval_dataset)), desc=f'Seq: {self.sequence}', leave=True):
                # get eval batch
                val_batch = self.get_evaluation_batch(f)
                # eval
                val_dict = self.pipeline.validation_step(val_batch, is_source=True, save_path=save_path, frame=f)
                val_dict['source/frame'] = f
                # store results
                # self.log(val_dict)
                source_results.append(val_dict)

        return source_results

    def set_loggers(self, sequence):
        # set current sequence in loggers, for logging purposes
        for logger in self.loggers:
            logger.set_sequence(sequence)

    def set_sequence(self, corrupt, sequence):
        # update current weight saving path
        self.sequence = str(sequence)
        path, _ = os.path.split(self.weights_save_path)
        self.weights_save_path = os.path.join(path, self.sequence)
        os.makedirs(self.weights_save_path, exist_ok=True)

        self.eval_dataset.set_sequence(sequence)      #self.selected_sequence：当前scene的token， self.token_list：当前scane中sample的token list, self.location:当前scene的地点
        self.adapt_dataset.set_sequence(sequence)

        if self.boost:
            self.eval_dataloader = iter(self.pipeline.get_online_dataloader(FrameOnlineDataset(self.eval_dataset),
                                                                            is_adapt=False))
            self.adapt_dataloader = iter(self.pipeline.get_online_dataloader(FrameOnlineDataset(self.adapt_dataset),
                                                                            is_adapt=False))



    def log(self, results_dict):
        # log in ach logger
        for logger in self.loggers:
            logger.log(results_dict)

    def save_state_dict(self, frame):
        # save stat dict of the model
        save_dict = {'frame': frame,
                     'model_state_dict': self.model.state_dict()}
        torch.save(save_dict, os.path.join(self.weights_save_path, f'checkpoint-frame{frame}.pth'))

    def reload_model(self, is_adapt=True):
        # reloads model
        def clean_state_dict(state):
            # clean state dict from names of PL
            for k in list(ckpt.keys()):
                if "model" in k:
                    ckpt[k.replace("model.", "")] = ckpt[k]
                del ckpt[k]
            return state

        if self.student_checkpoint is not None and is_adapt:
            checkpoint_path = self.student_checkpoint
            print(f'--> Loading student checkpoint {checkpoint_path}')
        else:
            checkpoint_path = self.source_checkpoint
            print(f'--> Loading source checkpoint {checkpoint_path}')

        # in case of SSL pretraining
        if isinstance(self.model, MinkUNet18_SSL):
            if checkpoint_path.endswith('.pth'):
                ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                self.model.load_state_dict(ckpt)

            elif checkpoint_path.endswith('.ckpt'):
                ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))["state_dict"]
                ckpt = clean_state_dict(ckpt)
                self.model.load_state_dict(ckpt, strict=True)

            else:
                raise NotImplementedError('Invalid source model extension (allowed .pth and .ckpt)')

        # in case of segmentation pretraining
        elif isinstance(self.model, MinkUNet18_HEADS):
            def clean_student_state_dict(ckpt):
                # clean state dict from names of PL
                for k in list(ckpt.keys()):
                    if "seg_model" in k:
                        ckpt[k.replace("seg_model.", "")] = ckpt[k]
                    del ckpt[k]
                return ckpt
            if checkpoint_path.endswith('.pth'):
                ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                ckpt = clean_student_state_dict(ckpt['model_state_dict'])
                self.model.seg_model.load_state_dict(ckpt)

            elif checkpoint_path.endswith('.ckpt'):
                ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))["state_dict"]
                ckpt = clean_state_dict(ckpt)
                self.model.seg_model.load_state_dict(ckpt, strict=True)

    def reload_model_from_scratch(self):

        # in case of SSL pretraining
        if isinstance(self.model, MinkUNet18_SSL):
            self.model.weight_initialization()

        # in case of segmentation pretraining
        elif isinstance(self.model, MinkUNet18_HEADS):
            seg_model = self.model.seg_model
            seg_model.weight_initialization()
            self.model = MinkUNet18_HEADS(seg_model=seg_model)

    def load_source_model(self):
        # reloads model
        def clean_state_dict(state):
            # clean state dict from names of PL
            for k in list(ckpt.keys()):
                if "model" in k:
                    ckpt[k.replace("model.", "")] = ckpt[k]
                del ckpt[k]
            return state

        print(f'--> Loading source checkpoint {self.source_checkpoint}')

        if self.source_checkpoint.endswith('.pth'):
            ckpt = torch.load(self.source_checkpoint, map_location=torch.device('cpu'))
            if isinstance(self.model, MinkUNet18_MCMC):
                self.model.seg_model.load_state_dict(ckpt)
            else:
                self.model.load_state_dict(ckpt)

        elif self.source_checkpoint.endswith('.ckpt'):
            ckpt = torch.load(self.source_checkpoint, map_location=torch.device('cpu'))["state_dict"]
            ckpt = clean_state_dict(ckpt)
            if isinstance(self.model, MinkUNet18_MCMC):
                self.model.seg_model.load_state_dict(ckpt, strict=True)
            else:
                self.model.load_state_dict(ckpt, strict=True)

        else:
            raise NotImplementedError('Invalid source model extension (allowed .pth and .ckpt)')

    def get_adaptation_batch(self, frame_idx):
        if self.adapt_dataloader is None:
            frame_idx += 1
            batch_idx = np.arange(frame_idx - self.pipeline.adaptation_batch_size, frame_idx)
            batch_data = [self.adapt_dataset.__getitem__(b) for b in batch_idx]    
            ''' {'points': points,'features': colors,'labels': label,'global_points': global_pts,'geometric_feats': geometric_feats}'''
#             batch_data = [self.adapt_dataset.get_double_data(batch_data[b-1], batch_data[b]) for b in range(1, len(batch_data))]
            batch_data = [self.adapt_dataset.get_single_data(batch_data[b]) for b in range(0, len(batch_data))]
            """
            'coordinates_all': coords_all.int(), 'coordinates': coords.int(),  'features': colors.float(), 'geometric_feats': geometric_feats.float(),  'labels': labels,
                'next_coordinates': next_coords.int(), 'next_features': next_colors.float(), 'next_labels': next_labels, 'matches0': matches0.int(), 'matches1': matches1.int(),
                'num_pts0': num_pts0,'num_pts1': num_pts1,'sampled_idx': sampled_idx}
            """
            batch = self.collate_fn_adapt(batch_data)
            # change list of dict into dict of list:e.g. [{data1},{data2}]->{'coords':[],'feat':[]...}
        else:
            batch = next(self.adapt_dataloader)

        return batch

    def get_evaluation_batch(self, frame_idx):
        if self.eval_dataloader is None:
            data = self.eval_dataset.__getitem__(frame_idx)
            data = self.eval_dataset.get_single_data(data)

            batch = self.collate_fn_eval([data])
        else:
            batch = next(self.eval_dataloader)
        return batch

    def save_final_results(self, seq_id):
        # stores final results in a final dict
        # finally saves results in a csv file

        final_dict = {}

        for seq in np.arange(self.seq_domains):
            # source_results = self.source_results_dict[seq]
            adaptation_results = self.adaptation_results_dict[seq]      

            # assert len(source_results) == len(adaptation_results)
            num_frames = len(adaptation_results)

            # source_results = self.format_val_dict(source_results)
            adaptation_results = self.format_val_dict(adaptation_results)
            
            final_dict[seq] = {}

            for k in adaptation_results.keys():
                # relative_tmp = adaptation_results[k] - source_results[k]
                # final_dict[seq][f'relative_{k}'] = relative_tmp
                # final_dict[seq][f'source_{k}'] = source_results[k]
                final_dict[seq][f'adapted_{k}'] = adaptation_results[k]

        # self.write_csv(final_dict, phase='final')
        # self.write_csv(final_dict, phase='source')
        self.write_csv(seq_id, final_dict, phase='adapt')
        self.save_pickle(final_dict)

    def save_eval_results(self):
        # stores final results in a final dict
        # finally saves results in a csv file

        final_dict = {}

        for seq in np.arange(self.seq_domains):
            eval_results = self.source_results_dict[seq]

            eval_results = self.format_val_dict(eval_results)

            final_dict[seq] = {}

            for k in eval_results.keys():
                final_dict[seq][f'eval_{k}'] = eval_results[k]

        self.write_csv(seq_id, final_dict, phase='eval')
        self.save_pickle(final_dict)

    def format_val_dict(self, list_dict):
        # input is a list of dicts for each frame
        # returns a dict with [miou, iou_per_frame, per_class_miou, per_class_iou_frame]

        def change_names(in_dict):
            for k in list(in_dict.keys()):
                if "training/" in k:
                    in_dict[k.replace("training/", "")] = in_dict[k]
                    del in_dict[k]
                elif "source/" in k:
                    in_dict[k.replace("source/", "")] = in_dict[k]
                    del in_dict[k]
                elif "validation" in k:
                    in_dict[k.replace("validation/", "")] = in_dict[k]
                    del in_dict[k]

            return in_dict

        list_dict = [change_names(list_dict[f]) for f in range(len(list_dict))]

        if self.num_classes == 7:
            classes = {'vehicle_iou': [],
                       'pedestrian_iou': [],
                       'road_iou': [],
                       'sidewalk_iou': [],
                       'terrain_iou': [],
                       'manmade_iou': [],
                       'vegetation_iou': []}
        elif self.num_classes == 3:
            classes = {'background_iou': [],
                       'vehicle_iou': [],
                       'pedestrian_iou': []}
        else:
            classes = {'vehicle_iou': [],
                       'pedestrian_iou': []}

        for f in range(len(list_dict)):
            val_tmp = list_dict[f]
            for key in classes.keys():
                if key in val_tmp:
                    classes[key].append(val_tmp[key])
                else:
                    classes[key].append(np.nan)

        all_iou = np.concatenate([np.asarray(v)[np.newaxis, ...] for k, v in classes.items()], axis=0).T

        per_class_iou = np.nanmean(all_iou, axis=0)
        miou = np.nanmean(per_class_iou)

        per_frame_miou = np.nanmean(all_iou, axis=-1)

        return {'miou': miou,
                'per_frame_miou': per_frame_miou,
                'per_class_iou': per_class_iou,
                'per_class_frame_iou': all_iou}

    def write_csv(self, seq_id, results_dict, phase='final'):
        if self.num_classes == 7:
            if phase == 'final':
                headers = ['sequence', 'relative_miou', 'relative_vehicle_iou',
                           'relative_pedestrian_iou', 'relative_road_iou',
                           'relative_sidewalk_iou', 'relative_terrain_iou',
                           'relative_manmade_iou', 'relative_vegetation_iou']
                file_name = 'final_main.csv'
            elif phase == 'source':
                headers = ['sequence', 'miou', 'source_vehicle_iou',
                           'source_pedestrian_iou', 'source_road_iou',
                           'source_sidewalk_iou', 'source_terrain_iou',
                           'source_manmade_iou', 'source_vegetation_iou']
                file_name = 'source_main.csv'
            elif phase == 'eval':
                headers = ['sequence', 'miou', 'eval_vehicle_iou',
                           'eval_pedestrian_iou', 'eval_road_iou',
                           'eval_sidewalk_iou', 'eval_terrain_iou',
                           'eval_manmade_iou', 'eval_vegetation_iou']
                file_name = 'evaluation_main.csv'
            elif phase == 'adapt':
                headers = ['sequence', 'adapted_miou', 'adapted_vehicle_iou',
                           'adapted_pedestrian_iou', 'adapted_road_iou',
                           'adapted_sidewalk_iou', 'adapted_terrain_iou',
                           'adapted_manmade_iou', 'adapted_vegetation_iou']
                file_name = 'adapted_main.csv'
            else:
                raise NotImplementedError
        elif self.num_classes == 3:
            if phase == 'final':
                headers = ['sequence', 'relative_miou',
                           'relative_background_iou',
                           'relative_vehicle_iou',
                           'relative_pedestrian_iou']
                file_name = 'final_main.csv'
            elif phase == 'source':
                headers = ['sequence', 'miou',
                           'source_background_iou',
                           'source_vehicle_iou',
                           'source_pedestrian_iou']
                file_name = 'source_main.csv'
            elif phase == 'eval':
                headers = ['sequence','miou',
                           'source_backround_iou',
                           'eval_vehicle_iou',
                           'eval_pedestrian_iou']
                file_name = 'evaluation_main.csv'
            elif phase == 'adapt':
                headers = ['sequence','miou',
                           'adapted_backround_iou',
                           'adapted_vehicle_iou',
                           'adapted_pedestrian_iou']
                file_name = 'adapted_main.csv'
            else:
                raise NotImplementedError
        elif self.num_classes == 2:
            if phase == 'final':
                headers = ['sequence', 'relative_miou',
                           'relative_vehicle_iou',
                           'relative_pedestrian_iou']
                file_name = 'final_main.csv'
            elif phase == 'source':
                headers = ['sequence', 'miou',
                           'source_vehicle_iou',
                           'source_pedestrian_iou']
                file_name = 'source_main.csv'
            elif phase == 'eval':
                headers = ['sequence','miou',
                           'eval_vehicle_iou',
                           'eval_pedestrian_iou']
                file_name = 'evaluation_main.csv'
            elif phase == 'adapt':
                headers = ['sequence', 'miou',
                           'adapted_vehicle_iou',
                           'adapted_pedestrian_iou']
                file_name = 'adapted_main.csv'
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        if self.dataset_name == 'nuScenes':
            cumul = []

#         results_dir = os.path.join(os.path.split(self.weights_save_path)[0], 'final_results')
        results_dir = os.path.join(os.path.split(self.weights_save_path)[0], self.corrupt, str(seq_id))
        os.makedirs(results_dir, exist_ok=True)
        
        with open(os.path.join(results_dir, file_name), 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(headers)

            for seq in results_dict.keys():
                dict_tmp = results_dict[seq]
                if phase == 'final':
                    per_class = dict_tmp['relative_per_class_iou']
                    if self.num_classes == 7:
                        data = [seq,
                                dict_tmp['relative_miou']*100,
                                per_class[0]*100,
                                per_class[1]*100,
                                per_class[2]*100,
                                per_class[3]*100,
                                per_class[4]*100,
                                per_class[5]*100,
                                per_class[6]*100]
                    elif self.num_classes == 3:
                        data = [seq,
                                dict_tmp['relative_miou']*100,
                                per_class[0]*100,
                                per_class[1]*100,
                                per_class[2]*100]
                    elif self.num_classes == 2:
                        data = [seq,
                                dict_tmp['relative_miou']*100,
                                per_class[0]*100,
                                per_class[1]*100]

                elif phase == 'source':
                    per_class = dict_tmp['source_per_class_iou']
                    if self.num_classes == 7:
                        data = [seq,
                                dict_tmp['source_miou']*100,
                                per_class[0]*100,
                                per_class[1]*100,
                                per_class[2]*100,
                                per_class[3]*100,
                                per_class[4]*100,
                                per_class[5]*100,
                                per_class[6]*100]
                    elif self.num_classes == 3:
                        data = [seq,
                                dict_tmp['source_miou']*100,
                                per_class[0]*100,
                                per_class[1]*100,
                                per_class[2]*100]

                    elif self.num_classes == 2:
                        data = [seq,
                                dict_tmp['source_miou']*100,
                                per_class[0]*100,
                                per_class[1]*100]

                elif phase == 'eval':
                    per_class = dict_tmp['eval_per_class_iou']
                    if self.num_classes == 7:
                        data = [seq,
                                dict_tmp['eval_miou']*100,
                                per_class[0]*100,
                                per_class[1]*100,
                                per_class[2]*100,
                                per_class[3]*100,
                                per_class[4]*100,
                                per_class[5]*100,
                                per_class[6]*100]

                    elif self.num_classes == 3:
                        data = [seq,
                                dict_tmp['eval_miou']*100,
                                per_class[0]*100,
                                per_class[1]*100,
                                per_class[2]*100]
                    
                    elif self.num_classes == 2:
                        data = [seq,
                                dict_tmp['eval_miou']*100,
                                per_class[0]*100,
                                per_class[1]*100]
                        
                elif phase == 'adapt':
                    per_class = dict_tmp['adapted_per_class_iou']
                    if self.num_classes == 7:
                        data = [seq,
                                dict_tmp['adapted_miou']*100,
                                per_class[0]*100,
                                per_class[1]*100,
                                per_class[2]*100,
                                per_class[3]*100,
                                per_class[4]*100,
                                per_class[5]*100,
                                per_class[6]*100]
                    elif self.num_classes == 3:
                        data = [seq,
                                dict_tmp['adapted_miou']*100,
                                per_class[0]*100,
                                per_class[1]*100,
                                per_class[2]*100]

                    elif self.num_classes == 2:
                        data = [seq,
                                dict_tmp['adapted_miou']*100,
                                per_class[0]*100,
                                per_class[1]*100]

                
                # write the data
                writer.writerow(data)

                if self.dataset_name == 'nuScenes':
                    if phase == 'final':
                        first_iou = dict_tmp['relative_miou']
                    elif phase == 'source':
                        first_iou = dict_tmp['source_miou']
                    elif phase == 'eval':
                        first_iou = dict_tmp['eval_miou']
                    elif phase == 'adapt':
                        first_iou = dict_tmp['adapted_miou']

                    if self.num_classes == 7:
                        cumul.append([first_iou*100,
                                      per_class[0]*100,
                                      per_class[1]*100,
                                      per_class[2]*100,
                                      per_class[3]*100,
                                      per_class[4]*100,
                                      per_class[5]*100,
                                      per_class[6]*100])
                    elif self.num_classes == 3:
                        cumul.append([first_iou*100,
                                      per_class[0]*100,
                                      per_class[1]*100,
                                      per_class[2]*100])
                    elif self.num_classes == 2:
                        cumul.append([first_iou*100,
                                      per_class[0]*100,
                                      per_class[1]*100])

            if self.dataset_name == 'nuScenes':
                avg_cumul = np.array(cumul)
                avg_cumul_tmp = np.nanmean(avg_cumul, axis=0)
                if self.num_classes == 7:
                    data = ['Average',
                            avg_cumul_tmp[0],
                            avg_cumul_tmp[1],
                            avg_cumul_tmp[2],
                            avg_cumul_tmp[3],
                            avg_cumul_tmp[4],
                            avg_cumul_tmp[5],
                            avg_cumul_tmp[6],
                            avg_cumul_tmp[7]]
                elif self.num_classes == 3:
                    data = ['Average',
                            avg_cumul_tmp[0],
                            avg_cumul_tmp[1],
                            avg_cumul_tmp[2]]

                elif self.num_classes == 2:
                    data = ['Average',
                            avg_cumul_tmp[0],
                            avg_cumul_tmp[1]]

                # write cumulative results
                writer.writerow(data)
                seq_locs = np.array([self.adapt_dataset.names2locations[self.adapt_dataset.online_keys[s]] for s in results_dict.keys()])

                for location in ['singapore-queenstown', 'boston-seaport', 'singapore-hollandvillage', 'singapore-onenorth']:
                    valid_sequences = seq_locs == location
                    avg_cumul_tmp = np.nanmean(avg_cumul[valid_sequences, :], axis=0)
                    if self.num_classes == 7:
                        data = [location,
                            avg_cumul_tmp[0],
                            avg_cumul_tmp[1],
                            avg_cumul_tmp[2],
                            avg_cumul_tmp[3],
                            avg_cumul_tmp[4],
                            avg_cumul_tmp[5],
                            avg_cumul_tmp[6],
                            avg_cumul_tmp[7]]

                    elif self.num_classes == 3:
                        data = [location,
                            avg_cumul_tmp[0],
                            avg_cumul_tmp[1],
                            avg_cumul_tmp[2]]

                    elif self.num_classes == 2:
                        data = [location,
                            avg_cumul_tmp[0],
                            avg_cumul_tmp[1]]

                    # write cumulative results
                    writer.writerow(data)

    def save_pickle(self, results_dict):
        results_dir = os.path.join(os.path.split(self.weights_save_path)[0], self.corrupt)
        with open(os.path.join(results_dir, 'final_all.pkl'), 'wb') as handle:
            pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
