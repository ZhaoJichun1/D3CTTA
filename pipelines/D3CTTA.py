from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
import torch.jit
import MinkowskiEngine as ME
import torch.nn.functional as F
import pickle as pkl
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
# from utils.losses import CELoss
from utils.distribution import *
from sklearn.decomposition import PCA
import math
from sklearn.metrics import jaccard_score
from sklearn.neighbors import NearestNeighbors
from models import DistanceBasedBatchNorm
import open3d as o3d

torch.cuda.set_device(1)

def softmax_entropy(x):
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)



class D3Ctta(nn.Module):

    def __init__(self, num_classes, hparams, algorithm, source_model):
        super().__init__()
        self.featurizer = algorithm
        self.classifier = torch.nn.functional.normalize(algorithm.final.kernel, dim=0).cuda()
        self.source_model = source_model 
        #7*96
        warmup_supports = algorithm.final.kernel.T
        self.warmup_supports = algorithm.final.kernel

        self.w_rand = torch.randn(96, 1024).cuda()
        # self.Q = torch.zeros(1024, num_classes).cuda()
        # self.G =torch.zeros(1024, 1024).cuda()
        self.Q = []
        self.G = []

        self.filter_K = hparams
        self.num_classes = num_classes

        self.alpha = 0.95
        self.min_feat = 1

        self.key_pool = warmup_supports.T.unsqueeze(0).cuda()
        self.update_count = [1]

        self.prev_mean = torch.ones(256).cuda()
        self.prev_domain = -1
        self.domain_params = []
        self.tau_c = 0.85

        self.batch_pred = []
        self.batch_feat= []
        self.batch_size = 1
        self.select_ratio = [0.05]*self.num_classes
        self.ratio_list = []
        self.first = 0

        self.nbrs = NearestNeighbors(n_neighbors=20)
        self.num_areas_d = 3
        self.num_areas_h = 3
        self.num_areas_n = 3
        self.num_areas_x = 3
        self.num_areas_y = 3
        self.proto = [deepcopy(self.warmup_supports).data.T] * 3


    def forward(self, x, y):


        self.source_model.train()
        coords = x.C.cpu().numpy()[:, 1:]
        self.nbrs = self.nbrs.fit(coords)
        _, near_indices = self.nbrs.kneighbors(coords)

        # feat, bottle = self.featurizer(x, is_seg=False)
        # bottle = bottle.F.detach()

        # domain agnoised prototype(T3A)
        # pred_source = feat.F @ self.classifier
        # print((pred_source.argmax(1)==y.cuda()).sum())
        feat_source, bottle = self.source_model(x, is_seg=False)
        pred_source = self.source_model.final(feat_source).F
        feat_source = feat_source.F
        ent = softmax_entropy(pred_source)

        neighbor_preds = pred_source.argmax(1)[near_indices]  # 形状为 (N, k)
        current_preds = pred_source.argmax(1)[:, np.newaxis]  # 转换为形状 (N, 1) 以便与邻居标签广播
        consistency_scores = np.mean(np.array((neighbor_preds == current_preds).detach().cpu()), axis=1)

        consistency_scores = np.array(consistency_scores)
        consistent_indices = consistency_scores > 0.8
        ratio = consistent_indices.sum()/len(consistent_indices)
        indices_ent, indices_prob = self.select_pseudo(pred_source, ent, ratio)
        

        g_index, m_index = self.prior_filter(pred_source, coords)

        indices_filter = consistent_indices & g_index & m_index
        indices_parts = self.distance_partion(x.C)

        pred_proto = torch.ones_like(pred_source)
        for i in range(self.num_areas_d):
            indices = indices_parts[i]
            pred_proto[indices] = feat_source[indices] @ torch.nn.functional.normalize(self.proto[i], dim=1).T.cuda()
            self.update_proto_multi(pred_source.argmax(1)[indices][indices_filter[indices]], feat_source[indices][indices_filter[indices]], i)
         

        domain = self.detect_domain_shift(bottle.cpu(), self.prev_mean)
        if domain == -1:
            self.Q.append(torch.zeros(1024, self.num_classes).cuda())
            self.G.append(torch.zeros(1024, 1024).cuda())

        Q = self.Q[domain].clone()
        G = self.G[domain].clone()
        feat_h = torch.nn.functional.relu(feat_source@self.w_rand)

        rand_num = np.random.random(1)
        feat_h = torch.nn.functional.relu(feat_source@self.w_rand)
        neighbor_preds = pred_proto.argmax(1)[near_indices]
        current_preds = pred_proto.argmax(1)[:, np.newaxis] 
        consistency_scores = np.mean(np.array((neighbor_preds == current_preds).detach().cpu()), axis=1)

        consistency_scores = np.array(consistency_scores)
        consistent_indices = consistency_scores > 0.8

        g_index, m_index = self.prior_filter(pred_proto, coords)

        indices_filter = consistent_indices & g_index & m_index

        yhat = torch.nn.functional.one_hot(pred_proto.argmax(1), num_classes=self.num_classes).float()
        final_indices = (torch.nonzero(torch.tensor(indices_filter)).squeeze())

        Q = Q + feat_h[final_indices].T @ yhat[final_indices]
        G = G + feat_h[final_indices].T @ feat_h[final_indices]
        self.Q[domain] = Q.clone()
        self.G[domain] = G.clone()
        ridge = self.optimise_ridge_parameter(feat_h[final_indices].detach().cpu(), yhat[final_indices].detach().cpu())
        wo=torch.linalg.solve(G+ridge*torch.eye(G.size(dim=0)).cuda(), Q).T
        pred_domain = feat_h @ wo.T

        return pred_domain.argmax(1)

    
    def normalize_logits(self, logits):
        max_logits = torch.max(logits)
        min_logits = torch.min(logits)
        return (logits - min_logits)/(max_logits-min_logits)


    def prior_filter(self, pred, points):
        # points = points[:, 1:]
        pred = pred.argmax(1).detach().cpu().numpy()
        pcd = o3d.geometry.PointCloud()  # 传入3d点云格式
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=50, max_nn=100))
        normals = np.fabs(np.asarray(pcd.normals)[:, 2])
        plane_norm_index = normals > 0.9
        manmade_norm_index = normals < 0.1
        height_index = points[:, 2] < -10
        plane_pred_index = ((pred == 2) | (pred == 3) | (pred==4))
        manmade_pred_index = (pred == 5)
        other_index = (pred == 0) | (pred == 1) | (pred == 6)
        ground_index = plane_pred_index & plane_norm_index & height_index
        manmade_index = manmade_pred_index & manmade_norm_index

        return ground_index | other_index | manmade_pred_index, manmade_index | other_index | plane_pred_index

    
    def update_proto_multi(self, pred_proto, feat, area):

        pred_label = pred_proto
        for i in range(self.num_classes):
            index_class = (pred_label == i)
            feat_i = feat[index_class].detach().cpu()
            if feat_i.shape[0] < self.min_feat:
                continue
            mean = torch.mean(feat_i, dim=0)
            self.proto[area][i] = self.alpha * self.proto[area][i] + (1-self.alpha) * mean

    

    def distance_partion(self, points):
        points = points[:, 1:]

        distance = torch.sqrt(points[:, 0]**2 + points[:, 1]**2)
        distance = torch.clamp(distance, 0+1e-3, 1000-1e-3)
        distance_list = np.linspace(0, 1000, self.num_areas_d + 1)

        # height
        height = torch.clamp(points[:, 2], -50+1e-3, 20-1e-3)
        height_list = np.linspace(-50, 20, self.num_areas_h + 1)

        # x
        x_coor = torch.clamp(points[:, 1], -1000+1e-3, 1000-1e-3)
        x_list = np.linspace(-1000, 1000, self.num_areas_x + 1)

        # y
        y_coor = torch.clamp(points[:, 0], -1000+1e-3, 1000-1e-3)
        y_list = np.linspace(-1000, 1000, self.num_areas_y + 1) 

        #norm
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.detach().cpu().numpy())
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=50, max_nn=100))
        normals = np.asarray(pcd.normals)
        normals = np.fabs(np.asarray(normals)[:, 2])
        normals_list = np.array([-1e-3, 0.2, 0.9, 1.0001])


        distance_labels = np.digitize(distance.detach().cpu().numpy(), bins=distance_list)-1
        height_labels = np.digitize(height.detach().cpu().numpy(), bins=height_list)-1
        x_labels = np.digitize(x_coor.detach().cpu().numpy(), bins=x_list)-1
        y_labels = np.digitize(y_coor.detach().cpu().numpy(), bins=y_list)-1
        norm_labels = np.digitize(normals, bins=normals_list)-1


        # labels = distance_labels * self.num_areas_d + x_labels
        labels = distance_labels
        idx_all = []
        for i in range(self.num_areas_d):
            idx_all.append(list(np.where(labels==i)[0]))
        return idx_all


    def copy_model(self):

        model_state = deepcopy(self.featurizer.state_dict())
        self.source_model.load_state_dict(model_state)
        for param in self.source_model.parameters():
            param.detach_()


    def detect_domain_shift(self, f_feat, mu_t_prev):
        def compute_bn_stats(f_feat):
            mu_t = torch.mean(f_feat, dim=0)
            sigma_t = torch.var(f_feat, dim=0)
            return mu_t, sigma_t
        def cosine_similarity(mu_t, mu_t_prev):
            return F.cosine_similarity(mu_t, mu_t_prev, dim=0)

        def wasserstein_dist(mu_t, sigma_t, mu_d, sigma_d):
            return torch.abs(mu_t - mu_d) + torch.abs(torch.sqrt(sigma_t) - torch.sqrt(sigma_d))

        def update_bn_params(mu_d, sigma_d, mu_t, sigma_t, t):
            mu_d = mu_d + (mu_t - mu_d) / t
            sigma_d = sigma_d + (sigma_t - sigma_d) / t
            return mu_d, sigma_d
        
        mu_t, sigma_t = compute_bn_stats(f_feat)
        c = cosine_similarity(mu_t, mu_t_prev)
        self.prev_mean = mu_t
        
        if c > self.tau_c:
            # Domain remains unchanged, update parameters
            self.domain_params[self.prev_domain]['mu'], self.domain_params[self.prev_domain]['sigma'] = update_bn_params(
                self.domain_params[self.prev_domain]['mu'], self.domain_params[self.prev_domain]['sigma'], mu_t, sigma_t, self.domain_params[self.prev_domain]['t']
            )
            self.domain_params[self.prev_domain]['t'] += 1
            return self.prev_domain
        else:
            # Domain shift detected, check if new domain exists
            min_dist = float('inf')
            assigned_domain = None
            
            for i, params in enumerate(self.domain_params):
                dist = wasserstein_dist(mu_t, sigma_t, params['mu'], params['sigma']).sum()
                if dist < min_dist:
                    min_dist = dist
                    assigned_domain = i
            
            if min_dist < self.tau_c:
                self.domain_params[assigned_domain]['t'] += 1
                self.domain_params[assigned_domain]['mu'], self.domain_params[assigned_domain]['sigma'] = update_bn_params(
                    self.domain_params[assigned_domain]['mu'], self.domain_params[assigned_domain]['sigma'], mu_t, sigma_t, self.domain_params[assigned_domain]['t']
                )
                self.prev_domain = assigned_domain
                return assigned_domain
            else:
                # Initialize new domain
                self.domain_params.append({'mu': mu_t.clone(), 'sigma': sigma_t.clone(), 't': 1})
                self.prev_domain = len(self.domain_params) - 1
                return -1


    def select_pseudo(self, pred_seg, ent, ratio=1):
        selected_indices = []

        prob = 1-pred_seg.softmax(1).max(1).values
        # ent = prob

        pred_seg = pred_seg.argmax(1)

        for label in range(self.num_classes):
            label_indices = torch.nonzero(pred_seg == label).squeeze()
            
            label_entropy = ent[label_indices]

            if label_indices.numel() == 1:
                # print('!!!!!!!!')
                continue
            
            sorted_label_indices = label_indices[torch.argsort(label_entropy)]
            num_selected = math.ceil(ratio * len(sorted_label_indices))
            selected_label_indices = sorted_label_indices[:num_selected]
            selected_indices.append(selected_label_indices)
        index_ent = torch.cat(selected_indices)

        for label in range(self.num_classes):
            label_indices = torch.nonzero(pred_seg == label).squeeze()
            
            label_entropy = prob[label_indices]

            if label_indices.numel() == 1:
                continue
            
            sorted_label_indices = label_indices[torch.argsort(label_entropy)]
            num_selected = math.ceil(ratio * len(sorted_label_indices))
            selected_label_indices = sorted_label_indices[:num_selected]
            selected_indices.append(selected_label_indices)
        index_prob = torch.cat(selected_indices)
        # count = torch.zeros(7)
        # count2 = torch.zeros(7)
        # for i in range(index_ent.shape[0]):
        #     count[pred_seg[index_ent][i]] += 1
        # index = index_ent[correct_index[index_ent]]
        # for i in range(index.shape[0]):
        #     count2[pred_seg[index][i]] += 1
        # # print(count)
        # # print(count2)
        # # input()
        # index2 = []
        # for i in range(len(count2)):
        #     class_indices = np.where(y == i)[0]
        #     if count2[i] == 0:
        #         continue
        #     index2.extend(np.random.choice(class_indices, int(count2[i]), replace=False))
#         # index = index2
        return index_ent, index_prob


    def select_pseudo_multi(self, pred_seg, ent, ratio=1):


        selected_indices = []
        # prob = pred_seg.softmax(1).max(1).values

        for label in range(self.num_classes):
            label_indices = torch.nonzero(pred_seg == label).squeeze()
            
            label_entropy = ent[label_indices]

            if label_indices.numel() == 1:
                # print('!!!!!!!!')
                continue
            
            sorted_label_indices = label_indices[torch.argsort(label_entropy)]
            num_selected = math.ceil(ratio[label] * len(sorted_label_indices))
            selected_label_indices = sorted_label_indices[:num_selected]
            selected_indices.append(selected_label_indices)
        index_ent = torch.cat(selected_indices)
        return index_ent
    
    


    def update_proto(self, pred_proto, feat):

        pred_label = pred_proto
        for i in range(self.num_classes):
            index_class = (pred_label == i)
            feat_i = feat[index_class].detach().cpu()
            if feat_i.shape[0] < self.min_feat:
                continue
            mean = torch.mean(feat_i, dim=0)
            self.proto[i] = self.alpha * self.proto[i] + (1-self.alpha) * mean


    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s))))

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s))))
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat==i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]
        
        return self.supports, self.labels


    def optimise_ridge_parameter(self,Features,Y):
        ridges=10.0**np.arange(-8,9)
        num_val_samples=int(Features.shape[0]*0.8)
        losses=[]
        Q_val=Features[0:num_val_samples,:].T @ Y[0:num_val_samples,:]
        G_val=Features[0:num_val_samples,:].T @ Features[0:num_val_samples,:]
        for ridge in ridges:
            Wo=torch.linalg.solve(G_val+ridge*torch.eye(G_val.size(dim=0)),Q_val).T #better nmerical stability than .inv
            Y_train_pred=Features[num_val_samples::,:]@Wo.T
            losses.append(np.array(F.mse_loss(Y_train_pred,Y[num_val_samples::,:])))
        ridge=ridges[np.argmin(np.array(losses))]
        return ridge

    def target2onehot(targets, n_classes):
        onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
        onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
        return onehot

    
    def predict(self, x, adapt=False):
        return self(x, adapt)

    def reset(self):
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data


    def configure_optimizers(self):
        
        # parameters = self.model.parameters()

        if self.optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(params=[self.prompt_pool],
                                        lr=self.lr,
                                        momentum=self.momentum,
                                        weight_decay=self.weight_decay)
        elif self.optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(params=[self.prompt_pool],
                                        lr=self.lr,
                                        weight_decay=self.weight_decay)
        else:
            raise NotImplementedError
        return optimizer



# if __name__ == '__main__':
#     a = torch.tensor([])