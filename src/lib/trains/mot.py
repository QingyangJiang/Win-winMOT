
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.decode import mot_decode
from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.utils import _sigmoid, _tranpose_and_gather_feat
from utils.post_process import ctdet_post_process

from .base_trainer import BaseTrainer


class MotLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MotLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt
        self.emb_dim = opt.reid_dim
        self.nID = opt.nID
        self.classifier = nn.Linear(self.emb_dim, self.nID)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        #self.TriLoss = TripletLoss()
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        #self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        #self.s_id = nn.Parameter(-1.05 * torch.ones(1))
        self.learningrate_np = 0.01
        self.rho = 0.5
    def forward(self, outputs, batch, p_tmp_j, phase):
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss = 0, 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                if opt.dense_wh:
                    mask_weight = batch['dense_wh_mask'].sum() + 1e-4
                    wh_loss += (
                                   self.crit_wh(output['wh'] * batch['dense_wh_mask'],
                                                batch['dense_wh'] * batch['dense_wh_mask']) /
                                   mask_weight) / opt.num_stacks
                else:
                    wh_loss += self.crit_reg(
                        output['wh'], batch['reg_mask'],
                        batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            if opt.id_weight > 0:
                id_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
                id_head = id_head[batch['reg_mask'] > 0].contiguous()
                id_head = self.emb_scale * F.normalize(id_head)
                id_target = batch['ids'][batch['reg_mask'] > 0]
                id_output = self.classifier(id_head).contiguous()
                id_loss += self.IDLoss(id_output, id_target)
                #id_loss += self.IDLoss(id_output, id_target) + self.TriLoss(id_head, id_target)

        #loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss + opt.id_weight * id_loss

        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss
        loss_set = {'hm':hm_loss,
                     'wh':wh_loss,
                     'off':off_loss,
                     'id':id_loss}

        def q(t,p_tmp_j):
            result = math.exp(math.log(p_tmp_j[t])+self.learningrate_np*loss_set[t])
            return result

        def f(lamda,p_tmp_j):
            sum1 = 0
            sum2 = 0
            sum3 = 0
            for t in ['hm','wh','off','id']:
                tmp_tmp = (1+lamda)**(-1)
                tmp = q(t,p_tmp_j)**tmp_tmp
                sum1 += math.log(q(t,p_tmp_j))*tmp
                sum2 += (1+lamda)*tmp
                sum3 += tmp
            result = sum1/sum2 - math.log(sum3) + math.log(len(loss_set)) - self.rho
            return result

        def compute_lamda(p_tmp_j,epsilon = 0.01, beta = 10):
            lamda_l, lamda_r = 0, 0
            if f(0,p_tmp_j) < 0 or f(0,p_tmp_j) == 0:   return 0
            while f(lamda_r,p_tmp_j) > 0 and f(lamda_r,p_tmp_j) == 0:
                lamda_l = lamda_r
                lamda_r = lamda_l + beta
            lamda_c = (lamda_l + lamda_r)/2
            while f(lamda_c,p_tmp_j) > epsilon or f(lamda_c,p_tmp_j) < -epsilon:
                lamda_c = (lamda_l + lamda_r) / 2
                if f(lamda_c,p_tmp_j) > 0:
                    lamda_l = lamda_c
                else:
                    lamda_r = lamda_c
            return lamda_c

        if p_tmp_j is {}:
            p_tmp_j={'hm':0.25,
                     'wh':0.25,
                     'off':0.25,
                     'id':0.25}
        elif phase == 'train':
            p_tmp_j = p_tmp_j
        else:
            lamda = compute_lamda(p_tmp_j)
            compute_p_tmp_j_son = {}
            compute_p_tmp_j_mor = 0
            for t in ['hm', 'wh', 'off', 'id']:
                son_tmp1 = (1 + lamda) ** (-1)
                son_tmp2 = math.log(p_tmp_j[t]) + self.learningrate_np * loss_set[t]
                compute_p_tmp_j_son[t] = math.exp(son_tmp1 * son_tmp2)
                compute_p_tmp_j_mor += compute_p_tmp_j_son[t]

            for t in ['hm', 'wh', 'off', 'id']:
                p_tmp_j[t] = compute_p_tmp_j_son[t] / compute_p_tmp_j_mor

        #loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss + (self.s_det + self.s_id)
        #loss *= 0.5
        loss = p_tmp_j['hm']*hm_loss + \
               p_tmp_j['wh']*wh_loss + \
               p_tmp_j['off']*off_loss + \
               p_tmp_j['id']*id_loss

        #print(loss, hm_loss, wh_loss, off_loss, id_loss)

        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': id_loss}
        return loss, loss_stats, p_tmp_j




class MotTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(MotTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss']
        loss = MotLoss(opt)
        return loss_states, loss

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = mot_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]