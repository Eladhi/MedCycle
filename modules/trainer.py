import logging
import os
import random
from abc import abstractmethod

import pandas as pd
import torch
import json
import numpy as np
from numpy import inf
import wandb
from matplotlib import pyplot as plt
from tqdm import tqdm

from .loss import get_sim_mat
from .utils import make_variable

use_amp = True  # mxp

class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, wandb):
        self.args = args

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        fh = logging.FileHandler(os.path.join(args.save_dir, 'log.log'), 'w+')
        self.logger.addHandler(fh)
        self.wandb = wandb

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        if type(optimizer) is list:
            self.optimizer = optimizer[0]
            self.optimizer_d = optimizer[1]
            self.lr_scheduler = lr_scheduler[0]
            self.lr_scheduler_d = lr_scheduler[1]
        else:
            self.optimizer = optimizer
            self.optimizer_d = None
            self.lr_scheduler = lr_scheduler

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.iterative_bt = None
        self.start_bt_epoch = None

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args)

    @abstractmethod
    def _eval_model(self, epoch):
        raise NotImplementedError

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    @abstractmethod
    def _back_translation_step(self, epoch):
        raise NotImplementedError

    def eval(self):
        result = self._eval_model()
        log = {'epoch': 'eval'}
        log.update(result)
        # print logged informations to the screen
        for key, value in log.items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result, test_reports = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log, test_reports, self.args.save_dir)
            self._print_best()

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                            self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

            if self.iterative_bt and (epoch >= self.start_bt_epoch):
                self._back_translation_step(epoch)
                # reset dataloader
                from modules.dataloaders import R2DataLoader
                from modules.tokenizers import Tokenizer
                tokenizer = Tokenizer(self.args)
                ann_path = self.args.ann_path
                self.args.ann_path = self.args.save_dir
                self.train_dataloader = R2DataLoader(self.args, tokenizer, split='train', shuffle=True)
                self.args.ann_path = ann_path
                self.logger.info('Finished resetting dataloader after back-translation.')

    def _record_best(self, log, test_reports, save_dir):
        with open(os.path.join(save_dir, 'reports_last.json'), 'w') as fout:
            json.dump(test_reports, fout)
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)
            with open(os.path.join(save_dir, 'reports_best.json'), 'w') as fout:
                json.dump(test_reports, fout)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        self.logger.info('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))

        self.logger.info('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        if self.optimizer_d is not None:
            state['optimizer_d'] = self.optimizer_d.state_dict()
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, args):
        resume_path = str(args.resume)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'optimizer_d' in checkpoint.keys():
            self.optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        if args.reset_lr:
            self.optimizer.param_groups[0]['lr'] = args.lr_ve
            self.optimizer.param_groups[1]['lr'] = args.lr_ed
            if self.optimizer_d is not None:
                self.optimizer_d.param_groups[0]['lr'] = args.lr_d

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader,
                 val_dataloader, test_dataloader, wandb=None, train_dataloader_bt=None):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler, wandb)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.train_dataloader_bt = train_dataloader_bt
        self.iterative_bt = args.iterative_bt
        self.start_bt_epoch = args.start_bt_epoch
        self.log_iter = 0
        self.lng_weight = args.lng_weight
        self.cyc_weight = args.cyc_weight
        self.cyc_local_weight = args.cyc_local
        self.cd_weight = args.cd_weight
        self.adv_weight = args.adv_weight
        self.temp_contrastive = args.temp_contrastive
        self.map_glob_only = args.map_glob_only
        self.freeze_cd = args.freeze_cd

    def _train_epoch(self, epoch):

        self.logger.info('[{}/{}] Start to train on the training set.'.format(epoch, self.epochs))
        train_loss = 0
        lng_loss_acc = 0
        cyc_loss_acc = 0
        cd_loss_acc = 0
        adv_loss_acc = 0
        d_acc_acc = 0
        net_batch_idx = 0

        if epoch >= self.start_bt_epoch:
            self.cd_weight = self.args.cd_weight
        else:
            self.cd_weight = 0

        self.model.train()
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)  # mxp
        for batch_idx, (images_id, images, reports_ids, reports_masks, pseudo_report, pseudo_mask) in enumerate(self.train_dataloader):

            images, reports_ids, reports_masks, pseudo_report, pseudo_mask = images.to(self.device), reports_ids.to(self.device), \
                                                 reports_masks.to(self.device), pseudo_report.to(self.device), pseudo_mask.to(self.device)

            ## optimize discriminator
            if self.adv_weight > 0:
                # get embeddings
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):  # mxp
                    output, enc_s_rec, enc_t_rec, enc_s, enc_t, enc_s2t, enc_t2s = self.model(images, reports_ids, mode='train')
                    if not self.map_glob_only:
                        d_model = enc_s.shape[-1]
                        enc_real = torch.cat((enc_s.view(-1, d_model), enc_t.view(-1, d_model)), 0)
                        enc_fake = torch.cat((enc_s2t.view(-1, d_model), enc_t2s.view(-1, d_model)), 0)
                    else:
                        enc_real = torch.cat((enc_s, enc_t), 0)
                        enc_fake = torch.cat((enc_s2t, enc_t2s), 0)
                    embeddings = torch.cat((enc_real, enc_fake), 0)
                    # run discriminator
                    pred_concat = self.model.discriminator(embeddings)
                    # make labels (0.1=fake, 0.9=real)
                    labels_r = make_variable(0.9 * torch.ones(len(enc_real)), requires_grad=False)
                    labels_f = make_variable(0.1 * torch.ones(len(enc_fake)), requires_grad=False)
                    label_concat = torch.cat((labels_r, labels_f), 0)
                    # compute loss
                    loss_adv = self.criterion[1](pred_concat[:, 0], label_concat)
                # update discriminator
                self.optimizer_d.zero_grad()
                scaler.scale(loss_adv).backward()
                scaler.unscale_(self.optimizer_d)
                scaler.step(self.optimizer_d)
                self.lr_scheduler_d.step()
                n_real = int(pred_concat.shape[0]/2)
                success_real = torch.sum((torch.sigmoid(pred_concat[:n_real]) > 0.5).long())
                success_fake = torch.sum((torch.sigmoid(pred_concat[-n_real:]) < 0.5).long())
                d_acc = (success_real + success_fake) / pred_concat.shape[0]
                d_acc_acc += d_acc.item()

            net_batch_idx += 1

            ## optimize network
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):  # mxp
                output, enc_s_rec, enc_t_rec, enc_s, enc_t, enc_s2t, enc_t2s = self.model(images, reports_ids, mode='train')

                if not self.map_glob_only:
                    d_model = enc_s.shape[-1]
                    enc_real = torch.cat((enc_s.view(-1, d_model), enc_t.view(-1, d_model)), 0)
                    enc_fake = torch.cat((enc_s2t.view(-1, d_model), enc_t2s.view(-1, d_model)), 0)
                    enc_s_rec_loc = enc_s_rec[:, 1:]
                    enc_t_rec_loc = enc_t_rec[:, 1:]
                    enc_s_loc = enc_s[:, 1:]
                    enc_t_loc = enc_t[:, 1:]
                    enc_s_rec_g = enc_s_rec[:, 0]
                    enc_t_rec_g = enc_t_rec[:, 0]
                    enc_s_g = enc_s[:, 0]
                    enc_t_g = enc_t[:, 0]
                else:
                    enc_real = torch.cat((enc_s, enc_t), 0)
                    enc_fake = torch.cat((enc_s2t, enc_t2s), 0)

                # language loss (auto-encode)
                loss_lng = self.criterion[0](output, reports_ids, reports_masks)

                # cycle loss
                cosine_only = False
                if cosine_only:
                    src_align = torch.nn.functional.cosine_similarity(enc_s_g, enc_s_rec_g)
                    tgt_align = torch.nn.functional.cosine_similarity(enc_t_g, enc_t_rec_g)
                    loss_cyc = - 1 * src_align.mean() - 1 * tgt_align.mean()
                else:
                    labels = torch.autograd.Variable(torch.LongTensor(range(enc_s_g.shape[0]))).to(enc_s_g.device)
                    src_g_align = get_sim_mat(enc_s_g, enc_s_rec_g) / self.temp_contrastive
                    tgt_g_align = get_sim_mat(enc_t_g, enc_t_rec_g) / self.temp_contrastive
                    loss_cyc_g = torch.nn.CrossEntropyLoss()(src_g_align, labels) + torch.nn.CrossEntropyLoss()(tgt_g_align, labels) \
                               + torch.nn.CrossEntropyLoss()(src_g_align.transpose(0, 1), labels) + torch.nn.CrossEntropyLoss()(tgt_g_align.transpose(0, 1), labels)
                    loss_cyc_l = torch.min(torch.cdist(enc_s_loc, enc_s_rec_loc), dim=1).values.mean() \
                                 + torch.min(torch.cdist(enc_t_loc, enc_t_rec_loc), dim=1).values.mean()
                    loss_cyc = loss_cyc_g + self.cyc_local_weight * loss_cyc_l

                # adversarial loss
                embeddings = torch.cat((enc_real, enc_fake), 0)
                pred_concat = self.model.discriminator(embeddings)
                # make labels (0=real, 1=fake) - now it's reversed
                labels_r = make_variable(0.0 * torch.ones(len(enc_real)), requires_grad=False)
                labels_f = make_variable(1.0 * torch.ones(len(enc_fake)), requires_grad=False)
                label_concat = torch.cat((labels_r, labels_f), 0)
                loss_adv = self.criterion[1](pred_concat[:, 0], label_concat)

            # back-translation
            if self.cd_weight > 0:
                unsupervised = self.model.encoder_decoder.model.unsupervised
                self.model.encoder_decoder.model.unsupervised = 0
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):  # mxp
                    output, enc_s_rec, enc_t_rec, enc_s, enc_t, enc_s2t, enc_t2s = self.model(images, pseudo_report, mode='train')
                    if not self.map_glob_only:
                        enc_s = enc_s[:, 0]
                        enc_t = enc_t[:, 0]
                        enc_s2t = enc_s2t[:, 0]
                        enc_t2s = enc_t2s[:, 0]
                    labels = torch.autograd.Variable(torch.LongTensor(range(enc_s.shape[0]))).to(enc_s.device)
                    src_g_align = get_sim_mat(enc_s, enc_t2s) / self.temp_contrastive
                    tgt_g_align = get_sim_mat(enc_t, enc_s2t) / self.temp_contrastive
                    loss_cd = torch.nn.CrossEntropyLoss()(src_g_align, labels) + torch.nn.CrossEntropyLoss()(tgt_g_align, labels) \
                              + torch.nn.CrossEntropyLoss()(src_g_align.transpose(0, 1), labels) + torch.nn.CrossEntropyLoss()(tgt_g_align.transpose(0, 1), labels)
                self.model.encoder_decoder.model.unsupervised = unsupervised
            else:
                loss_cd = torch.zeros(1).to(self.device)

            # pseudo reports will/not update the textual components
            if self.freeze_cd:
                loss_cd_tot = self.cd_weight * loss_cd
                loss = self.lng_weight * loss_lng + self.cyc_weight * loss_cyc + self.adv_weight * loss_adv
            else:
                loss = self.lng_weight * loss_lng + self.cyc_weight * loss_cyc + self.cd_weight * loss_cd + self.adv_weight * loss_adv

            train_loss += loss.item()
            lng_loss_acc += loss_lng.item()
            cyc_loss_acc += loss_cyc.item()
            cd_loss_acc += loss_cd.item()
            adv_loss_acc += loss_adv.item()

            self.optimizer.zero_grad()
            # pseudo reports will/not update the textual components
            if self.freeze_cd:
                for param in self.model.encoder_decoder.model.tgt_embed.parameters():
                    param.requires_grad = False
                for param in self.model.encoder_decoder.model.encoder_tgt.parameters():
                    param.requires_grad = False
                for param in self.model.encoder_decoder.model.decoder.parameters():
                    param.requires_grad = False
                scaler.scale(loss_cd_tot).backward()
                for param in self.model.encoder_decoder.model.tgt_embed.parameters():
                    param.requires_grad = True
                for param in self.model.encoder_decoder.model.encoder_tgt.parameters():
                    param.requires_grad = True
                for param in self.model.encoder_decoder.model.decoder.parameters():
                    param.requires_grad = True
                scaler.scale(loss).backward()
            else:
                scaler.scale(loss).backward()

            scaler.unscale_(self.optimizer)
            scaler.step(self.optimizer)
            old_scaler = scaler.get_scale()
            scaler.update()
            new_scaler = scaler.get_scale()
            if old_scaler < new_scaler:
                self.lr_scheduler.step()

            if batch_idx % self.args.log_period == 0:
                self.logger.info('[{}/{}] Step: {}/{}, Training Loss: {:.5f}.'
                                 .format(epoch, self.epochs, batch_idx, len(self.train_dataloader),
                                         train_loss / (net_batch_idx)))
                if self.wandb is not None:
                    self.wandb.log({"loss": train_loss / (net_batch_idx),
                                    "language_loss": lng_loss_acc / (net_batch_idx),
                                    "cyc_loss": cyc_loss_acc / (net_batch_idx),
                                    "cd_loss": cd_loss_acc / (net_batch_idx),
                                    "adv_loss": adv_loss_acc / (net_batch_idx),
                                    "discrim_acc": d_acc_acc / (batch_idx + 1),
                                    "lr_ve": self.lr_scheduler.get_last_lr()[0],
                                    "lr_ed": self.lr_scheduler.get_last_lr()[1]}, step=self.log_iter)
                    self.log_iter += 1

        log = {'train_loss': train_loss / len(self.train_dataloader)}

        self.logger.info('[{}/{}] Start to evaluate on the validation set.'.format(epoch, self.epochs))
        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks, _, _) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):  # mxp
                    output, _, others = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)

            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        self.logger.info('[{}/{}] Start to evaluate on the test set.'.format(epoch, self.epochs))
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            cyc_loss_acc = 0
            lng_loss_acc = 0
            for batch_idx, (images_id, images, reports_ids, reports_masks, _, _) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):  # mxp
                    output, _, others = self.model(images, mode='sample')
                enc_s2t = torch.stack([o[0] for o in others])  # at 'sample' mode others contain auxiliary information
                loss_cyc = torch.concat([o[0] for o in others]).mean()
                cyc_loss_acc += loss_cyc.item()
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})

            test_reports_list = []
            for ri, r in enumerate(test_gts):
                e = {}
                e['id'] = self.test_dataloader.dataset.examples[ri]['id']
                e['image_path'] = self.test_dataloader.dataset.examples[ri]['image_path']
                e['output'] = test_res[ri]
                e['gt'] = test_gts[ri]
                test_reports_list.append(e)

            log.update(**{'test_' + k: v for k, v in test_met.items()})
            if self.wandb is not None:
                self.wandb.log({'test_cyc_loss': cyc_loss_acc / (batch_idx + 1),
                                'test_lng_loss': lng_loss_acc / (batch_idx + 1)}, step=self.log_iter - 1)

        return log, test_reports_list


    def _eval_model(self):

        log = {}
        print('Start to evaluate on the test set.')
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            cyc_loss_acc = 0
            for batch_idx, (images_id, images, reports_ids, reports_masks, _, _) in enumerate(tqdm(self.test_dataloader)):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):  # mxp
                    output, _, others = self.model(images, mode='sample')
                loss_cyc = torch.concat([o[0] for o in others]).mean()
                cyc_loss_acc += loss_cyc.item()
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})

            log.update(**{'test_' + k: v for k, v in test_met.items()})

        return log


    def _back_translation_step(self, epoch):
        self.logger.info('[{}/{}] Back-translating all training images.'.format(epoch, self.epochs))
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            beam_size = self.model.args.beam_size
            self.model.args.beam_size = 1
            for batch_idx, (images_id, images, reports_ids, reports_masks, _, _) in enumerate(tqdm(self.train_dataloader_bt, smoothing=0.01)):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                if random.random() < 0.1:
                    with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):  # mxp
                        output, _, _ = self.model(images, mode='sample')
                    reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                else:
                    reports = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
            self.model.args.beam_size = beam_size

            if epoch <= self.start_bt_epoch:
                df = pd.read_csv(os.path.join(self.args.ann_path, 'train.csv'), keep_default_na=False)
                self.logger.info('Read csv from: ' + os.path.join(self.args.ann_path, 'train.csv'))
            else:
                df = pd.read_csv(os.path.join(self.args.save_dir, 'train.csv'), keep_default_na=False)
                self.logger.info('Read csv from: ' + os.path.join(self.args.save_dir, 'train.csv'))

            for ri in range(df.shape[0]):
                if len(test_res[ri]) > 0:
                    df.at[ri, 'I Initial Report'] = test_res[ri]
            df.to_csv(os.path.join(self.args.save_dir, 'train.csv'), index=False)
            self.logger.info('Saved csv to: ' + os.path.join(self.args.save_dir, 'train.csv'))

        return None
