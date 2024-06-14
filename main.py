import argparse
import os

import wandb

import numpy as np
import torch
import random

from models.models import BaseCMNModel
from modules.dataloaders import R2DataLoader
from modules.loss import compute_loss, compute_adv_loss
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.tokenizers import Tokenizer
from modules.trainer import Trainer


def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/',
                        help='the path to the directory containing the data.')
    parser.add_argument('--image_dir_test', type=str, default='data/iu_xray/images/',
                        help='the path to the directory containing the data for test.')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json',
                        help='the path to the directory containing the data.')
    parser.add_argument('--img_label_path', type=str, default='data/iu_xray/auto_labeled/image_labels.pkl',
                        help='the path to pkl file with image labels.')
    parser.add_argument('--report_label_path', type=str, default='data/iu_xray/auto_labeled/labeled_reports.pkl',
                        help='the path to pkl file with report labels.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'],
                        help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=100, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=10, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=8, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=32, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')

    # for Cross-modal Memory
    parser.add_argument('--topk', type=int, default=32, help='the number of k.')
    parser.add_argument('--cmm_size', type=int, default=1024, help='the numebr of cmm size.')
    parser.add_argument('--cmm_dim', type=int, default=512, help='the dimension of cmm dimension.')
    parser.add_argument('--smax_temperature', type=float, default=1.0, help='the temperature of the softmax.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=15, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/mimic_cxr', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments.')
    parser.add_argument('--log_period', type=int, default=1000, help='the logging interval (in batches).')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period (in epochs).')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=1e-4, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_d', type=float, default=2e-4, help='the learning rate for the discriminator.')
    parser.add_argument('--lr_ed', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--adam_betas', type=tuple, default=(0.9, 0.98), help='the weight decay.')
    parser.add_argument('--adam_eps', type=float, default=1e-9, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')
    parser.add_argument('--noamopt_warmup', type=int, default=5000, help='.')
    parser.add_argument('--noamopt_factor', type=int, default=1, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50000, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.8, help='the gamma of the learning rate scheduler.')

    # unsupervised params
    parser.add_argument('--unsupervised', type=int, default=1, help='unsupervised (1) or supervised (0) training.')
    parser.add_argument('--use_glob_feat', type=int, default=1, help='use global features (1) or not (0) in the language model.')
    parser.add_argument('--lng_weight', type=float, default=1.0, help='the weight for language loss.')
    parser.add_argument('--cyc_weight', type=float, default=0.25, help='the weight for cycle loss.')
    parser.add_argument('--cyc_local', type=float, default=1.0, help='the weight for local cycle loss.')
    parser.add_argument('--cd_weight', type=float, default=1.0, help='the weight for cross-domain loss.')
    parser.add_argument('--adv_weight', type=float, default=1.0, help='the weight for adversarial loss.')
    parser.add_argument('--temp_contrastive', type=float, default=0.3, help='the temperature tau for contrastive loss.')
    parser.add_argument('--noise_std', type=float, default=0.0, help='the amount of noise to be added to text embeddings.')
    parser.add_argument('--kq_normalize', type=int, default=1, help='whether or not normalize keys and queries in memory')
    parser.add_argument('--decoder_dropout', type=float, default=0.9, help='the amount dropout in language decoder.')
    parser.add_argument('--iterative_bt', type=int, default=0, help='whether or not back-translate iteratively')
    parser.add_argument('--start_bt_epoch', type=int, default=1, help='in which epoch start training with back-translation')
    parser.add_argument('--map_glob_only', type=int, default=0, help='map only global feature or all')
    parser.add_argument('--freeze_cd', type=int, default=1, help='whether to freeze target weights for pseudo-text')
    parser.add_argument('--i2r_architecture', type=str, default='MHA', help='the architecture of the mapping module')

    # Others
    parser.add_argument('--seed', type=int, default=1003, help='.')
    parser.add_argument('--resume', type=str, help='existing checkpoint to resume / eval.')
    parser.add_argument('--reset_lr', type=int, default=0, help='whether to take lr from checkpoint (0) or arguments (1).')
    parser.add_argument('--eval_only', type=int, default=0, help='whether to run only evaluation.')
    parser.add_argument('--wandb', type=int, default=0, help='whether to use w&b or not')
    parser.add_argument('--wandb_project', type=str, default=None, help='w&b project name')
    parser.add_argument('--wandb_name', type=str, default=None, help='w&b run name')
    parser.add_argument('--debug', type=int, default=0, help='debug mode')

    args = parser.parse_args()
    return args


def main():
    # parse arguments
    args = parse_agrs()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    # mixed precision
    torch.backends.cuda.matmul.allow_tf32 = True

    # W & B
    if args.wandb:
        project_name = args.wandb_project if args.wandb_project is not None else "ReportGen_" + args.dataset_name
        if args.wandb_name is None:
            wandb_run = wandb.init(project=project_name)
        else:
            wandb_run = wandb.init(project=project_name, name=args.wandb_name)
            args.save_dir = os.path.join(args.save_dir, args.wandb_name)
        wandb_run.config.update(args)
    else:
        wandb_run = None

    # create tokenizer
    tokenizer = Tokenizer(args)

    # create data loader
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)
    train_dataloader_bt = R2DataLoader(args, tokenizer, split='train_bt', shuffle=False)

    # build model architecture
    model = BaseCMNModel(args, tokenizer)

    # get function handles of loss and metrics
    criterion = (compute_loss, compute_adv_loss)
    metrics = compute_scores

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # build trainer and start to train
    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                      test_dataloader, wandb_run, train_dataloader_bt)
    if args.eval_only == 1:
        trainer.eval()
    else:
        trainer.train()


if __name__ == '__main__':
    main()
