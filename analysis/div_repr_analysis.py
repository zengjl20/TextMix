"""
Code from https://github.com/IBM/low-resource-text-classification-framework
"""
import os
import random
import logging

import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

sys.path.append("../../../")
sys.path.append("../../")
sys.path.append("../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.active_learning.main_al_transformer import test_transformer_model
from analysis.selected_data import load_json
from utilities.general import create_dir
from utilities.general_preprocessors import processors, output_modes
from models.deep.main_transformer import get_glue_dataset, get_glue_tensor_dataset, train_transformer

from sys_config import CACHE_DIR, glue_datasets, GLUE_DIR, DATA_DIR, CKPT_DIR, BERT_DIR, FIG_DIR, ANALYSIS_DIR

logger = logging.getLogger(__name__)

"""
Computes diversity (in feature space) and represenativeness as in https://www.aclweb.org/anthology/2020.emnlp-main.638.pdf
"""

class DiversityCalculator():
    def __init__(self, all_data):
        self.all_data = all_data

    def compute_batch_score(self, batch):
        distances = self.get_per_element_score(batch)
        return 1 / np.mean(distances)

    def get_per_element_score(self, batch):
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(batch)
        distances, indices = nbrs.kneighbors(self.all_data)
        return distances[:, 1]

class KnnOutlierCalculator():
    def __init__(self,all_data):
        self.all_data = all_data
        self.nbrs = NearestNeighbors(n_neighbors=11, algorithm='ball_tree').fit(self.all_data)

    def get_per_element_score(self, batch):
        distances, indices = self.nbrs.kneighbors(batch)
        return np.mean(distances, axis=-1)

    def compute_batch_score(self, batch):
        return np.mean(self.get_per_element_score(batch))


if __name__ == '__main__':
    # all_data = np.random.rand(15000, 50)
    # batch = all_data[1:50,:]
    # repr = KnnOutlierCalculator(all_data)
    # div = DiversityCalculator(all_data)
    #
    # repr_score = round(repr.compute_batch_score(batch),3)
    # div_score = round(div.compute_batch_score(batch),3)
    #
    # print('Diversity score: {}'.format(div_score))
    # print('Representativeness score: {}'.format(repr_score))
    # print('Done!')
    # # print(repr.compute_batch_score(batch))

    import argparse

    parser = argparse.ArgumentParser()
    ##########################################################################
    # Setup args
    ##########################################################################
    parser.add_argument("--local_rank",
                        type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    ##########################################################################
    # Model args
    ##########################################################################
    parser.add_argument("--model_type", default="bert", type=str, help="Pretrained model")
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str, help="Pretrained ckpt")
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_fast_tokenizer",
        default=True,
        type=bool,
        help="Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true",
        default=False,
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument("--tapt", default=None, type=str,
                        help="ckpt of tapt model")
    ##########################################################################
    # Training args
    ##########################################################################
    parser.add_argument("--do_train", default=True, type=bool, help="If true do train")
    parser.add_argument("--do_eval", default=True, type=bool, help="If true do evaluation")
    parser.add_argument("--overwrite_output_dir", default=True, type=bool, help="If true do evaluation")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=128, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=0, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("-seed", "--seed", required=False, type=int, help="seed")
    parser.add_argument("--indicator", required=False,
                        default=None,
                        type=str,
                        help="experiment indicator")
    parser.add_argument("-patience", "--patience", required=False, type=int, help="patience for early stopping (steps)")
    parser.add_argument("-test_uncertainty", "--test_uncertainty", required=False, type=bool, default=True,
                        help=" whether to evaluate uncertainty estimates for [vanilla, mc_3, mc_5, mc_10]")
    ##########################################################################
    # Data args
    ##########################################################################
    parser.add_argument("--dataset_name", default=None, required=True, type=str,
                        help="Dataset [mrpc, ag_news, qnli, sst-2, trec-6]")
    parser.add_argument("--task_name", default=None, type=str, help="Task [MRPC, AG_NEWS, QNLI, SST-2]")
    parser.add_argument("--max_seq_length", default=256, type=int, help="Max sequence length")
    parser.add_argument("--counterfactual", default=None, type=str, help="Max sequence length")
    # parser.add_argument("--ood_name", default=None, type=str, help="name of ood dataset for counterfactual data: [amazon, yelp, semeval]")
    ##########################################################################
    # Data augmentation args
    ##########################################################################
    parser.add_argument("--da", default=None, type=str, help="Apply DA method: [ssmba]")
    parser.add_argument("--da_set", default="lab", type=str, help="if lab augment labeled set else unlabeled")
    parser.add_argument("--num_per_augm", default=1, type=int, help="number of augmentations per example")
    parser.add_argument("--num_augm", default=100, type=int, help="number of examples to augment")
    parser.add_argument("--da_all", default=False, type=bool, help="if True augment the entire dataset")
    parser.add_argument("--uda", default=False, type=bool, help="if true consistency loss, else supervised learning")
    parser.add_argument("--uda_confidence_thresh", default=0.5, type=float, help="confidence threshold")
    parser.add_argument("--uda_softmax_temp", default=0.4, type=float, help="temperature to sharpen predictions")
    parser.add_argument("--uda_coeff", default=1, type=float, help="lambda value (weight) of KL loss")
    ##########################################################################
    # Server args
    ##########################################################################
    parser.add_argument("-g", "--gpu", required=False,
                        default='0', help="gpu on which this experiment runs")
    parser.add_argument("-server", "--server", required=False,
                        default='ford', help="server on which this experiment runs")

    args = parser.parse_args()

    # Setup
    if args.server is 'ford':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print("\nThis experiment runs on gpu {}...\n".format(args.gpu))
        # VIS['enabled'] = True
        args.n_gpu = 1
        args.device = torch.device('cuda:{}'.format(args.gpu))
    else:
        # Setup CUDA, GPU & distributed training
        if args.local_rank == -1 or args.no_cuda:
            args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            args.n_gpu = 0 if args.no_cuda else 1
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(args.local_rank)
            args.device = torch.device("cuda", args.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            args.n_gpu = 1

    print('device: {}'.format(args.device))

    # Setup args
    if args.seed == None:
        seed = random.randint(1, 9999)
        args.seed = seed
    args.seed = 964
    if args.task_name is None: args.task_name = args.dataset_name.upper()

    args.cache_dir = CACHE_DIR

    if args.dataset_name in glue_datasets:
        args.data_dir = os.path.join(GLUE_DIR, args.task_name)
    else:
        args.data_dir = os.path.join(DATA_DIR, args.task_name)
    args.overwrite_cache = bool(True)
    args.evaluate_during_training = True

    if args.tapt is not None:
        # orig_ft_model = args.model_name_or_path
        tapt_ckpt_path = os.path.join(CKPT_DIR, '{}_ft'.format(args.dataset_name), args.tapt)
        # args.model_name_or_path = tapt_ckpt_path
    # Output dir
    model_type = args.model_type if args.model_type != 'allenai/scibert' else 'bert'
    output_dir = os.path.join(BERT_DIR, '{}_{}'.format(args.dataset_name, args.model_type))
    # if args.model_type == 'bert':
    #     ckpt_dir = BERT_DIR
    # elif args.model_type == 'distilbert':
    #     ckpt_dir = BERT_DIR = os.path.join(CKPT_DIR, 'distilbert')
    # elif args.model_type == 'roberta':
    #     ckpt_dir = BERT_DIR = os.path.join(CKPT_DIR, 'roberta')
    # else:
    #     ckpt_dir = BERT_DIR
    # # args.output_dir = os.path.join(BERT_DIR, '{}_{}'.format(args.dataset_name, args.model_type))
    # args.output_dir = os.path.join(ckpt_dir, '{}_{}'.format(args.dataset_name, args.model_type))
    # if args.model_type == 'allenai/scibert': args.output_dir = os.path.join(ckpt_dir,
    #                                                                         '{}_{}'.format(args.dataset_name, 'bert'))
    args.output_dir = os.path.join(output_dir, 'all_{}'.format(args.seed))

    if args.indicator is not None: args.output_dir += '-{}'.format(args.indicator)
    if args.patience is not None: args.output_dir += '-early{}'.format(int(args.num_train_epochs))
    if args.tapt is not None: args.output_dir += '-{}'.format(args.tapt)
    if args.da is not None:
        if args.da_all:
            args.output_dir += '-{}-{}-{}-{}'.format(args.da, args.num_per_augm, 'all', args.da_set)
        else:
            args.output_dir += '-{}-{}-{}-{}'.format(args.da, args.num_per_augm, args.num_augm, args.da_set)
    if args.uda: args.output_dir += '-uda'
    args.current_output_dir = args.output_dir
    create_dir(args.output_dir)

    if args.uda: args.da_model = "cons"

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        args.device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=args.use_fast_tokenizer,
    )

    #########################################
    # Check if experiment already done
    #########################################
    path = os.path.join(FIG_DIR, '{}_{}_100%'.format(args.task_name,
                                                     # args.model_type
                                                     model_type
                                                     ))
    create_dir(path)

    if args.patience is not None:
        name = 'det_{}_lr_{}_bs_{}_early_epochs_{}_{}'.format(args.seed, args.learning_rate,
                                                              args.per_gpu_train_batch_size,
                                                              args.num_train_epochs,
                                                              args.indicator)
    else:
        print(args.indicator)
        name = 'det_{}_lr_{}_bs_{}_epochs_{}_{}'.format(args.seed, args.learning_rate,
                                                        args.per_gpu_train_batch_size,
                                                        args.num_train_epochs,
                                                        args.indicator)

    if args.tapt is not None: name += '_ft_{}'.format(args.tapt)
    # if args.da is not None: name += '_{}_{}_{}'.format(args.da, args.da_num, args.da_perc)
    if args.da is not None:
        if args.da_all:
            name += '_{}_{}_{}_{}'.format(args.da, args.num_per_augm, 'all', args.da_set)
        else:
            name += '_{}_{}_{}_{}'.format(args.da, args.num_per_augm, args.num_augm, args.da_set)
    if args.uda: name += '_uda'

    print(name)

    dirname = os.path.join(path, name)
    create_dir(dirname)

    # assert not os.path.isfile(os.path.join(dirname, 'results.json')), "Experiment done already! {}".format(dirname)

    # Load (raw) dataset
    X_train, y_train = get_glue_dataset(args, args.data_dir, args.task_name, args.model_type, evaluate=False)
    X_val, y_val = get_glue_dataset(args, args.data_dir, args.task_name, args.model_type, evaluate=True)
    X_test, y_test = get_glue_dataset(args, args.data_dir, args.task_name, args.model_type, test=True)

    X_orig = X_train  # original train set
    y_orig = y_train  # original labels
    X_inds = list(np.arange(len(X_orig)))  # indices to original train set
    X_unlab_inds = []  # indices of ulabeled set to original train set

    args.binary = True if len(set(y_train)) == 2 else False
    args.num_classes = len(set(y_train))

    args.undersampling = False
    if args.indicator is not None:
        # Undersample training dataset (stratified sampling)
        if "sample_" in args.indicator:
            args.undersampling = True
            num_to_sample = int(args.indicator.split("_")[1])
            X_train_orig_after_sampling_inds, X_train_orig_remaining_inds, _, _ = train_test_split(
                X_inds,
                y_orig,
                train_size=num_to_sample,
                random_state=args.seed,
                stratify=y_train)
            X_inds = X_train_orig_after_sampling_inds  # indices of train set to original train set
            # X_train = list(np.array(X_train)[X_inds])   # train set
            # y_train = list(np.array(y_train)[X_inds])   # labels
            # Treat the rest of training data as unlabeled data
            X_unlab_inds = X_train_orig_remaining_inds  # indices of ulabeled set to original train set

    assert len(X_unlab_inds) + len(X_inds) == len(X_orig)
    assert bool(not (set(X_unlab_inds) & set(X_inds)))
    assert max(X_inds) < len(X_orig)
    if X_unlab_inds != []:
        assert max(X_unlab_inds) < len(X_orig)

    #######################
    # Datasets
    #######################
    args.output_dir += '-first-it'
    args.current_output_dir = args.output_dir
    create_dir(args.output_dir)
    if len(os.listdir(args.output_dir)) != 0:
        model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
        model.to(args.device)
    else:
        # load selected indices
        filepath = os.path.join(FIG_DIR, '{}_{}_{}'.format(args.dataset_name, args.model_type, 'random'), 'prob_{}_{}'.format(args.seed, '25_config_early20'))
        if os.path.exists(filepath):

            sel_ids_filename = os.path.join(filepath, 'selected_ids_per_iteration.json')

            sel_ids = load_json(sel_ids_filename)
            X_train_init_inds = sel_ids['0']

        init_train_dataset = get_glue_tensor_dataset(X_train_init_inds, args, args.task_name, tokenizer, train=True)
        minibatch = int(len(X_inds) / (args.per_gpu_train_batch_size * max(1, args.n_gpu)))
        args.logging_steps = min(int(minibatch / 5), 500)
        if args.logging_steps < 1:
            args.logging_steps = 1
        if args.tapt is not None:
            model = AutoModelForSequenceClassification.from_pretrained(tapt_ckpt_path, config=config)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir,
            )

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        model.to(args.device)

        logger.info("Training/evaluation parameters %s", args)
        eval_dataset = get_glue_tensor_dataset(None, args, args.task_name, tokenizer, evaluate=True)
        model, tr_loss, val_loss, val_results = train_transformer(args, init_train_dataset, eval_dataset, model, tokenizer)

    #######################################
    # Load actively acquired datasets
    #######################################
    args.mc_samples = None
    sel_data_df = pd.DataFrame(columns=['dataset', 'acquisition', 'diversity_score', 'repr_score'])
    datasets = [args.dataset_name]
    acq_funs = ['random', 'FTbertKM', 'entropy', 'alps', 'badge', 'adv_train']
    # acq_funs = ['badge','adv_train']
    # seeds = [964, 131, 821, 71, 12]
    seeds=[964]
    model_name = 'bert'
    indicator = '25_config_early20'
    al_dataset = {}
    for dataset in datasets:
        # file.write("***{}***\n".format(dataset))
        for acq_fun in acq_funs:
            seed_data_df = pd.DataFrame(
                columns=['dataset', 'acquisition', 'seed', 'diversity_score', 'repr_score'])
            for seed in seeds:
                if acq_fun == 'entropy':
                    _i = indicator + '_vanilla'
                elif acq_fun == 'adv_train':
                    _i = indicator + '_cls'
                else:
                    _i = indicator

                filepath = os.path.join(FIG_DIR, '{}_{}_{}'.format(dataset, model_name, acq_fun),
                                        'prob_{}_{}'.format(seed, _i))
                if os.path.exists(filepath):
                    results_filename = os.path.join(filepath, 'results_of_iteration.json')
                    sel_ids_filename = os.path.join(filepath, 'selected_ids_per_iteration.json')

                    # Find actively acquired datasets
                    results = load_json(results_filename)
                    sel_ids = load_json(sel_ids_filename)
                    al_dataset[acq_fun] = sel_ids

                    # Compute diversity
                    data_sampled_inds_1 = sel_ids['0'] + sel_ids['1']
                    if data_sampled_inds_1 is not None:
                        data_pool = [x for x in X_inds if x not in data_sampled_inds_1]
                        all_data_dataset = get_glue_tensor_dataset(X_inds, args, args.task_name, tokenizer, train=True)
                        _, _, _results = test_transformer_model(args, dataset=all_data_dataset, model=model, return_cls=True)
                        bert_representations = _results["bert_cls"]

                        all_data = bert_representations.cpu().numpy()
                        # data_sampled_inds_1 = [0,5,7,2,47,76,34,6,21,63]
                        batch = bert_representations[data_sampled_inds_1].cpu().numpy()

                        # Representativeness
                        repr = KnnOutlierCalculator(all_data)
                        repr_score = round(repr.compute_batch_score(batch),3)

                        # Diversity
                        div = DiversityCalculator(all_data)
                        div_score = round(div.compute_batch_score(batch),3)

                        stats_dict = {'dataset': dataset,
                                      'acquisition': acq_fun,
                                      'seed': seed,
                                      'diversity_score': round(div_score, 3),
                                      'repr_score': round(repr_score, 3)}
                        print(stats_dict)
                        seed_data_df = seed_data_df.append(stats_dict, ignore_index=True)

            div = seed_data_df['diversity_score'].mean()
            unc = seed_data_df['repr_score'].mean()
            stats_dict = {'dataset': dataset, 'acquisition': acq_fun,
                          'diversity_score': round(div, 3),
                          'repr_score': round(unc, 3)}
            sel_data_df = sel_data_df.append(stats_dict, ignore_index=True)
            sel_data_df.to_csv(os.path.join(ANALYSIS_DIR, '{}_repr_div_stats.csv'.format(args.dataset_name)),
                               index=False)
    print()