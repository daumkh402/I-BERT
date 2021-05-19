import os
import subprocess
import argparse
from time import gmtime, strftime

def make_dir(args, is_large):
    root = args.output_dir
    no_save = args.no_save
    task_dir = args.task + '-' + ('large' if is_large else 'base')

    time = strftime("%m%d-%H%M%S", gmtime())
    log_name = '%s.log' % time

    log_dir = os.path.join(root, 'none')
    log_dir = os.path.join(log_dir, task_dir)

    log_file = os.path.join(log_dir, log_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    return log_file


def arg_parse():
    parser = argparse.ArgumentParser(
        description='This repository contains the PyTorch implementation for the paper ZeroQ: A Novel Zero-Shot Quantization Framework.')

    parser.add_argument('--bs', type=int, default=None, help='batch size')

    parser.add_argument('--arch', type=str, default='roberta_base',
                        choices=['roberta_base', 'roberta_large', ],
                        help='model architecture')

    parser.add_argument('--task', type=str, 
                        choices=['RTE', 'SST-2', 'MNLI', 'QNLI',
                                 'CoLA', 'QQP', 'MRPC', 'STS-B',],               
                        help='finetuning task')

    parser.add_argument('--path', type=str, default=None, required=True, help='model directory')
                        
    parser.add_argument('--output-dir', type=str, default='outputs', help='folder name to store logs and checkpoints')
                        
    parser.add_argument('--restore-file', type=str, default=None, help='finetuning from the given checkpoint')
                        
    parser.add_argument('--no-save', action='store_true')
    
    parser.add_argument('--exp_filename', type=str, default = 'None')
    parser.add_argument('--div_filename', type=str, default = 'None')
    parser.add_argument('--gelu_filename', type=str, default = 'None')
    parser.add_argument('--layernorm_filename', type=str, default = 'None')
    parser.add_argument('--softmax_type', type=str, default = 'None')
    parser.add_argument('--gelu_type', type=str, default = 'None')
    parser.add_argument('--layernorm_type', type=str, default = 'None')

    args = parser.parse_args()
    return args


args = arg_parse()
task = args.task
######################## Task specs ##########################

task_specs = {
    'RTE' : {
        'dataset': 'RTE-bin',
        'num_classes': '2',
    },

    'SST-2' : {
        'dataset': 'SST-2-bin',
        'num_classes': '2',
    },

    'MNLI' : {
        'dataset': 'MNLI-bin',
        'num_classes': '3',
    },

    'QNLI' : {
        'dataset': 'QNLI-bin',
        'num_classes': '2',
    },

    'CoLA' : {
        'dataset': 'CoLA-bin',
        'num_classes': '2',
    },

    'QQP' : {
        'dataset': 'QQP-bin',
        'num_classes': '2',
    },

    'MRPC' : {
        'dataset': 'MRPC-bin',
        'num_classes': '2',
    },

    'STS-B' : {
        'dataset': 'STS-B-bin',
        'num_classes': '1',
    },
}


is_large = 'large' in args.arch
spec = task_specs[task]
dataset = '%s-bin' % task
num_classes = spec['num_classes']
bs = str(args.bs) 
log_file = make_dir(args, is_large)
valid_subset = 'valid' if task != 'MNLI' else 'valid,valid1'

print('valid_subset:',valid_subset)
    
#######
if args.task in ["SST-2", "RTE", "QNLI", "MNLI"]:
    best_metric = 'accuracy'
if args.task in ["MRPC", "QQP"]:
    best_metric = "f1"
if args.task == "CoLA":
    best_metric = 'mcc'
if args.task == "STS-B":
    best_metric = 'loss'
#######

###############################################################

subprocess_args = [
    'fairseq-validate', dataset,
    '--task', 'sentence_prediction',
    '--path', args.path,
    '--valid-subset', valid_subset,
    '--max-sentences', str(bs),
    '--log-file', log_file,
    '--exp_filename', args.exp_filename,
    '--div_filename', args.div_filename,
    '--gelu_filename', args.gelu_filename,
    '--layernorm_filename', args.layernorm_filename,
    '--softmax_type', args.softmax_type,
    '--gelu_type', args.gelu_type,
    '--layernorm_type', args.layernorm_type,
]   

subprocess.call(subprocess_args)
