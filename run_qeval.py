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

    log_dir = os.path.join(root, args.quant_mode)
    log_dir = os.path.join(log_dir, task_dir)

    log_file = os.path.join(log_dir, log_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    return log_file


def arg_parse():
    parser = argparse.ArgumentParser(
        description='This repository contains the PyTorch implementation for the paper ZeroQ: A Novel Zero-Shot Quantization Framework.')
    # hyperparameters

    parser.add_argument('--bs', type=int, default=None, help='batch size')
    parser.add_argument('--arch', type=str, default='roberta_base',
                        choices=['roberta_base', 'roberta_large', ],
                        help='model architecture')
    parser.add_argument('--task', type=str,
                        choices=['RTE', 'SST-2', 'MNLI', 'QNLI',
                                 'CoLA', 'QQP', 'MRPC', 'STS-B',],
                        help='finetuning task')
    parser.add_argument('--quant-mode', type=str,
                        default='symmetric',
                        choices=['none', 'symmetric',],
                        help='quantization mode')
    parser.add_argument('--force-dequant', type=str, default='none', 
                        choices=['none', 'gelu', 'layernorm', 'softmax', 'nonlinear'],
                        help='force dequantize the specific layers')

    parser.add_argument('--model-dir', type=str, default='models',
                        help='model directory')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='folder name to store logs and checkpoints')
    parser.add_argument('--restore-file', type=str, default=None,
                        help='finetuning from the given checkpoint')
    parser.add_argument('--no-save', action='store_true')

    parser.add_argument('--softmax_type' , type=str)
    parser.add_argument('--exp_filename' , type=str)
    parser.add_argument('--div_filename' , type=str)
    ###
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
if args.restore_file :
    model_path = args.restore_file
else:
    model_path = args.model_dir  + '/roberta.large/model.pt' if is_large \
            else args.model_dir + '/roberta.base/model.pt'

valid_subset = 'valid' if task != 'MNLI' else 'valid,valid1'
print('valid_subset:',valid_subset)

###############################################################

subprocess_args = [
    'fairseq-validate', dataset,
    '--path', model_path,
    '--valid-subset', valid_subset,
    '--max-sentences', bs,
    '--task', 'sentence_prediction',
    '--criterion', 'sentence_prediction',
    '--num-classes', num_classes,
    '--log-file', log_file,
]

subprocess.call(subprocess_args)
