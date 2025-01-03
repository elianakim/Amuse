'''
train.py

'''

import sys
import argparse
import random
import json
import gzip
from collections import defaultdict

import pickle
import numpy as np
import time
import os
from copy import deepcopy
from datetime import datetime
from tqdm import tqdm
import warnings

from openai import OpenAI
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# from utils import tr
import conf
from utils import (
    example_to_events, 
    transpose_chord_events, 
    text_to_event, 
    read_api_keys, 
    generate_llm_chords
)

warnings.filterwarnings("ignore")

def get_path():
    path = 'log/'

    # information about used data type
    path += conf.args.dataset + '/'

    # information about used model type
    path += conf.args.model + '/'

    path += conf.args.log_prefix + '/'

    checkpoint_path = path + 'cp/'
    log_path = path
    result_path = path + '/'

    print('Path:{}'.format(path))
    return result_path, checkpoint_path, log_path

def train():
    
    device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
    print(device) 

    # make dir if doesn't exist
    result_path, checkpoint_path, log_path = get_path()
    for path in [result_path, checkpoint_path, log_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    # initialize TensorBoard writer.
    # run on terminal: tensorboard --logdir={logdir} --host 0.0.0.0 --port {port}
    writer = SummaryWriter(log_dir=log_path)
    writer.add_text('conf', str(conf.args))

    # Hyperparameters
    if 'hooktheory' in conf.args.dataset:
        opt = conf.HooktheoryOpt
    elif 'llmchords' in conf.args.dataset:
        opt = conf.LLMChordsOpt
    else:
        raise NotImplementedError

    # override hyperparameters, if specified
    if conf.args.model_embed_dim:
        opt['model_embed_dim'] = conf.args.model_embed_dim
    if conf.args.model_hidden_dim:
        opt['model_hidden_dim'] = conf.args.model_hidden_dim
    if conf.args.model_num_layers:
        opt['model_num_layers'] = conf.args.model_num_layers
    if conf.args.bidirectional:
        opt['bidirectional'] = True

    if conf.args.summarize_frequency:
        opt['summarize_frequency'] = conf.args.summarize_frequency
    if conf.args.eval_frequency:
        opt['eval_frequency'] = conf.args.eval_frequency
    if conf.args.max_num_steps:
        opt['max_num_steps'] = conf.args.max_num_steps

    if conf.args.lr:
        opt['lr'] = conf.args.lr
    if conf.args.weight_decay:
        opt['weight_decay'] = conf.args.weight_decay
    if conf.args.batch_size:
        opt['batch_size'] = conf.args.batch_size   
    if conf.args.dropout:
        opt['dropout'] = conf.args.dropout

    conf.args.opt = opt 

    # Load Hooktheory data & create tokenizer
    hooktheory_events = None
    tokenizer = None
    if conf.args.dataset in ['hooktheory', 'llmchords']:
        from tokenizer import HooktheoryChordTokenizer

        with gzip.open('./dataset/Hooktheory/Hooktheory.json.gz', 'rb') as f:
            raw_data = json.load(f)
        print(f'Loaded {len(raw_data)} examples')
        assert len(raw_data) == 26175

        hooktheory_events = defaultdict(list)
        hooktheory_events['TRAIN'] = []
        hooktheory_events['TEST'] = []
        hooktheory_events['VALID'] = []

        # preprocess
        for _, example in tqdm(raw_data.items()):
            # filter out examples without 'HARMONY' tag
            if 'HARMONY' not in example['tags']:
                continue
        
            for event in example_to_events(example):
                assert event[0][0] == 'key'
                
                if len(event[1:]) < 4: 
                    continue
                if event[0][1] == 0: # chords are already in C 
                    hooktheory_events[example['split']].append(event[1:])

                else: # chords are in different key; transpose to C
                    hooktheory_events[example['split']].append(transpose_chord_events(event, 0)[1:])

        print('Extracted Hooktheory events: ')
        for split, examples in hooktheory_events.items():
            print(split, len(examples))

        assert sum([len(examples) for examples in hooktheory_events.values()]) == 25601

        tokenizer = HooktheoryChordTokenizer(hooktheory_events['TRAIN'])
    else:
        raise NotImplementedError
    
    # Create dataloader
    events = None

    if conf.args.dataset == 'hooktheory':
        events = hooktheory_events
    elif conf.args.dataset == 'llmchords':
        from dataset import HooktheoryDataset
        # load llmchords dataset instead of Hooktheory data
        events = defaultdict(list)
        events['TRAIN'] = []
        events['VALID'] = []
        events['TEST'] = []

        # if open fails (does not exists), generate and save to conf.args.llmchords_path
        llm_chords = None
        try:
            f = open(conf.args.llmchords_path, 'r')
            llm_chords = f.readlines()
        except:
            # generate
            print("LLM-generated chords not found. Creating...")
            api_keys = read_api_keys("./assets/")
            client = OpenAI(api_key=api_keys['openai'])
            llm_chords = generate_llm_chords(conf.args.llmchords_path, client)
        
        # process llmchords data
        for i, line in enumerate(llm_chords):
            try:
                chord_texts = line.strip().split(' ')
                chord_texts = [c.strip() for c in chord_texts]

                if len(chord_texts) < 4:
                    continue
                # error handling: if "min" in chord, convert to "m"
                chord_texts = [c.replace('min', 'm') for c in chord_texts]
                # convert text to Hooktheory events
                chord_events = [text_to_event(c) for c in chord_texts]
                if i < int(0.9 * len(llm_chords)):
                    events['TRAIN'].append(chord_events)
                else:
                    events['VALID'].append(chord_events)
            except Exception as e:
                continue  
        
        print('Extracted LLM Chords: ')
        for split, examples in events.items():
            print(split, len(examples))

    else:
        raise NotImplementedError

    train_loader = None
    eval_loader = None
    if conf.args.dataset in ['hooktheory', 'llmchords']:
        from dataset import HooktheoryDataset

        train_dataset = HooktheoryDataset(events['TRAIN'], conf.args.opt['seq_len'], tokenizer, train=True)
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=conf.args.opt['batch_size'], sampler=train_sampler, drop_last=True
        )

        eval_dataset = HooktheoryDataset(events['VALID'], conf.args.opt['seq_len'], tokenizer, train=False)
        eval_sampler = torch.utils.data.SequentialSampler(eval_dataset)
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=conf.args.opt['batch_size'], sampler=eval_sampler, drop_last=False
        )
    else:
        raise NotImplementedError

    # Load model
    if conf.args.model == 'lstm':
        from model.lstm import LSTM
        model = LSTM(
            vocab_size=len(tokenizer),
            embed_dim=conf.args.opt['model_embed_dim'],
            hidden_dim=conf.args.opt['model_hidden_dim'],
            num_layers=conf.args.opt['model_num_layers'],
            dropout=conf.args.opt['dropout'],
            bidirectional=conf.args.opt['bidirectional']
        )
    else:
        raise NotImplementedError
    
    since = time.time()
    
    with open(log_path + 'conf.json', 'w') as f:
        f.write(json.dumps(conf.args.__dict__, indent=4))
    with open(log_path + 'tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    print(str(log_path))

    model.train()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=conf.args.opt['lr'], weight_decay=conf.args.opt['weight_decay'])

    print("="*80)
    step = 0
    best_eval_nll = float("inf")

    while True:
        for inputs, targets in train_loader:
            # eval loop
            if step % conf.args.eval_frequency == 0:
                model.eval()
                with torch.no_grad():
                    total_nll = 0.0
                    total_targets = 0
                    for eval_inputs, eval_targets in eval_loader:
                        eval_inputs = eval_inputs.to(device)
                        eval_targets = eval_targets.to(device)
                        eval_outputs, _ = model(eval_inputs)

                        nll = F.cross_entropy(
                            eval_outputs.view(-1, len(tokenizer)),
                            eval_targets.view(-1),
                            ignore_index=-1,
                            reduction='none'
                        )
                        total_nll += nll.sum().item()
                        total_targets += (targets >= 0).long().sum().item()
                    eval_nll = total_nll / total_targets
                    if eval_nll < best_eval_nll:
                        print(f'Step: {step}\tEval nll: {eval_nll}')
                        if not conf.args.remove_cp:
                            torch.save(model.state_dict(), checkpoint_path + f'model-{step}.pt')
                        best_eval_nll = eval_nll
                writer.add_scalar('eval/nll', eval_nll, step)
                model.train()
            
            # train loop
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            nll = F.cross_entropy(
                outputs.view(-1, len(tokenizer)),
                targets.view(-1),
                ignore_index=-1)
            nll.backward()
            optimizer.step()

            # summarize training
            if step % conf.args.summarize_frequency == 0:
                writer.add_scalar('train/nll', nll.item(), step)
            
            # increment step counter
            step += 1
        
        if conf.args.max_num_steps is not None and step >= conf.args.max_num_steps:
            print(f'Saving the last model')
            torch.save(model.state_dict(), checkpoint_path + f'model-last.pt')
            break

    time_elapsed = time.time() - since
    print('Completion time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    writer.close()
    

def parse_arguments():
    """Command line parse."""

    # Note that 'type=bool' args should be False in default. Any string argument is recognized as "True". Do not give "--bool_arg 0"

    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument('--dataset', type=str, default='hooktheory',
                        help='Dataset to be used, in [hooktheory, llmchords]')
    parser.add_argument('--model', type=str, default='lstm',
                        help='Model to pretrain/finetune, in [lstm]')
    parser.add_argument('--gpu_idx', type=int, default=0, help='which gpu to use')

    # Model specific
    parser.add_argument('--model_embed_dim', type=int, default=None,
                        help='embed dim to overwrite conf.py')
    parser.add_argument('--model_hidden_dim', type=int, default=None,
                        help='hidden dim to overwrite conf.py')
    parser.add_argument('--model_num_layers', type=int, default=None,
                        help='num layers to overwrite conf.py')
    parser.add_argument('--bidirectional', action='store_true',
                        help='Bidirectional LSTM?')

    # Train specific    
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--summarize_frequency', type=int, default=32,
                        help='summarization frequency')
    parser.add_argument('--eval_frequency', type=int, default=128,
                        help='evaluation frequency')
    parser.add_argument('--max_num_steps', type=int, default=100000,
                        help='maximum number of steps to train')
    
    parser.add_argument('--lr', type=float, default=None,
                        help='learning rate to overwrite conf.py')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='weight_decay to overwrite conf.py')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='batch size to overwrite conf.py')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout to overwrite conf.py')


    # Logging & data
    parser.add_argument('--log_prefix', type=str, default='test',
                        help='Prefix of log file path')
    parser.add_argument('--remove_cp', action='store_true',
                        help='Remove checkpoints except the last one')
    parser.add_argument('--llmchords_path', type=str, default='./dataset/llmchords/chords.txt',
                        help='Path to llmchords data file. If not exists, download and save')
    parser.add_argument('--llmchords_num', type=int, default=25000,
                        help='Number of chord progressions to generate. API call is made for every 30 progressions.')

    return parser.parse_args()


def set_seed():
    torch.manual_seed(conf.args.seed)
    np.random.seed(conf.args.seed)
    random.seed(conf.args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    print('Command:', end='\t')
    print(" ".join(sys.argv))
    conf.args = parse_arguments()
    print(conf.args)
    set_seed()
    train()