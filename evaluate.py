'''
Evaluate rejection-sampling method
'''

import sys
import os
import gzip
import argparse
import random
import pathlib
import json
import pickle
import warnings
import itertools
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from openai import OpenAI
from scipy.spatial.distance import jensenshannon

import conf
from utils import (
    read_api_keys,
    generate_llm_chords,
    text_to_event,
    example_to_events,
    transpose_chord_events,
)

warnings.filterwarnings("ignore")

def load_models(model_type, model_path, step, device):
    # load tokenizer
    tokenizer = None
    with open(pathlib.Path(model_path, 'tokenizer.pkl'), "rb") as f:
        tokenizer = pickle.load(f)
    
    # load model
    model = None
    if model_type == 'lstm':
        from model.lstm import LSTM
        with open(pathlib.Path(model_path, 'conf.json'), 'r') as f:
            conf = json.load(f)
            print(conf['opt'])
        model = LSTM(
            vocab_size=len(tokenizer),
            embed_dim=conf['opt']['model_embed_dim'],
            hidden_dim=conf['opt']['model_hidden_dim'],
            num_layers=conf['opt']['model_num_layers'],
            dropout=conf['opt']['dropout'],
            bidirectional=conf['opt']['bidirectional'],
        )
        model.load_state_dict(torch.load(pathlib.Path(model_path, 'cp/model-{:s}.pt'.format(step))))
        model.eval()
        model.to(device)
    else:
        raise NotImplementedError

    return model, tokenizer, conf['opt']

def truncate_to_nucleus(logits, p=0.9):
    logits_sorted = torch.sort(logits, descending=True, dim=-1)
    probs_sorted = F.softmax(logits_sorted.values, dim=-1)
    probs_cumulative = torch.cumsum(probs_sorted, dim=-1)
    probs_cumulative_safe = torch.cat([
        torch.zeros_like(probs_cumulative[..., :1]),
        probs_cumulative[..., :-1]],
        dim=-1
    )
    k = (probs_cumulative_safe < p).long().sum(dim=-1)
    vocab_size = logits.shape[-1]
    koh = F.one_hot(k - 1, vocab_size).float()
    thresholds = (logits_sorted.values * koh).sum(-1)
    partition = (logits >= thresholds.unsqueeze(-1)).type(torch.uint8)
    truncated = torch.where(
        partition,
        logits,
        float('-inf')
    )
    return truncated

def score_chord_progression(prog_events, model, tokenizer, device):
    prog_tokens = tokenizer.events_to_tokens(prog_events)[1:-1] # exclude <SOS> and <EOS>
    with torch.no_grad():
        h_i = model.init_hidden(1, device=device)
        input_seq = [tokenizer.sos]
        total_log_prob = 0.0

        for token in prog_tokens:
            x = torch.tensor([input_seq], dtype=torch.int64, device=device)
            logits, h_i = model(x, h_i)
            logits = logits[0, -1] / conf.args.rej_T
            prob = F.softmax(logits, dim=-1)[token].item()
            total_log_prob += np.log(prob)
            input_seq.append(token)
        
        return total_log_prob

def find_smallest_M(p_x, q_x, p_x_tokenizer, q_x_tokenizer, llm_events, device):
    '''
    Find the smallest possible M such that P(x) <= M * Q(x) for all x
    (i.e., P(x) / Q(x) <= M for all x)
    P(x): score generated by prior model
    Q(x): score generated by llm model
    '''
    p_q_ratios = []
    for prog_events in tqdm(llm_events, total=len(llm_events)):
        p_x_score = score_chord_progression(prog_events, p_x, p_x_tokenizer, device)
        q_x_score = score_chord_progression(prog_events, q_x, q_x_tokenizer, device)
        p_q_ratios.append(np.exp(p_x_score - q_x_score))
    
    M = np.percentile(p_q_ratios, 95)

    return p_q_ratios, M

def compute_jsd(data_dict, mode='unigram'):
    '''
    Calculate Jensen-Shannon Divergence (JSD) between data_dict[0] and others.
    data_dict: A dictionary where:
        - data_dict[0] contains the Hooktheory data (ground truth).
        - data_dict[1:] contains other chord datasets to compare with.
    mode: 'unigram' or 'bigram' to specify the type of calculation.
    return: A dictionary of JSD values for each dataset compared to data_dict[0].
    '''

    def generate_bigrams(tokens):
        return tuple(zip(tokens[:-1], tokens[1:]))

    ngram_freqs = {}

    if mode == 'unigram':
        for tag, data in data_dict.items():
            flattened = list(itertools.chain(*data))
            ngram_freqs[tag] = pd.Series(flattened).value_counts(normalize=True).reset_index()
    elif mode == 'bigram':
        for tag, data in data_dict.items():
            flattened = list(itertools.chain(*[generate_bigrams(tokens) for tokens in data]))
            ngram_freqs[tag] = pd.Series(flattened).value_counts(normalize=True).reset_index()
    else:
        raise ValueError("Mode must be either 'unigram' or 'bigram'")

    # rename columns and merge counts
    for tag in ngram_freqs:
        ngram_freqs[tag] = ngram_freqs[tag].rename(
            columns={'index': 'N-gram Token', ngram_freqs[tag].columns[1]: tag})

    merged_freqs = None
    for tag in list(data_dict.keys()):
        if merged_freqs is None:
            merged_freqs = ngram_freqs[tag]
        else:
            merged_freqs = pd.merge(merged_freqs, ngram_freqs[tag], 
                                    on='N-gram Token', how='outer').fillna(0)

    # Drop rows where all values (excluding the 'N-gram Token' column) are 0
    merged_freqs = merged_freqs.loc[~(merged_freqs.iloc[:, 1:] == 0).all(axis=1)]

    # extract frequencies for JSD calculation
    gt_tag = list(data_dict.keys())[0]
    gt_freq = merged_freqs[gt_tag].values

    js_divergences = {}
    for tag in list(data_dict.keys())[1:]:
        comparison_freq = merged_freqs[tag].values
        js_divergences[tag] = jensenshannon(gt_freq, comparison_freq)

    return js_divergences

def evaluate():
    device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
    print(device) 

    # load Hooktheory data 
    print('Loading Hooktheory data...')
    with gzip.open('./dataset/Hooktheory/Hooktheory.json.gz', 'rb') as f:
        raw_data = json.load(f)
    assert len(raw_data) == 26175

    hooktheory_events = []
    for _, example in tqdm(raw_data.items()):
        # filter out examples without 'HARMONY' tag
        if 'HARMONY' not in example['tags']:
            continue
            
        for event in example_to_events(example):
            assert event[0][0] == 'key'
            
            if len(event[1:]) < 4: 
                continue
            if event[0][1] == 0: # chords are already in C 
                hooktheory_events.append(event[1:])

            else: # chords are in different key; transpose to C
                hooktheory_events.append(transpose_chord_events(event, 0)[1:])
    print(f'Loaded {len(hooktheory_events)} events from Hooktheory dataset')

    # load models
    print('\nLoading models...')
    px, px_tokenizer, px_conf = load_models(conf.args.px_model, conf.args.px_path, conf.args.px_step, device)
    qx, qx_tokenizer, qx_conf = load_models(conf.args.qx_model, conf.args.qx_path, conf.args.qx_step, device)

    # Generate chord progressions from px
    print('\nGenerating chord progressions from Prior model P(x)...')
    batch_size = 8
    nucleus_p = 0.9
    seq_len = px_conf['seq_len']
    num_progs = 25000
    special_tokens = list(px_tokenizer.special_tokens().values())

    prior_generated = []

    with torch.no_grad():
        for _ in tqdm(range(num_progs // batch_size)):
            x = torch.full((batch_size, 1), px_tokenizer.sos, dtype=torch.int64, device=device)
            h_i = px.init_hidden(batch_size, device=device)
            while x.shape[-1] <= seq_len:
                logits, h_i = px(x[:, -1:], h_i)
                logits = truncate_to_nucleus(logits, nucleus_p)
                probs = F.softmax(logits, dim=-1)
                x_i = torch.multinomial(probs.view(batch_size, -1), 1)
                x = torch.cat([x, x_i], dim=-1)
            x = x[:, 1:].cpu().numpy()
        
            for tokens in x:
                # Filter out special tokens and append the first 4 tokens
                filtered_tokens = [token for token in tokens if token not in special_tokens]
                prior_generated.append(filtered_tokens[:4])

    # load llm chords to evaluate
    print('\nLoading llm chords...')
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
    llm_chord_events = []
    for line in llm_chords:
        try:
            chord_texts = line.strip().split(' ')
            chord_texts = [c.strip() for c in chord_texts]

            if len(chord_texts) < 4:
                continue
            # error handling: if "min" in chord, convert to "m"
            chord_texts = [c.replace('min', 'm') for c in chord_texts]
            # convert to events
            chord_events = [text_to_event(c) for c in chord_texts]
            llm_chord_events.append(chord_events)
        except:
            continue
    print('Number of llm-generated chord progressions:', len(llm_chord_events))

    rejection_sampled = []
    M = conf.args.rej_M
    if M is None:
        print('\nFinding the smallest M for rejection sampling...')
        _, M = find_smallest_M(px, qx, px_tokenizer, qx_tokenizer, llm_chord_events, device)
        print('Smallest M:', M)
    
    print('Rejection-sampling LLM chords using M =', M)
    for prog in tqdm(llm_chord_events, total=len(llm_chord_events)):
        p_x_score = score_chord_progression(prog, px, px_tokenizer, device)
        q_x_score = score_chord_progression(prog, qx, qx_tokenizer, device)

        alpha = np.exp(p_x_score - np.log(M) - q_x_score)

        if random.random() < alpha:
            rejection_sampled.append(prog)

    print(f'Number of rejection-sampled chord progressions: {len(rejection_sampled)} / {len(llm_chord_events)}')

    # Calculate Jensen-Shannon divergence (Unigram)
    hooktheory_tokens = [px_tokenizer.events_to_tokens(prog)[1:-1] for prog in hooktheory_events]
    llm_chord_tokens = [px_tokenizer.events_to_tokens(prog)[1:-1] for prog in llm_chord_events]
    rejection_sampled_tokens = [px_tokenizer.events_to_tokens(prog)[1:-1] for prog in rejection_sampled]
    data_dict = {
        'Hooktheory': hooktheory_tokens, # ground truth
        'LSTM Prior': prior_generated,
        'GPT-4o': llm_chord_tokens,
        'Amuse': rejection_sampled_tokens
    }

    print('\nCalculating Jensen-Shannon Divergence (Unigram)...')
    unigram_jsds = compute_jsd(data_dict, mode='unigram')
    for tag, jsd in unigram_jsds.items():
        print(f'Unigram JSD (Hooktheory vs. {tag}): {jsd:.4f}') 

    print('\nCalculating Jensen-Shannon Divergence (Bigram)...')
    bigram_jsds = compute_jsd(data_dict, mode='bigram')
    for tag, jsd in bigram_jsds.items():
        print(f'Bigram JSD (Hooktheory vs. {tag}): {jsd:.4f}')


def parse_arguments():
    """Command line parse."""

    # Note that 'type=bool' args should be False in default. Any string argument is recognized as "True". Do not give "--bool_arg 0"

    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument('--px_model', type=str, default='lstm',
                        help='Architecture of the model P(x) to load, in [lstm]')
    parser.add_argument('--px_path', type=str, default='./log/hooktheory/lstm/test/',
                        help='Path to the checkpoint of the model P(x)')
    parser.add_argument('--px_step', type=str, default='last',
                        help='Step of the model P(x) to load. model-[step].pt')

    parser.add_argument('--qx_model', type=str, default='lstm',
                        help='Architecture of the model Q(x) to load, in [lstm]')
    parser.add_argument('--qx_path', type=str, default='./log/llmchords/lstm/test/',
                        help='Path to the checkpoint of the model Q(x)')
    parser.add_argument('--qx_step', type=str, default='last',
                        help='Step of the model Q(x) to load. model-[step].pt')

    parser.add_argument('--rej_T', type=float, default=1.7,
                        help='Temperature for computing scores in rejection sampling')
    parser.add_argument('--rej_M', type=float, default=None,
                        help='M for rejection sampling. If None, find the smallest M from the data')

    parser.add_argument('--gpu_idx', type=int, default=0, help='which gpu to use')

    # Evaluation
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--llmchords_path', type=str, default='./dataset/llmchords/chords.txt',
                        help='Path to llmchords data file. If not exists, download and save')

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
    evaluate()