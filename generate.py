'''
generate.py

Interactive terminal for
generating chord progressions
'''

import sys
import argparse
import random
import pathlib
import json
import pickle
import warnings
import concurrent.futures
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from openai import OpenAI

import conf
from utils import (
    read_api_keys,
    event_to_text,
    text_to_event,
    get_chord_progressions_from_openAI,
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

def truncate_unk(logits, tokenizer):
    vocab_size = logits.shape[-1]
    truncated = torch.where(
        torch.arange(vocab_size, device=logits.device).view(-1) == tokenizer.unk,
        float('-inf'),
        logits,
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

def compute_alpha(progression, p_x, q_x, p_x_tokenizer, q_x_tokenizer, M, device):
    prog_text, prog_events = progression

    p_x_score = score_chord_progression(prog_events, p_x, p_x_tokenizer, device)
    q_x_score = score_chord_progression(prog_events, q_x, q_x_tokenizer, device)
    log_alpha = p_x_score - np.log(M) - q_x_score
    alpha = np.exp(log_alpha)

    return prog_text, alpha

def generate():
    device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
    print(device) 

    # load models
    print('\nLoading models...')
    px, px_tokenizer, px_conf = load_models(conf.args.px_model, conf.args.px_path, conf.args.px_step, device)
    qx, qx_tokenizer, qx_conf = load_models(conf.args.qx_model, conf.args.qx_path, conf.args.qx_step, device)

    # get API key
    api_keys = read_api_keys("./assets/")
    client = OpenAI(api_key=api_keys['openai'])

    # interactive terminal
    while True:
        keywords = input('\n===\nEnter keywords (separated by comma): ').split(',')
        keywords = [kw.strip() for kw in keywords]

        if len(keywords) == 0:
            print('Please enter at least one keyword.')
            continue
    
        progressions = get_chord_progressions_from_openAI(keywords, "C", "Maj", 4, client)
        progressions_text_event_pairs = []
        for prog in progressions:
            try:
                prog_texts = [c.strip() for c in prog]
                if len(prog_texts) < 4:
                    continue
                prog_texts = [c.replace('min', 'm') for c in prog_texts]
                prog_events = [text_to_event(c) for c in prog_texts]
                progressions_text_event_pairs.append((' '.join(prog_texts), prog_events))
            except:
                continue
        
        scored_progressions = []
        sampled_progressions = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            tasks = []

            for _ in range(1):
                tasks.extend(executor.submit(
                    compute_alpha, instance, px, qx, px_tokenizer, qx_tokenizer, 
                    conf.args.rej_M, device) for instance in progressions_text_event_pairs)
            
            for future in concurrent.futures.as_completed(tasks):
                try:
                    prog_text, alpha = future.result()
                    scored_progressions.append((prog_text, alpha))
                    if random.random() < alpha and prog_text not in sampled_progressions:
                        sampled_progressions.append(prog_text)
                except Exception as e:
                    continue 
    
            if len(sampled_progressions) < 4:
                num_k = 4 - len(sampled_progressions)

                # sort by alpha
                scored_progressions.sort(key=lambda x: x[1], reverse=True)
                sampled_progressions.extend([prog_text for prog_text, _ in scored_progressions[:num_k]])
        
        print('\nRandom-sampled chord progressions (GPT-4o):\n')
        for prog, alpha in random.sample(scored_progressions, 4):
            print(prog)

        print('\nRejection-sampled chord progressions (Amuse):\n')
        for prog in sampled_progressions[:4]:
            print(prog)


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
                        help='Temperature for computing scores in rejection sampling. 1.7 - 3 is recommended for LSTM.')
    parser.add_argument('--rej_M', type=float, default=7.64,
                        help='M for rejection sampling. Values less than 8.0 is recommended.')

    parser.add_argument('--gpu_idx', type=int, default=0, help='which gpu to use')

    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    
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
    generate()