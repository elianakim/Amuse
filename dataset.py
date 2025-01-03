import numpy as np
import torch

class HooktheoryDataset(torch.utils.data.Dataset):
    def __init__(self, all_events, seq_len, tokenizer, train=False):
        self._all_events = sorted(all_events)
        self._seq_len = seq_len
        self._train = train
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self._all_events)

    def __getitem__(self, idx):
        # tokenize
        events = self._all_events[idx]
        tokens = self.tokenizer.events_to_tokens(events)
        assert len(tokens) > 2

        # Shift
        inputs = np.array(tokens[:-1], dtype=np.int64)
        targets = np.array(tokens[1:], dtype=np.int64)
        
        # truncate to seq_len
        inputs = inputs[:self._seq_len]
        targets = targets[:self._seq_len]
        
        # pad to seq_len
        inputs = np.pad(inputs, [0, self._seq_len - inputs.shape[0]], constant_values = self.tokenizer.pad)
        targets = np.pad(targets, [0, self._seq_len - targets.shape[0]], constant_values = -1)

        assert inputs.shape[0] == self._seq_len
        assert targets.shape[0] == self._seq_len
            
        return inputs, targets

    