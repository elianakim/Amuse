from collections import Counter

class HooktheoryChordTokenizer:
    def __init__(
            self,
            all_events,
            max_num_chord_tokens=1200):
        
        self.id_to_token = []

        # add reserved tokens
        reserved_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        self.id_to_token.extend([('reserved', t) for t in reserved_tokens])

        # Compute chord tokens
        counts = Counter()
        for events in all_events:
            for e in events:
                if e[0] != 'chord':
                    continue
                counts[e[1:]] += 1
        self.id_to_token.extend([('chord',) + details for details, _ in counts.most_common(max_num_chord_tokens)])
        oov = sum([c for _, c in counts.most_common()[max_num_chord_tokens:]])
        print(f'chord <UNK>: {oov}/{sum(counts.values())} ({oov / sum(counts.values()) * 100:.2f}%)')

        # Create reverse map
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}
        assert len(self.id_to_token) == len(self.token_to_id)
        self.pad = self.token_to_id[('reserved', '<PAD>')]
        self.sos = self.token_to_id[('reserved', '<SOS>')]
        self.eos = self.token_to_id[('reserved', '<EOS>')]
        self.unk = self.token_to_id[('reserved', '<UNK>')]
    
    def __getitem__(self, idx):
        return self.id_to_token[idx]
    
    def __len__(self):
        return len(self.id_to_token)
    
    def events_to_tokens(self, events: list) -> list:
        tokens = []
        tokens.append(self.sos)
        for e in events:
            tokens.append(self.token_to_id.get(e, self.unk))
        tokens.append(self.eos)
        return tokens

    def tokens_to_events(self, tokens: list) -> list:
        events = []
        for t in tokens:
            t = self.id_to_token[t]
            if t[0] == 'reserved':
                if t[1] == '<EOS>':
                    break
            else:
                events.append(t)
        return events
    
    def special_tokens(self):
        return {
            'pad': self.pad,
            'sos': self.sos,
            'eos': self.eos,
            'unk': self.unk,
        }