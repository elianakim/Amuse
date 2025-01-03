args = None

HooktheoryOpt = {
    'name': 'hooktheory',
    'batch_size': 32,
    'dropout': 0.2,
    'seq_len': 32,
    
    'model_embed_dim': 512,
    'model_hidden_dim': 512,
    'model_num_layers': 2,

    'lr': 1e-5,
    'weight_decay': 0.0,
    'bidirectional': False,

    'file_path': './dataset/Hooktheory',
}

LLMChordsOpt = {
    'name': 'llmchords',
    'batch_size': 32,
    'dropout': 0.2,
    'seq_len': 32,

    'model_embed_dim': 256,
    'model_hidden_dim': 256,
    'model_num_layers': 2,

    'lr': 1e-5,
    'weight_decay': 0.0,
    'bidirectional': False,

    'file_path': './dataset/llmchords',
}