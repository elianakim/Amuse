import os
import csv
import json
import random
from tqdm import tqdm

import conf

_NOTES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

_NOTE_CONVERTER = {
    'C#': 'Db',
    'D#': 'Eb',
    'Gb': 'F#',
    'G#': 'Ab',
    'A#': 'Bb',
}

_CHORD_DEGREES_TO_NAME = {
    # Chords with frequency more than 100 in the Hookthoery dataset
    (4, 3): "",  # Major 
    (3, 4): "m",  # Minor
    (3, 4, 3): "m7",  # Minor 7
    (4, 3, 3): "7",  # 7
    (4, 3, 4): "maj7",  # Major 7
    (4, 3, 7): "add9",  # Add 9
    (3, 3): "dim",  # Dim
    (5, 2): "sus4",  # Sus 4
    (2, 5): "sus2",  # Sus 2
    (3, 3, 4): "m7b5",  # Half-diminished
    (5, 2, 3): "7sus4",  # 7 Sus 4
    (3, 4, 3, 4): "m9",  # Minor 9
    (3, 4, 7): "madd9",  # Minor Add 9
    (4, 3, 4, 3): "maj9",  # Major 9
    (4, 3, 3, 4, 3): "11", # Dominant 11th
    (2, 5, 3): "7sus2",  # 7 Sus 2
    (3, 4, 3, 4, 3): "m11", # Minor 11
    (3, 3, 3): "dim7",  # Dim 7
    (2, 5, 4): "maj7sus2",  # Major 7 Sus 2
    (4, 3, 3, 4): "9",  # Dominant 9th
    (2, 3, 2): "sus2sus4",  # Sus 2 Sus 4
    (4, 4): "aug",  # Aug
    (7,) : "5", # Power
    (3, 4, 3, 7): "m7add11", # Minor 7 Add 11
    (6, 1): "sus#4", # Sus Sharp 4th
    (4, 3, 3, 3): "7b9", # Dominant 7th Flat 9
    (4, 3, 14): "6", # Major 6th
    (4, 3, 10): "add11", # Add 11
    (3, 4, 4): "minmaj7", # Minor Major 7
    (4, 3, 3, 4, 3, 4): "13", # Dominant 13th
    (3, 3, 4, 3, 4): "m11b5b9", # Minor 11 Flat 5 Flat 9
    (4, 3, 3, 5): "7#9", # Dominant 7th Sharp 9
    (4, 3, 4, 3, 4): "maj9#11", # Major 9 Sharp 11
    (4, 3, 3, 10): "7b13", # Dominant 7th Flat 13
    (4, 4, 3): "maj7#5", # Major 7 Sharp 5
}

_CHORD_NAMES_TO_DEGREES = {v: k for k, v in _CHORD_DEGREES_TO_NAME.items()}

_SCALE_DEGREES_TO_NAME = {
    (2, 2, 1, 2, 2, 2): "Maj",  # Major
    (2, 1, 2, 2, 2, 1): "Dor",  # Dorian
    (1, 2, 2, 2, 1, 2): "Phr",  # Phrygian
    (2, 2, 2, 1, 2, 2): "Lyd",  # Lydian
    (2, 2, 1, 2, 2, 1): "Mix",  # Mixolydian
    (2, 1, 2, 2, 1, 2): "Min",  # Minor
    (1, 2, 2, 1, 2, 2): "Loc",  # Locrian
    (2, 1, 2, 2, 1, 3): "Hmin", # Harmonic Minor
    (1, 3, 1, 2, 1, 2): "Phdm", # Phrygian Dominant
}

_SCALE_NAMES_TO_DEGREES = {v: k for k, v in _SCALE_DEGREES_TO_NAME.items()}

def example_to_events(example: dict, remove_duplicates=True) -> list:
    '''
    Convert each example to simple list of events [scale, *tuples] 
    where each tuple represents a chord: (root_pitch_class, root_position_intervals, inversion)
    '''
    events = []

    # Add key events: ('key', root note, scale degrees)
    key_beatstamps = [key['beat'] for key in example['annotations']['keys']]
    events = [
        [('key', key['tonic_pitch_class'], tuple(key['scale_degree_intervals']))]
        for key in example['annotations']['keys']
    ]
    
    # add chord events: ('chord', root note, chord intervals in semitones, inversion) 
    # based on key_beatstamps
    current_key_idx = 0
    last_chord = None

    for chord in example['annotations']['harmony']:
        onset = float(chord['onset'])

        # Update key index based on onset
        while current_key_idx + 1 < len(key_beatstamps) and onset >= key_beatstamps[current_key_idx + 1]:
            current_key_idx += 1
            last_chord = None

        if tuple(chord['root_position_intervals']) == ():
            break

        processed_chord = (
            'chord',
            chord['root_pitch_class'],
            tuple(chord['root_position_intervals']),
            chord['inversion']
        )

        if remove_duplicates:
            if processed_chord != last_chord:
                events[current_key_idx].append(processed_chord)
                last_chord = processed_chord
        else:
            events[current_key_idx].append(processed_chord)

    return events

def transpose_chord_events(chord_progressions: list, target_key=0) -> list:
    '''
    Transpose chords to a target key
    Input: list of chord events and a target key index (C = 0)
    Output: list of chord events (converted to the target key)
    '''
    def transpose_note(note, steps):
        return (note + steps) % 12
    
    def get_transpose_steps(original_key, target_key=target_key):
        # determine the interval between the keys
        return (target_key - original_key) % 12

    converted_progressions = []
    transpose_steps = None

    for event in chord_progressions:
        if event[0] == 'key':
            key, scale_degrees = event[1], event[2]
            transpose_steps = get_transpose_steps(key, target_key)
            converted_progressions.append(('key', transpose_note(key, transpose_steps), scale_degrees))
        elif event[0] == 'chord': 
            assert(transpose_steps is not None)
            root, chord_intervals, inversion = event[1], event[2], event[3]
            converted_root = transpose_note(root, transpose_steps)
            converted_progressions.append(('chord', converted_root, chord_intervals, inversion))

    return converted_progressions

def event_to_text(event: tuple) -> str:
    '''
    Converts Hooktheory event into key/chord in text.
    Example input (key): ('key', 5, (2, 1, 2, 2, 1, 2))
    Example input (chord): ('chord', 5, (3, 4, 7), 0)
    output example: 'Emadd9'
    '''
    event_type, root_idx, *details = event
    root = _NOTES[root_idx]

    if event_type == 'key':
        mode = _SCALE_DEGREES_TO_NAME[details[0]]
        return f'{root} {mode}'
    elif event_type == 'chord':
        intervals, inversion = details
        quality = _CHORD_DEGREES_TO_NAME[intervals]
        if inversion > 0:
            chord_notes = [(root_idx + sum(intervals[:i])) % 12 for i in range(len(intervals) + 1)]
            bass_note = _NOTES[chord_notes[inversion]]
            return f'{root}{quality}/{bass_note}'
        return f'{root}{quality}'
    else:
        raise ValueError(f'Unknown event type: {event[0]}')

def text_to_event(text: str) -> tuple:
    '''
    Converts chord text to a Hooktheory event.
    Example input: 'Gsus4'
    Example output: ('chord', 7, (5, 2), 0)
    '''
    text = text.strip()

    # Handle inversion
    if "/" in text:
        chord_text, bass_note = text.split("/")
        bass_note_idx = _NOTES.index(_NOTE_CONVERTER.get(bass_note, bass_note))
    else:
        chord_text, bass_note_idx = text, 0

    # Extract root and quality
    for note in _NOTES + list(_NOTE_CONVERTER.keys()):
        if chord_text.startswith(note):
            root = _NOTE_CONVERTER.get(note, note)
            quality = chord_text[len(note):]
            try:
                root_idx = _NOTES.index(root)
                intervals = _CHORD_NAMES_TO_DEGREES[quality]
                return ('chord', root_idx, intervals, bass_note_idx)
            except KeyError:
                continue

    # if not found, return dummy chord
    # TODO: debug
    return ('chord', 0, (-1, -1), 0)

def read_api_keys(config_dir="./configs"):
    """
    Read API keys from a CSV file.
    """
    path = os.path.join(config_dir, 'api_keys.csv')

    api_keys = dict()
    if not os.path.exists(path):
        raise RuntimeError(f'Cannot find API keys in the file: {path}')
    
    with open(path) as f:
        rows = csv.DictReader(f)
        for row in rows:
            host = row['host']  # 'openai'
            api_keys[host] = row['key']
    return api_keys

def get_chord_progressions_from_openAI(keywords, key, mode, bars, client):
    """
    Generate chord progressions from OpenAI API.
    """
    _GPT_MODEL = "gpt-4o"

    prompt_path = "./assets/prompts/amuse_chord_generation.json"
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt = json.load(f)
    
    user_message = {
        "role": "user",
        "content": f"User keywords: {', '.join(keywords)}\nKey: {key}\nMode: {mode}\nBars: {bars}"
    }

    completion = client.chat.completions.create(
        model=_GPT_MODEL,
        temperature=1.0,
        messages= prompt["messages"] + [user_message]
    )

    response = completion.choices[0].message.content
    progressions = response.split("\n")
    return_progressions = []
    for progression in progressions:
        p_list = progression.strip().split(" ")
        if len(p_list) == bars:
            return_progressions.append(p_list)
    
    return return_progressions

def generate_llm_chords(write_path, client):
    """
    Generate and save LLM chords from OpenAI API.
    """
    # read keywords from ./assets/music_keywords.txt and create as list
    # keywords are divided in lines
    with open('./assets/music_keywords.txt', 'r') as f:
        music_keywords = f.read().splitlines()

    modes_prob = {
        # Calculated from the Hooktheory dataset
        "Maj": 0.5036,
        "Min": 0.3933,
        "Dor": 0.0377,
        "Phr": 0.0108,
        "Lyd": 0.0386,
        "Mix": 0.0105,
        "Loc": 0.0024,
        "Hmin": 0.0017,
        "Phdm": 0.0013
    }

    # Ensure the directory exists
    dir = os.path.dirname(write_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    chord_lines = []
    with open(write_path, 'w') as f:
        f.write('')

        for _ in tqdm(range(int(conf.args.llmchords_num / 30))):
            # random sample keyword(s)
            sampled_keywords = random.sample(music_keywords, 2)
            # sample mode
            mode = random.choices(list(modes_prob.keys()), weights=list(modes_prob.values()), k=1)[0]
            progressions = get_chord_progressions_from_openAI(sampled_keywords, "C", mode, 4, client)
            for progression in progressions:
                try:
                    line = ' '.join(progression) + '\n'
                    f.write(line)
                    chord_lines.append(line)
                except Exception as e:
                    print(f"Error writing progression: {e} when writing: {progression}")
    
        return chord_lines