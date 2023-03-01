from pathlib import Path
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import torch.utils.data
import numpy as np

import string

import h5py
import csv


class AACDataset(torch.utils.data.Dataset):
    def __init__(self, settings, data_dir, split, tokenizer):
        super(AACDataset, self).__init__()
        
        # Load audio features
        self.audio_data = h5py.File(data_dir.joinpath(split+'_audio_logmels.hdf5'), 'r')
        # Load captions
        self.text_data = []
        
        with open(data_dir.joinpath(split+'_text.csv'), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)
            for r in reader:
                self.text_data.append([r[1],r[2],r[3]]) # 1: audio file id, 2: audio file name, 3: raw caption, 4: processed caption (lowercase no punctuation)
        
        self.max_audio_len = settings['data']['max_audio_len']
        self.max_caption_tok_len = settings['data']['max_caption_tok_len']
        self.input_name = settings['data']['input_field_name']
        self.output_name = settings['data']['output_field_name']
        
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.text_data)
        
    def __getitem__(self, item):
        # ----- Labels/Decoder inputs -----
        ou_e = self.text_data[item][2]
        
        if ou_e is not None and ou_e!='':
            ou_e = ou_e.translate(str.maketrans('', '', string.punctuation))
            ou_e = ou_e.lower()
            
            tok_e = self.tokenizer(ou_e, max_length=self.max_caption_tok_len, return_tensors='pt', padding='max_length')
            if tok_e['input_ids'].size(1) > self.max_caption_tok_len:
                print('Found caption longer than max_caption_tok_len parameter ({} tokens).'.format(tok_e['input_ids'].size(1)))
                tok_e['input_ids'] = tok_e['input_ids'][:,:self.max_caption_tok_len]
                tok_e['attention_mask'] = tok_e['attention_mask'][:,:self.max_caption_tok_len]
        else:
            tok_e = {'input_ids': None, 'attention_mask': None}
        
        # ----- Audio conditioning -----
        in_e = self.audio_data[self.text_data[item][0]][()]
        
        in_e = torch.Tensor(in_e).float().unsqueeze(0)
        
        in_e = in_e.squeeze()
        if len(list(in_e.size())) == 1:
            in_e = in_e.unsqueeze(0)
        
        # ----- Reformat audio inputs -----
        audio_att_mask = torch.zeros((self.max_audio_len,)).long()
        audio_att_mask[:in_e.size(0)] = 1
        # Audio embeddings sequence length is divided by 64 by the CNN14 encoder
        audio_att_mask = audio_att_mask[::64]
        
        if in_e.size(0) > self.max_audio_len:
            print('Found audio longer than max_audio_len parameter ({} frames).'.format(in_e.size(0)))
            in_e = in_e[:self.max_audio_len, :]
        elif in_e.size(0) < self.max_audio_len:
            in_e = torch.cat([in_e, torch.zeros(self.max_audio_len - in_e.size(0), in_e.size(1)).float()])
        
        return {'audio_features': in_e,
                'attention_mask': audio_att_mask,
                'decoder_attention_mask': tok_e['attention_mask'].squeeze() if tok_e['attention_mask'] is not None else None,
                'file_name': self.text_data[item][1],
                'labels': tok_e['input_ids'].squeeze().long() if tok_e['input_ids'] is not None else None}
    


# Modification of the transformers default_data_collator function to allow string and list inputs
InputDataClass = NewType("InputDataClass", Any)
def default_data_collator(features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:
        - ``label``: handles a single value (int or float) per object
        - ``label_ids``: handles a list of values per object
    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    """

    # In this function we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.
    
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, List) and v != [] and isinstance(v[0], str):
                batch[k] = [f[k] for f in features]
            else:
                batch[k] = torch.tensor([f[k] for f in features])
        elif k not in ("label", "label_ids") and v is not None: # str
            batch[k] = [f[k] for f in features]
    
    return batch

def get_dataset(split, settings, tokenizer):
    #data_dir = Path(settings['data']['root_dir'], settings['data']['features_dir'])
    data_dir = Path(settings['data']['root_dir'])
    if split == 'training' and settings['workflow']['validate']:
        return AACDataset(settings, data_dir, 'development', tokenizer), \
               AACDataset(settings, data_dir, 'validation', tokenizer)
    else:
        return AACDataset(settings, data_dir, split, tokenizer), None
        
