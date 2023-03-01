import torch
import torch.nn.functional as F
from tqdm import tqdm
import csv
import numpy as np

from transformers import Trainer

from metrics import aac_metrics
from eval_metrics import write_json
from pathlib import Path

# Apart from the compute_loss function, all training operations are defined in the Transformers Trainer class
class BARTAACTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if 'file_name' in inputs.keys():
            file_name = inputs.pop('file_name')
        
        # Inputs should now only contain audio_features, input_ids and attention_mask
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs['labels']
        else:
            labels = None
        
        if model.freeze_audio_enc:
            model.audio_enc.eval() # CNN14 Dropout/BatchNorm
        loss, outputs = model(**inputs)
        
        if labels is not None:
            loss = self.label_smoother({'logits': outputs}, labels)
            
        tqdm.write(str(loss.item()))
        
        return (loss, outputs) if return_outputs else loss # Do not return past key values
    
    def caption_evaluate(self, eval_dataset, tokenizer, generation_mode='beam'):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        model = self.model
        model.eval()
        
        all_labels = []
        all_preds = []
        all_filenames = []
        with torch.no_grad():
            for step, inputs in enumerate(tqdm(eval_dataloader)):
                
                inputs = self._prepare_inputs(inputs)
                
                labels = inputs.pop('labels')
                all_labels.append(labels.cpu())
                
                inputs.pop('decoder_attention_mask')
                
                file_name = inputs.pop('file_name')
                
                if generation_mode == 'beam':
                    outputs = model.generate_beam(**inputs)
                elif generation_mode == 'greedy':
                    outputs = model.generate_greedy(**inputs)
                else:
                    raise NotImplementedError
                    
                all_preds.append(F.pad(outputs, (0,512-outputs.size(1)),'constant', 1).cpu()) # Pad with pad token for concatenation
                all_filenames.extend(file_name)
                
        if all_labels != []:
            all_labels = torch.cat(all_labels, dim=0).numpy()
            print(all_labels)
        all_preds = torch.cat(all_preds, dim=0).numpy()
        print(all_preds)
        metrics, all_gt_captions, all_pred_captions = aac_metrics({'predictions': all_preds, 'label_ids': all_labels, 'filenames': all_filenames}, tokenizer)
        
        # Write outputs to disk
        write_json(metrics, Path(self.args.output_dir).joinpath('metrics_coco_'+generation_mode+'.json'))
        with open(Path(self.args.output_dir).joinpath('generated_captions_'+generation_mode+'.txt'), 'w') as f:
            for i_file in range(len(all_pred_captions)):
                f.write('----- File {} -----\n'.format(i_file))
                f.write('GT:   '+'\n')
                for i_gt in range(len(all_gt_captions[i_file])):
                    f.write('      '+all_gt_captions[i_file][i_gt]+'\n')
                f.write('Pred: '+all_pred_captions[i_file]+'\n')
        
        return metrics
    
    def caption_infer(self, test_dataset, tokenizer, generation_mode='beam'):
        test_dataloader = self.get_eval_dataloader(test_dataset)
        
        model = self.model
        model.eval()
        
        all_preds = []
        all_filenames = []
        with torch.no_grad():
            for step, inputs in enumerate(tqdm(test_dataloader)):
                inputs = self._prepare_inputs(inputs)
                
                if 'labels' in inputs.keys():
                    inputs.pop('labels')
                if 'decoder_attention_mask' in inputs.keys():
                    inputs.pop('decoder_attention_mask')
                
                file_name = inputs.pop('file_name')
                
                if generation_mode == 'beam':
                    outputs = model.generate_beam(**inputs)
                elif generation_mode == 'greedy':
                    outputs = model.generate_greedy(**inputs)
                else:
                    raise NotImplementedError
                
                all_preds.append(F.pad(outputs, (0,512-outputs.size(1)),'constant', 1).cpu()) # Pad with pad token for concatenation
                all_filenames.extend(file_name)
        
        all_preds = torch.cat(all_preds, dim=0).numpy()
        
        # Decoding
        all_pred_caps = []
        for i_pred in range(all_preds.shape[0]):
            pred_ = tokenizer.decode(all_preds[i_pred,:])
            all_pred_caps.append(pred_.replace('<|pad|>', '').replace('<|endoftext|>', '').replace('</s>', '').replace('<s>', '').replace('<pad>', ''))
        
        # Write submission-ready file
        with open(Path(self.args.output_dir).joinpath('test_output_captions_'+generation_mode+'.csv'), 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['file_name', 'caption_predicted'])
            for fname_, pcap_ in zip(all_filenames, all_pred_caps):
                writer.writerow([fname_, pcap_])
        

