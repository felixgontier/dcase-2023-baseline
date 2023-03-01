import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BartConfig, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import BartEncoder
from transformers.modeling_outputs import BaseModelOutput

from torch.nn import Linear, LayerNorm
from transformers.models.bart.modeling_bart import BartAttention

from audio_encoders import CNN14Encoder

class BARTAAC(nn.Module):
    def __init__(self, settings, device):
        super().__init__()
        
        self.device = device
        
        # Audio encoder from t6b
        self.audio_enc = CNN14Encoder(out_dim=300) # Fixed
        state_dict = torch.load(settings['lm']['audio_enc_path'], map_location=device)
        self.audio_enc.load_state_dict(state_dict)
        self.freeze_audio_enc = settings['lm']['freeze_audio_enc']
        if self.freeze_audio_enc:
            for p in self.audio_enc.parameters():
                p.requires_grad = False
        
        
        
        # Main model configuration
        bart_config = BartConfig(vocab_size=settings['lm']['config']['vocab_size'],
                                encoder_layers=settings['lm']['config']['encoder_layers'],
                                encoder_ffn_dim=settings['lm']['config']['encoder_ffn_dim'],
                                encoder_attention_heads=settings['lm']['config']['encoder_attention_heads'],
                                decoder_layers=settings['lm']['config']['decoder_layers'],
                                decoder_ffn_dim=settings['lm']['config']['decoder_ffn_dim'],
                                decoder_attention_heads=settings['lm']['config']['decoder_attention_heads'],
                                activation_function=settings['lm']['config']['activation_function'],
                                d_model=settings['lm']['config']['d_model'],
                                dropout=settings['lm']['config']['dropout'],
                                attention_dropout=settings['lm']['config']['attention_dropout'],
                                activation_dropout=settings['lm']['config']['activation_dropout'],
                                classifier_dropout=settings['lm']['config']['classifier_dropout'],
                                max_length=settings['lm']['generation']['max_length'],
                                min_length=settings['lm']['generation']['min_length'],
                                early_stopping=settings['lm']['generation']['early_stopping'],
                                num_beams=settings['lm']['generation']['num_beams'],
                                length_penalty=settings['lm']['generation']['length_penalty'],
                                no_repeat_ngram_size=settings['lm']['generation']['no_repeat_ngram_size'])
        print(bart_config)
        
        # Other parameters
        audio_emb_size = settings['adapt']['audio_emb_size']
        lm_emb_size = bart_config.d_model
        pretrained_lm = settings['lm']['pretrained']
        n_adapt_layers = settings['adapt']['nb_layers']
        
        # Audio features to d_model embeddings
        if n_adapt_layers >= 1:
            audio_adapt_list = [nn.Linear(audio_emb_size, lm_emb_size)]
            for i_adapt in range(n_adapt_layers-1):
                audio_adapt_list.append(nn.ReLU(inplace=True))
                audio_adapt_list.append(nn.Linear(lm_emb_size, lm_emb_size))
            self.audio_adapt = nn.Sequential(*audio_adapt_list)
        else:
            self.audio_adapt = None
        
        if pretrained_lm is not None: # Bypass model configuration to load a pre-trained model (e.g. facebook/bart-base)
            self.bart_lm = BartForConditionalGeneration.from_pretrained(pretrained_lm)
        else:
            self.bart_lm = BartForConditionalGeneration(bart_config)
        
        # Freezing
        if settings['lm']['freeze']['all']:
            for p in self.bart_lm.parameters():
                p.requires_grad = False
            for p in self.bart_lm.model.encoder.embed_positions.parameters():
                p.requires_grad = True
            for p in self.bart_lm.model.encoder.layers[0].self_attn.parameters():
                p.requires_grad = True
        if settings['lm']['freeze']['dec']:
            for p in self.bart_lm.model.shared.parameters():
                p.requires_grad = False
            for p in self.bart_lm.model.decoder.parameters():
                p.requires_grad = False
            for p in self.bart_lm.lm_head.parameters():
                p.requires_grad = False
        if settings['lm']['freeze']['enc']:
            for p in self.bart_lm.model.encoder.parameters():
                p.requires_grad = False
        if settings['lm']['freeze']['attn']:
            for l in self.modules():
                if isinstance(l, BartAttention):
                    for p in l.parameters():
                        p.requires_grad = False
        if settings['lm']['freeze']['mlp']:
            for l in self.bart_lm.modules():
                if isinstance(l, Linear):
                    for p in l.parameters():
                        p.requires_grad = False
        if settings['lm']['freeze']['dec_attn']:
            for l in self.bart_lm.model.decoder.modules():
                if isinstance(l, BartAttention):
                    for p in l.parameters():
                        p.requires_grad = False
        if settings['lm']['freeze']['dec_mlp']:
            for l in self.bart_lm.model.decoder.layers:
                for p in l.fc1.parameters():
                    p.requires_grad = False
                for p in l.fc2.parameters():
                    p.requires_grad = False
        if settings['lm']['freeze']['dec_self_attn']:
            for l in self.bart_lm.model.decoder.layers:
                for p in l.self_attn.parameters():
                    p.requires_grad = False
        if settings['lm']['freeze']['enc_mlp']:
            for l in self.bart_lm.model.encoder.layers:
                for p in l.fc1.parameters():
                    p.requires_grad = False
                for p in l.fc2.parameters():
                    p.requires_grad = False
        if settings['lm']['freeze']['enc_attn']:
            for l in self.bart_lm.model.encoder.layers:
                for p in l.self_attn.parameters():
                    p.requires_grad = False
        
    # Custom implementation of the Bart forward function
    def forward(self,
                audio_features=None,
                cond_tokens=None,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                head_mask=None,
                decoder_head_mask=None,
                encoder_outputs=None,
                past_key_values=None,
                inputs_embeds=None,
                decoder_inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=True,
        ):
        
        audio_embs = self.audio_enc(audio_features, skip_fc=True)
        
        if self.audio_adapt is not None:
            audio_embs = self.audio_adapt(audio_embs)
        else:
            audio_embs = audio_features
        
        # Encoder pass
        encoder_outputs = self.bart_lm.model.encoder(
                    input_ids=None,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=audio_embs,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True)['last_hidden_state']
        
        encoder_outputs = [encoder_outputs]
        
        # Decoder-only pass
        outputs = self.bart_lm(input_ids=None,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    head_mask=head_mask,
                    decoder_head_mask=decoder_head_mask,
                    encoder_outputs=encoder_outputs,
                    past_key_values=past_key_values,
                    inputs_embeds=None,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
          )
        
        return outputs['loss'], outputs['logits']
    
    def generate_greedy(self,
                audio_features=None,
                cond_tokens=None,
                attention_mask=None,
                inputs_embeds=None
        ):
        
        audio_embs = self.audio_enc(audio_features, skip_fc=True)
        
        if self.audio_adapt is not None:
            audio_embs = self.audio_adapt(audio_embs)
        else:
            audio_embs = audio_features
        
        encoder_outputs = self.bart_lm.model.encoder(
                input_ids=None,
                attention_mask=attention_mask,
                head_mask=None,
                inputs_embeds=audio_embs,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=True)
        
        max_len = self.bart_lm.config.max_length
        cur_len = 0
        
        input_ids = torch.zeros((audio_embs.size(0),1)).long().to(self.device)
        input_ids[:, 0] = self.bart_lm.config.decoder_start_token_id
        
        outputs = self.bart_lm(input_ids=None,
                        attention_mask=attention_mask,
                        decoder_input_ids=input_ids,
                        decoder_attention_mask=None,
                        inputs_embeds=audio_embs,
                        use_cache=True,
                        return_dict=True)
        
        _next_token = torch.argmax(outputs['logits'][:, -1, :], dim=-1)
        _past = outputs['past_key_values']
        _encoder_last_hidden_state = outputs['encoder_last_hidden_state']
        input_ids = torch.cat([input_ids, _next_token.unsqueeze(-1)], dim=-1)
        
        # Override with bos token
        input_ids[:, 1] = self.bart_lm.config.bos_token_id
        cur_len += 1
        
        while cur_len < max_len:
            model_inputs = self.bart_lm.prepare_inputs_for_generation(input_ids, past=_past, attention_mask=attention_mask, encoder_outputs=[encoder_outputs['last_hidden_state']])
            outputs = self.bart_lm(**model_inputs)
            _next_token = torch.argmax(outputs['logits'][:, -1, :], dim=-1)
            _past = outputs['past_key_values']
            _encoder_last_hidden_state = outputs['encoder_last_hidden_state']
            input_ids = torch.cat([input_ids, _next_token.unsqueeze(-1)], dim=-1)
            cur_len += 1
        
        #print(input_ids)
        return input_ids
    
    def generate_beam(self,
                audio_features=None,
                cond_tokens=None,
                attention_mask=None,
                inputs_embeds=None
        ):
        
        self.bart_lm.force_bos_token_to_be_generated=True
        
        audio_embs = self.audio_enc(audio_features, skip_fc=True)
        
        if self.audio_adapt is not None:
            audio_embs = self.audio_adapt(audio_embs)
        else:
            audio_embs = audio_features
        
        # Encoder pass
        encoder_outputs = self.bart_lm.model.encoder(
                input_ids=None,
                attention_mask=attention_mask,
                head_mask=None,
                inputs_embeds=audio_embs,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=True)
        
        input_ids = torch.zeros((encoder_outputs['last_hidden_state'].size(0),1)).long().to(self.device)
        input_ids[:, 0] = self.bart_lm.config.decoder_start_token_id
        decoder_attention_mask = torch.ones((encoder_outputs['last_hidden_state'].size(0),1)).long().to(self.device)
        # Beam decoding
        outputs = self.bart_lm.generate(input_ids=None,
                    attention_mask=attention_mask,
                    decoder_input_ids=input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    encoder_outputs=encoder_outputs,
                    head_mask=None,
                    decoder_head_mask=None,
                    inputs_embeds=None,
                    decoder_inputs_embeds=None,
                    use_cache=None,
                    output_attentions=None,
                    output_hidden_states=None)
        
        #print(outputs)
        return outputs
        

