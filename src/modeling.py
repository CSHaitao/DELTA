# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'
from typing import Optional, Dict
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForMaskedLM 
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers import AutoTokenizer
from transformers.models.bert.modeling_bert import BertLayer
from transformers.modeling_outputs import MaskedLMOutput
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl

@dataclass
class MaskedLMOutputWithLogs(MaskedLMOutput):
    logs: Optional[Dict[str, any]] = None

class BertForDELTA(BertForMaskedLM):
    def __init__(
        self,
        config,
        tokenizer: AutoTokenizer,
        use_decoder_head: bool = True,
        n_head_layers: int = 2,
        enable_head_mlm: bool = True,
        head_mlm_coef: float = 1.0,
        translation_coef: float = 1.0,
        info_nce_coef: float = 0.05,
        temperature: float = 1.0,
        top_align: float = 0.5,
        n_shallow_decoder_layers = 1,
        n_deep_decoder_layers = 6,
    ):  
        
        super().__init__(config)
        if use_decoder_head:
            
            self.reason_shallow_decoder = nn.ModuleList(
                [BertLayer(config) for _ in range(n_shallow_decoder_layers)]
            )        
            self.jud_shallow_decoder = nn.ModuleList(
                [BertLayer(config) for _ in range(n_shallow_decoder_layers)]
            )
            
            self.fact_shallow_decoder = nn.ModuleList(
                [BertLayer(config) for _ in range(n_shallow_decoder_layers)]
            )  
            self.fact_shallow_decoder.apply(self._init_weights)
            self.reason_shallow_decoder.apply(self._init_weights)
            self.jud_shallow_decoder.apply(self._init_weights)
            
            deep_decoder_config = copy.deepcopy(config)
            deep_decoder_config.is_decoder = True
            deep_decoder_config.add_cross_attention = True
            deep_decoder_config.num_attention_heads = 8
            self.reason_deep_decoder = nn.ModuleList(
                [BertLayer(deep_decoder_config) for _ in range(n_deep_decoder_layers)]
            )    
            self.reason_deep_decoder.apply(self._init_weights)

            self.jud_deep_decoder = nn.ModuleList(
                    [BertLayer(deep_decoder_config) for _ in range(n_deep_decoder_layers)]
                ) 
            self.jud_deep_decoder.apply(self._init_weights) 
        self.cross_entropy = nn.CrossEntropyLoss()
        self.use_decoder_head = use_decoder_head
        self.n_head_layers = n_head_layers
        self.enable_head_mlm = enable_head_mlm
        self.head_mlm_coef = head_mlm_coef
        self.translation_cofe = translation_coef
        self.info_nce_cofe = info_nce_coef
        self.temperature = temperature
        self.top_align = top_align
        self.tokenizer = tokenizer

        self.info_NCE_loss = nn.CrossEntropyLoss(reduction='mean')
        self.trans_reason_cls = BertOnlyMLMHead(config)
        self.n_deep_decoder_layers = n_deep_decoder_layers
  
    def forward(self, **model_input):
     
        lm_out: MaskedLMOutput = super().forward(
            input_ids = model_input['input_ids'],
            attention_mask = model_input['attention_mask'],
            labels=model_input['labels'],
            output_hidden_states=True,
            return_dict=True
        ) 
        device = model_input['input_ids'].device
        encoder_hidden_states = lm_out.hidden_states[-1]
        cls_hiddens = lm_out.hidden_states[-1][:, 0] #
        logs = dict()
        
        encoder_attention_mask = self.get_extended_attention_mask(
                model_input['attention_mask'],
                model_input['attention_mask'].shape,
                model_input['attention_mask'].device,
                see_after=True
            )
        
        #### encoder_mlm_loss
        loss = lm_out.loss
        logs["encoder_mlm_loss"] = lm_out.loss.item() 
        
        if self.use_decoder_head and self.enable_head_mlm:
            
            decoder_embedding_output = self.bert.embeddings(input_ids=model_input['reason_input_ids']) 
            decoder_attention_mask = self.get_extended_attention_mask(
                                        model_input['reason_attention_mask'],
                                        model_input['reason_attention_mask'].shape,
                                        model_input['reason_attention_mask'].device,
                                        see_after=True
                                    )
            hiddens = torch.cat([cls_hiddens.unsqueeze(1), decoder_embedding_output[:, 1:]], dim=1)
            for layer in self.reason_shallow_decoder:
                layer_out = layer(
                    hiddens,
                    decoder_attention_mask,
                )
                hiddens = layer_out[0]
                
            head_mlm_loss = self.mlm_loss(hiddens, model_input['reason_labels']) * self.head_mlm_coef 
            logs["reason_loss"] = head_mlm_loss.item()
            loss += head_mlm_loss


            decoder_embedding_output = self.bert.embeddings(input_ids=model_input['cause_input_ids']) 
            decoder_attention_mask = self.get_extended_attention_mask(
                                        model_input['cause_attention_mask'],
                                        model_input['cause_attention_mask'].shape,
                                        model_input['cause_attention_mask'].device,
                                        see_after=True
                                    )
            # Concat cls-hiddens of span A & embedding of span B
            hiddens = torch.cat([cls_hiddens.unsqueeze(1), decoder_embedding_output[:, 1:]], dim=1)

            for layer in self.jud_shallow_decoder:
                layer_out = layer(
                    hiddens,
                    decoder_attention_mask,
                )
                hiddens = layer_out[0]
        
            cause_mlm_loss = self.mlm_loss(hiddens, model_input['cause_labels']) * self.head_mlm_coef ###decoder的loss比例
            logs["cause_loss"] = cause_mlm_loss.item()
            loss += cause_mlm_loss
        
 
         
        
            layer_out,attention_probs = self.deep_decoder_forward(
                input_ids_with_bos=model_input['cause_input_ids_with_bos'], # begin with <bos> (use [PAD])
                attention_mask_bos=model_input['cause_attention_mask_bos'],
                encoder_hidden_states=encoder_hidden_states.detach(),
                encoder_attention_mask=encoder_attention_mask.detach(),
                reason=True,
                alignment=False
                )
            cause_translation_loss,pred_tokens = self.trans_loss(layer_out[0],model_input['cause_input_ids_with_eos'],reason=True) # end with <eos> (use [SEP])
           

            attention_label_Gp = self.layer_avg_baseline(attention_probs)            
          
   
            cause_info_nce_loss = self.calc_constractive_learning_loss(
                attention_label_Gp=attention_label_Gp.detach(),
                tgt_attention_mask=model_input['cause_attention_mask_bos'].detach(),
                src_attention_mask=model_input['attention_mask'].detach(),
                encoder_hidden_states=encoder_hidden_states.detach(),
                cls_hidden_states=cls_hiddens,
                device=device,
                )
      
            loss += cause_translation_loss * self.translation_cofe
    
            loss += cause_info_nce_loss * self.info_nce_cofe
            
            
            logs["reason_translation_loss"] = cause_translation_loss.item() * self.translation_cofe
            logs["reason_info_nce_loss"] = cause_info_nce_loss.item() * self.info_nce_cofe
            logs['loss'] = loss.item()
            
        return MaskedLMOutputWithLogs(
            loss=loss,
            logits=lm_out.logits,
            hidden_states=lm_out.hidden_states,
            attentions=lm_out.attentions,
            logs=logs,   
        )

    
    
    def deep_decoder_forward(self,input_ids_with_bos,attention_mask_bos,encoder_hidden_states,encoder_attention_mask,reason=True,alignment=False):
        decoder_embedding_output = self.bert.embeddings(input_ids=input_ids_with_bos)
        decoder_attention_mask = self.get_extended_attention_mask(
                                        attention_mask_bos,
                                        attention_mask_bos.shape,
                                        attention_mask_bos.device,
                                        see_after=alignment 
                                    )
        hiddens = decoder_embedding_output
        if (reason):
            deep_decoder_layers = self.reason_deep_decoder
        else :
            deep_decoder_layers = self.jud_deep_decoder
        for i,layer in enumerate(deep_decoder_layers):
            layer_out = layer(
                hidden_states = hiddens,
                attention_mask = decoder_attention_mask,
                encoder_hidden_states = encoder_hidden_states,
                encoder_attention_mask = encoder_attention_mask,
                output_attentions = True,
            )
            hiddens = layer_out[0]
            if i == (self.n_deep_decoder_layers-1 ): 
                middle_layer_attention_probs = layer_out[2]
        return layer_out,middle_layer_attention_probs

    def layer_avg_baseline(self,attention_probs):
        avg_attention_probs = torch.mean(attention_probs,dim=1)
        return avg_attention_probs
        
  
    def calc_constractive_learning_loss(self,attention_label_Gp,tgt_attention_mask,src_attention_mask,encoder_hidden_states,cls_hidden_states,device):
        
        batch_size = attention_label_Gp.shape[0]
        batch_avg_important_hidden_states = []
        batch_avg_unimportant_hidden_states = []
        for i in range(batch_size):
            reason_seq_len = torch.sum(tgt_attention_mask[i]).item()
            fact_seq_len = torch.sum(src_attention_mask[i]).item()
            attention_metric = attention_label_Gp[i][:reason_seq_len][:fact_seq_len] 
            token_attention_sums = torch.sum(attention_metric,dim=0) 
            sorted_fact_tokens = torch.argsort(token_attention_sums, descending=True)
            fact_important_tokens = sorted_fact_tokens[:int(self.top_align*fact_seq_len)]
            fact_unimportant_tokens = sorted_fact_tokens[int((1-self.top_align)*fact_seq_len):int(fact_seq_len)]
            important_hidden_states = encoder_hidden_states[i][fact_important_tokens,:]    
            important_hidden_states = torch.mean(important_hidden_states,dim=0)
            batch_avg_important_hidden_states.append(important_hidden_states)
            
            unimportant_hidden_states = encoder_hidden_states[i][fact_unimportant_tokens,:]
            batch_avg_important_hidden_states.append(torch.mean(unimportant_hidden_states,dim=0))
         
        batch_avg_important_hidden_states = torch.stack(batch_avg_important_hidden_states,dim=0)
        return self.constractive_learning_loss(cls_hidden_states=cls_hidden_states,pos_hidden_states=batch_avg_important_hidden_states)
    
   

    def mlm_loss(self, hiddens, labels):
        pred_scores = self.cls(hiddens)
        masked_lm_loss = self.cross_entropy(
            pred_scores.view(-1, self.config.vocab_size),
            labels.view(-1)
        )
        return masked_lm_loss
    
    
    def trans_loss(self,hiddens,labels,reason:bool):  ####思索一下这个改进一下是不是合乎翻译。
        if reason:
            pred_scores = self.trans_reason_cls(hiddens)
        
        pred_seq_token = torch.argmax(pred_scores,dim=2)
        masked_lm_loss = self.cross_entropy(
            pred_scores.view(-1, self.config.vocab_size),
            labels.view(-1)
        )
        return masked_lm_loss,pred_seq_token
    
    def constractive_learning_loss(self,cls_hidden_states,pos_hidden_states):
    
        
        # cls_hidden_states = F.normalize(cls_hidden_states, dim=1)
        # pos_hidden_states = F.normalize(pos_hidden_states, dim=1)
        batch_size = cls_hidden_states.shape[0]
        
        similarity_pos = torch.matmul(cls_hidden_states, pos_hidden_states.transpose(0, 1)) / self.temperature
        similarity_pos = similarity_pos.view(cls_hidden_states.size(0), -1)

        labels = torch.arange(0,2*batch_size,2).to(cls_hidden_states.device)
        # labels = torch.arange(cls_hidden_states.size(0), device=cls_hidden_states.device, dtype=torch.long)
      
        return self.info_NCE_loss(similarity_pos,labels)
    
    def get_extended_attention_mask(self, attention_mask, input_shape, device,see_after) :
        """
        see_after = True:
        [
            [1,0,0,0],
            [1,1,0,0],
            [1,1,1,0],
            [1,1,1,1]
        ]
        see_after = False:
        [
            [1,1,1,0],
            [1,1,1,0],
            [1,1,1,0],
            [1,1,1,0]
        ]
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if see_after == False:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    causal_mask = torch.cat(
                        [
                            torch.ones(
                                (batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )

                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask