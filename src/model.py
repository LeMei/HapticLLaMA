import torch
from torch import nn
import torch.nn.functional as F
from config import DEVICE
# from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
from transformers import LlamaTokenizer, AutoTokenizer
from modules.modeling_llama import LlamaForCausalLM
from modules.lora_layer import LoRALinear



class Model(nn.Module):

    def __init__(self, args, mode=None):

        super().__init__()
        access_token = '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'
        if mode == None:
            mode = args.mode
        updated_tokenizer = "../data/Haptic/pretrain/updated_llama_tokenizer_{}.pt".format(mode)

        self.tokenizer = AutoTokenizer.from_pretrained(updated_tokenizer)
        print('token size:', len(self.tokenizer))
        # self.llama_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B",token=access_token).to(DEVICE)
        self.llama_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B",token=access_token).to(DEVICE)
        self.llama_model.resize_token_embeddings(len(self.tokenizer))

        self.llama_model.config.pad_token_id = self.llama_model.config.eos_token_id  # Set pad_token_id explicitly, if you want this behavior

        if args.param_mode == 'lora':
            for name, module in list(self.llama_model.named_modules()): 
                if "q_proj" in name or "v_proj" in name:
                    setattr(self.llama_model, name, LoRALinear(module.in_features, module.out_features))

    def forward(self, inputs, att_mask, labels):

        output = self.llama_model(input_ids=inputs, labels=labels)
        loss, logits, hidden_states = output.loss, output.logits, output.hidden_states

        return loss, logits, hidden_states

    def generate(self, inputs, input_atts):

        generate_ids = self.llama_model.generate(inputs, attention_mask=input_atts, max_new_tokens=15,\
                                                  temperature=0.7, top_p=0.9, num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id)

        token_start_index = inputs.shape[1]

        token_ids = generate_ids[:,token_start_index:]

        generate_tokens = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        return generate_tokens

