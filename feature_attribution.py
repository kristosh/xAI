import os
os.environ['HF_HOME'] = "/local/athanasiadisc/cache"

from huggingface_hub import whoami

# import pdb
# pdb.set_trace()
# list available attribution methods
# import inseq

# inseq.list_feature_attribution_methods()

# # load a model from HuggingFace model hub and define the feature attribution 
# # method you want to use
# mdl_gpt2 = inseq.load_model("gpt2", "lime")

# # compute the attributions for a given prompt
# attr = mdl_gpt2.attribute(
#     "Hello ladies and",
#     generation_args={"max_new_tokens": 50},
#     n_steps=500,
#     internal_batch_size=50 )

# # display the generated attributions
# attr.show()

# import pdb
# pdb.set_trace()


import warnings

import bitsandbytes as bnb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from captum.attr import (
    FeatureAblation, 
    ShapleyValues,
    LayerIntegratedGradients, 
    LLMAttribution, 
    LLMGradientAttribution, 
    TextTokenInput, 
    TextTemplateInput,
    ProductBaselines,
)

# Ignore warnings due to transformers library
warnings.filterwarnings("ignore", ".*past_key_values.*")
warnings.filterwarnings("ignore", ".*Skipping this token.*")

def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    max_memory = "10000MB"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # dispatch efficiently the model on the available ressources
        max_memory = {i: max_memory for i in range(n_gpus)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config

model_name = "microsoft/Phi-3-medium-4k-instruct"
#model_name = "meta-llama/Llama-2-13b-chat-hf" 

bnb_config = create_bnb_config()

model, tokenizer = load_model(model_name, bnb_config)

eval_prompt = "Dave lives in Palm Coast, FL and is a lawyer. His personal interests include"

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
model.eval()
with torch.no_grad():
    output_ids = model.generate(model_input["input_ids"], max_new_tokens=150)[0]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(response)


fa = FeatureAblation(model)
llm_attr = LLMAttribution(fa, tokenizer)