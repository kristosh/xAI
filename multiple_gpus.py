import os
os.environ['HF_HOME'] = "/local/athanasiadisc/cache"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed
import time
import pdb

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-small-4k-instruct", use_fast=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-small-4k-instruct", torch_dtype=torch.float16, low_cpu_mem_usage=True,device_map=local_rank)

pdb.set_trace()
ds_model = deepspeed.init_inference(
    model=model, mp_size=world_size, dtype=torch.float16, replace_method="auto", replace_with_kernel_inject=True)

start = time.time()
prompt = 'Write a detailed note on AI'
inputs = tokenizer.encode(f"<human>: {prompt} \n<bot>:", return_tensors='pt').to(model.device)
outputs = ds_model.generate(inputs, max_new_tokens=2000)
output_str = tokenizer.decode(outputs[0])
print(output_str)
end = time.time()
print('Inference Time is:', end - start)
                                        