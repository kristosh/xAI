import os
os.environ['HF_HOME'] = "/local/athanasiadisc/cache"

from transformers import AutoTokenizer, AutoModel, utils
from bertviz import model_view
utils.logging.set_verbosity_error()  # Suppress standard warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pdb
tokenizer = AutoTokenizer.from_pretrained("microsoft/xtremedistil-l12-h384-uncased")

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/xtremedistil-l12-h384-uncased",
    device_map = "cuda:0",
    torch_dtype = "auto",
    trust_remote_code = True
)

# Create a pipeline
generator = pipeline(
    "text-generation",
    model = model,
    tokenizer = tokenizer,
    return_full_text= False,
    max_new_tokens = 50,
    do_sample = False
)


#model_name = "microsoft/xtremedistil-l12-h384-uncased"  # Find popular HuggingFace models here: https://huggingface.co/models
input_text = "The cat sat on the mat"  
#model = AutoModel.from_pretrained(model_name, output_attentions=True)  # Configure model to return attention values
#tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer.encode(input_text, return_tensors='pt')  # Tokenize input text

input_ids = tokenizer(input_text, return_tensors = "pt").input_ids

# tokenize the input prompt
input_ids = input_ids.to("cuda:0")
pdb.set_trace()
model_output = model.model(input_ids)


# outputs = model(inputs)  # Run model
# attention = outputs[-1]  # Retrieve attention from model outputs
# tokens = tokenizer.convert_ids_to_tokens(inputs[0])  # Convert input ids to token strings
# model_view(attention, tokens)  # Display model view