# In this code file wie will check multiple attention visualizations for couple of pre-trained huggingface models
import os
os.environ['HF_HOME'] = "/local/athanasiadisc/cache"

from transformers import AutoTokenizer, AutoModel, utils
from bertviz import model_view
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

import pdb
import matplotlib.pyplot as plt
utils.logging.set_verbosity_error()  # Suppress standard warnings

# Find popular HuggingFace models here: https://huggingface.co/models
# model that will be used in this tutorial
#model_name = "microsoft/xtremedistil-l12-h384-uncased" 
# microsoft/xtremedistil-l12-h384-uncased
#microsoft/microsoft/Phi-3-medium-4k-instruct

# Save attention visualization code 
def save_attentions(attentions, inputs, tokenizer, layer, head, filename):
    """
    Save the attention weights for a specific layer and head as an image.
    
    :param attention: The attention weights from the model.
    :param tokens: The tokens corresponding to the input.
    :param layer_num: The layer number to visualize.
    :param head_num: The head number to visualize.
    :param filename: The filename to save the image.
    """

    attn = attentions.detach().cpu().numpy()  # shape (seq_len, seq_len)

    # Get tokens for labeling the axes
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

    # Visualize the attention matrix using matplotlib
    plt.figure(figsize=(8,8))
    plt.imshow(attn, cmap="viridis")
    plt.colorbar()
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.title(f"Attention Matrix (Layer {layer}, Head {head})")
    plt.savefig(filename)
    plt.close()

# the model and tokenizer are loaded and the attention weights are extracted
def main():

    # tokenizer that we will make use in this tutorial 
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load tokenizer and model (make sure you have a valid license for the model)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-medium-4k-instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-medium-4k-instruct",  # note: check spelling if you get error
        device_map="auto",
        torch_dtype=torch.float16,            # or torch.float32 if preferred
        trust_remote_code=True
    )

    # Prepare a prompt
    prompt = "Whats is the co-capital of Greece according to the country's public opinion?"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to("cuda:0")  # send inputs to cuda

    # Run the model with attention outputs enabled
    # Make sure to pass output_attentions=True
    outputs = model(input_ids=inputs.input_ids, output_attentions=True)

    # outputs.attentions is a tuple with one element per layer
    # Each element is a tensor of shape (batch_size, num_heads, seq_len, seq_len)
    attentions = outputs.attentions

    lyr = 0
    for head in attentions:
        file_name = "img/attention_layer_%d_head_%d.png"
        _hd_= 0

        file_name = file_name % (lyr+1) %(_hd_+1)
        for attn in head[0]:
            # For example, choose layer 0 and head 0 to visualize
            save_attentions(attn, inputs, tokenizer, lyr, _hd_, file_name)
            _hd_ = _hd_ + 1
        lyr = lyr + 1
    

if __name__ == "__main__":
    main()