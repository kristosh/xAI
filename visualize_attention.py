# In this code file wie will check multiple attention visualizations for couple of pre-trained huggingface models
import os
os.environ['HF_HOME'] = "/local/athanasiadisc/cache"

from transformers import AutoTokenizer, AutoModel, utils
from bertviz import model_view
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pdb
import matplotlib.pyplot as plt
utils.logging.set_verbosity_error()  # Suppress standard warnings

# Find popular HuggingFace models here: https://huggingface.co/models
# model that will be used in this tutorial
model_name = "microsoft/xtremedistil-l12-h384-uncased" 
# microsoft/xtremedistil-l12-h384-uncased
#microsoft/microsoft/Phi-3-medium-4k-instruct

# tokenizer that we will make use in this tutorial 
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save attention visualization code 
def save_attention_image(attention, tokens, filename='attention.png'):
    """
    Save the attention weights for a specific layer and head as an image.
    
    :param attention: The attention weights from the model.
    :param tokens: The tokens corresponding to the input.
    :param layer_num: The layer number to visualize.
    :param head_num: The head number to visualize.
    :param filename: The filename to save the image.
    """

    attn = attention[0].detach().cpu().float().numpy()    
    num_heads = attn.shape[0]
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))  # Adjust the grid size as needed
    
    for i, ax in enumerate(axes.flat):
        if i < num_heads:
            cax = ax.matshow(attn[i], cmap='viridis')
            ax.set_title(f'Head {i + 1}')
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=90)
            ax.set_yticklabels(tokens)
        else:
            ax.axis('off')
    
    fig.colorbar(cax, ax=axes.ravel().tolist())
    plt.suptitle(f'Layer {1}')
    plt.savefig(filename)
    plt.close()


def main():

    # loading the model from hugging face
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map = 'auto',
        torch_dtype = "auto",
        trust_remote_code = True,
        output_attentions=True
    )

    # Create a pipeline
    generator = pipeline(
        "text-generation",
        model = model,
        tokenizer = tokenizer,
        return_full_text= False,
        max_new_tokens = 100,
        do_sample = False
    )

    input_text = "What is the co-capital of Greece according to citizens opinions: "  
    inputs = tokenizer.encode(input_text, return_tensors='pt')  # Tokenize input text
    input_ids = tokenizer(input_text, return_tensors = "pt").input_ids

    # tokenize the input prompt
    input_ids = input_ids.to("cuda")

    # calculate the output using the LLM model
    model_output = model(input_ids)
    # extract the attention layer
    attention = model_output[-1] 
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0]) 

    # # view the attention layers
    # model_view(attention, tokens)  # Display model view

    counter = 0
    for head in attention:
        file_name = "img/attention_head%d.png"
        file_name = file_name % (counter+1)
        save_attention_image(head, tokens, filename= file_name)
        counter = counter + 1

    # The prompt (user input / query)
    messages = "What is the co-capital of greece according to citizens' opinions?"

    # Generate output
    output = generator(messages)
    print(output[0]["generated_text"])
    pdb.set_trace()

if __name__ == "__main__":
    main()