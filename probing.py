import os
os.environ['HF_HOME'] = "/local/athanasiadisc/cache"

import pdb
# Standard imports
from transformers import (
    AutoTokenizer,
    MistralForCausalLM,
    Trainer,
    BitsAndBytesConfig,
    TrainingArguments,
    GPT2Model,
    GPT2LMHeadModel,
)
from datasets import load_dataset
from peft import LoraConfig
import torch

# Imports from the transformer_heads library
from transformer_heads import load_headed
from transformer_heads.util.helpers import DataCollatorWithPadding, get_model_params
from transformer_heads.config import HeadConfig
from transformer_heads.util.model import print_trainable_parameters
from transformer_heads.util.evaluate import evaluate_head_wise, get_top_n_preds

device = "cuda:0"

def main():

  # GPT2 is the fastest and requires fewest memory. However, this works just the same with any Llama or Mistral model. Just change model_path to its huggingface path.
  model_path = "gpt2"
  train_epochs = 1
  eval_epochs = 1
  logging_steps = 100

  model_params = get_model_params(model_path)
  model_class = model_params["model_class"]
  hidden_size = model_params["hidden_size"]
  vocab_size = model_params["vocab_size"]
  print(model_params)

  model_params = get_model_params(model_path)
  model_class = model_params["model_class"]
  hidden_size = model_params["hidden_size"]
  vocab_size = model_params["vocab_size"]
  print(model_params)

  heads_configs = [
      HeadConfig(
          name="wikitext_head",
          layer_hook=-4,  # Hook to layer [-4] (Drop 3 layers from the end)
          in_size=hidden_size,
          num_layers=1,
          output_activation="linear",
          is_causal_lm=True,
          loss_fct="cross_entropy",
          num_outputs=vocab_size,
          is_regression=False,
          output_bias=False,
      )
  ]

  dd = load_dataset("wikitext", "wikitext-2-v1")
  #device
  tokenizer = AutoTokenizer.from_pretrained(model_path)

  if tokenizer.pad_token_id is None:
      tokenizer.pad_token = tokenizer.eos_token

  def tokenize_function(examples):
      out = tokenizer(examples["text"], padding=False, truncation=True)
      out[heads_configs[0].name] = out["input_ids"].copy()
      return out.to(device)

  for split in dd.keys():
      dd[split] = dd[split].filter(function=lambda example: len(example["text"]) > 10)
      dd[split] = dd[split].map(tokenize_function, batched=True)

  dd.set_format(type="torch", columns=["input_ids", "attention_mask", heads_configs[0].name])
  
  for split in dd.keys():
      dd[split] = dd[split].remove_columns("text")
  
  quantization_config = BitsAndBytesConfig(
      load_in_4bit=True,
      load_in_8bit=False,
      llm_int8_threshold=6.0,
      llm_int8_has_fp16_weight=False,
      bnb_4bit_compute_dtype=torch.float32,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4",
  )

  model = load_headed(
      model_class,
      model_path,
      head_configs=heads_configs,
      quantization_config=quantization_config,
      device_map=device,
  )
  print_trainable_parameters(model)

 #print(get_top_n_preds(n=5, model=model, text="The historical significance of", tokenizer=tokenizer))
  args = TrainingArguments(
      output_dir="linear_probe_test",
      learning_rate=0.0002,
      num_train_epochs=train_epochs,
      logging_steps=logging_steps,
      do_eval=False,
      remove_unused_columns=False  # Important to set to False, otherwise things will fail
      
  )
  collator = DataCollatorWithPadding(
      feature_name_to_padding_value={
          "input_ids": tokenizer.pad_token_id,
          heads_configs[0].name: -100,
          "attention_mask": 0,
      }
  )
  trainer = Trainer(
      model,
      args=args,
      train_dataset=dd["train"],
      data_collator=collator
  )

  pdb.set_trace()
  trainer.train()

  print(evaluate_head_wise(model, dd["validation"], collator, epochs=eval_epochs))
  print(evaluate_head_wise(model, dd["validation"], collator, epochs=eval_epochs))
    

if __name__ == "__main__":
    main()

