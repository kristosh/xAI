import os
os.environ['HF_HOME'] = "/local/athanasiadisc/cache"
# list available attribution methods
import inseq

inseq.list_feature_attribution_methods()

# load a model from HuggingFace model hub and define the feature attribution 
# method you want to use
mdl_gpt2 = inseq.load_model("gpt2", "integrated_gradients")

# compute the attributions for a given prompt
attr = mdl_gpt2.attribute(
    "Hello ladies and",
    generation_args={"max_new_tokens": 9},
    n_steps=500,
    internal_batch_size=50 )

# display the generated attributions
attr.show()

import pdb
pdb.set_trace()