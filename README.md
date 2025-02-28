# xAI
This code will provide examples for several methodologies to explain Tranformer style architectures for Vision and NLP datasets. We will check some basic methodologies to ivestigate the inner workings of modern AI technique such as posthoc interpretability, visualization of attention maps, probing and finally mech-interp for text.

We will provide code and also some analysis and taxonomy in xAI.

1. Explain the produced output based on the input
2. Explain the produced output based on the training data
3. Explain the role of individual neurons in embedding features
4. Extract explainable features from poly-semantic neurons

We will start this tutorial with attention based method and vizualitions using Attention of a specific model. We will make use of bertviz a tool for visualizing the attention layers of the model.

The basic files to be examined 'feature attribution.py' 'probing.py', 'visualizing_attention.py'

The 'visualizing_attention' file loads 'Phi-3-mini-4k-instruct' or 'Phi-3-medium-4k-instruct' and the corresponding tokenizers and extracts the attention layer before the final MLP and plots the matrix that shows how attention is distributed to the input tokens. Note, that this can be done for different layers and different heads so we extract all these in the img folder.

The 'probing' file loads a decoding 'gpt2' model and the corresponding tokenizer!