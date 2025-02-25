# xAI
This code will provide examples for several methodologies to explain Tranformer style architectures for Vision and NLP datasets. We will check some basic methodologies to ivestigate the inner workings of modern AI technique such as posthoc interpretability, visualization of attention maps, probing and finally mech-interp for text.

We will provide code and also some analysis and taxonomy in xAI.

1. Explain the produced output based on the input
2. Explain the produced output based on the training data
3. Explain the role of individual neurons in embedding features
4. Extract explainable features from poly-semantic neurons

We will start this tutorial with attention based method and vizualitions using Attention of a specific model. We will make use of bertviz a tool for visualizing the attention layers of the model.

The basic files to be examined 'feature attribution.py' 'probing.py', 'visualizing_attention.py'