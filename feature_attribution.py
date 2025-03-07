import os
os.environ['HF_HOME'] = "/local/athanasiadisc/cache"
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

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import shap
from transformers_interpret import SequenceClassificationExplainer
from ferret import Benchmark

from datasets import load_dataset

dataset = load_dataset("imdb")
df = dataset['test'].to_pandas()

short_data = [v[:500] for v in df["text"][:20]]

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
tokenizer = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")
model = AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb")
classifier = pipeline('text-classification', return_all_scores=True, model=model,tokenizer=tokenizer)
classifier(short_data[:1])

# import shap
# explainer = shap.Explainer(classifier)
# shap_values = explainer([short_data[0]])
# shap.plots.text(shap_values)
# shap.plots.bar(shap_values[0,:,"POSITIVE"])

# ## Multiple predictions at once:
# shap_values = explainer(short_data[:2])

# import pdb
# pdb.set_trace()
# shap.plots.text(shap_values[:,:,"POSITIVE"])
# shap.plots.bar(shap_values[:, :, "POSITIVE"].mean(0),max_display=50)

# test = ["I love sci-fi and eat a lot. you?"]
# shap_values = explainer(test)
# shap_values.feature_names[0]

# cluster_matrix = shap_values.clustering
# labels = list(shap_values.feature_names[0])

# from scipy.cluster import hierarchy
# hierarchy.dendrogram(cluster_matrix[0],labels=list(labels))

# shap_values.clustering[0]


# from transformers_interpret import SequenceClassificationExplainer
# cls_explainer = SequenceClassificationExplainer(
#     model,
#     tokenizer)

# word_attributions = cls_explainer(short_data[0])
# cls_explainer.visualize()

# word_attributions

# def Sort_Tuple(tup):
#     # reverse = None (Sorts in Ascending order)
#     # key is set to sort using second element of
#     # sublist lambda has been used
#     tup.sort(key = lambda x: x[1])
#     return tup

# sorted = Sort_Tuple(word_attributions)

# import matplotlib.pyplot as plt
# labels, values = zip(*(sorted[:20] +sorted[-20:]))
# plt.rcParams["figure.figsize"] = (12,10)
# plt.barh(range(len(labels)),values)
# plt.yticks(range(len(values)),labels)
# plt.show()

from ferret import Benchmark
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")

bench = Benchmark(model, tokenizer)
explanations = bench.explain("You look stunning!", target=1)

bench.show_table(explanations)

evaluations = bench.evaluate_explanations(explanations, target=1)
bench.show_evaluation_table(evaluations)


model2 = AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb")
tokenizer2 = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")
bench = Benchmark(model2, tokenizer2)
explanations = bench.explain(short_data[0])

bench.show_table(explanations)

evaluations = bench.evaluate_explanations(explanations, target=1)
bench.show_evaluation_table(evaluations)