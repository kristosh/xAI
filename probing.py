import os
os.environ['HF_HOME'] = "/local/athanasiadisc/cache"
from datasets import load_dataset
from transformers import ImageGPTFeatureExtractor, ImageGPTModel
import torch
from PIL import Image
import requests
import numpy as np

from sklearn.linear_model import LogisticRegression
from tqdm.notebook import tqdm
import pdb

def extract_features(examples):
  # take a batch of images
  images = examples['img']
  # convert to list of NumPy arrays of shape (C, H, W)
  images = [np.array(image, dtype=np.uint8) for image in images]
  images = [np.moveaxis(image, source=-1, destination=0) for image in images]
  # tokenize images
  encoding = feature_extractor(images=images, return_tensors="pt")
  pixel_values = encoding.pixel_values.to(device)
  # forward through model to get hidden states
  with torch.no_grad():
    outputs = model(pixel_values, output_hidden_states=True)
  hidden_states = outputs.hidden_states
  # add features of each layer
  for i in range(len(hidden_states)):
      features = torch.mean(hidden_states[i], dim=1)
      examples[f'features_{i}'] = features.cpu().detach().numpy()
  
  return examples

def main():
    #load cifar10 (only small portion for demonstration purposes) 
    train_ds, test_ds = load_dataset('cifar10', split=['train[:10]', 'test[:10]'])
    # split up training into training + validation
    splits = train_ds.train_test_split(test_size=0.1)
    train_ds = splits['train']
    val_ds = splits['test']

    dataset = load_dataset('cifar10')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_extractor = ImageGPTFeatureExtractor.from_pretrained("openai/imagegpt-small")
    model = ImageGPTModel.from_pretrained("openai/imagegpt-small")
    model.to(device)

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)

    encoding = feature_extractor(image, return_tensors="pt")
    pixel_values = encoding.to(device)

    # forward pass
    outputs = model(pixel_values, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    print(len(hidden_states))


    feature_vector = torch.mean(hidden_states[13], dim=1)
    feature_vector.shape

    encoded_dataset = dataset.map(extract_features, batched=True, batch_size=2)
    encoded_dataset = encoded_dataset.with_format("numpy")
    encoded_dataset.save_to_disk("/content/drive/MyDrive/ImageGPT")
    encoded_dataset['train']

    encoded_dataset['train']['features_0'][0].shape
    encoded_dataset['train']['label']

    train_dataset = encoded_dataset['train']
    test_dataset = encoded_dataset['test']

    scores = dict()
    for i in tqdm(range(model.config.n_layer + 1)):
        # fit linear classifier
        lr_clf = LogisticRegression(max_iter=1000)
        lr_clf.fit(train_dataset[f'features_{i}'], train_dataset['label'])
        # compute accuracy on training + test set
        training_score = lr_clf.score(train_dataset[f'features_{i}'], train_dataset['label'])
        test_score = lr_clf.score(test_dataset[f'features_{i}'], test_dataset['label'])
        scores[f'features_{i}'] = (training_score, test_score)


if __name__ == "__main__":
    main()

