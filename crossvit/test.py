import torch
from torch.utils.data import DataLoader
from transformers import AdamW, ViTImageProcessor, ViTForImageClassification
from NWRD_dataset import NWRD
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import os
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
CUDA_LAUNCH_BLOCKING=1
TORCH_USE_CUDA_DSA=1
print("device is:", device)

transformations = transforms.Compose([
    transforms.ToTensor()            
])

test_ds = NWRD(root_dir="/scratch/wej36how/Datasets/NWRDProcessed/test/calssification", train=False, transform=transformations)


test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
model.classifier = torch.nn.Linear(768,2)
model.to(device)


model_weights = torch.load('./22.pth') 
model.load_state_dict(model_weights.state_dict())

criterion = torch.nn.CrossEntropyLoss()

#Testing

true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

rust_dir = "./results/nwrd22/rust"
fn_dir = "./results/nwrd22/fn"
fp_dir = "./results/nwrd22/fp"

img_paths = test_ds.images
count=0
#softmax = nn.SoftMax()
model.eval
loop = tqdm(enumerate(test_loader), total=len(test_loader))
with torch.no_grad():
    for batch_idx, (images, labels) in loop:
        inputs = processor(images=images, return_tensors="pt", do_rescale=False).to(device)
        #print(inputs)
        labels = labels.to(device)

        outputs = model(**inputs)
        logits = outputs.logits
        #logits = F.softmax(logits)
        prediction = logits.argmax(axis=1)
        # print(logits)
        # print(labels)
        # print(prediction)
        if (prediction==1):
            image_path = os.path.join(rust_dir, img_paths[count].split('/')[-1])
            image = images.squeeze().cpu()
            save_image(image,image_path)

        if ((prediction==1) and (labels==0)):
            image_path = os.path.join(fp_dir, img_paths[count].split('/')[-1])
            image = images.squeeze().cpu()
            save_image(image,image_path)

        if ((prediction==0) and (labels==1)):
            image_path = os.path.join(fn_dir, img_paths[count].split('/')[-1])
            image = images.squeeze().cpu()
            save_image(image,image_path)

        true_positives += torch.sum((prediction == 1) & (labels == 1)).item()
        false_positives += torch.sum((prediction == 1) & (labels == 0)).item()
        true_negatives += torch.sum((prediction == 0) & (labels == 0)).item()
        false_negatives += torch.sum((prediction == 0) & (labels == 1)).item()
        count+=1

# Calculate metrics
precision = true_positives / (true_positives + false_positives + 1e-10)
recall = true_positives / (true_positives + false_negatives + 1e-10)
F1 = 2 * (precision * recall) / (precision + recall)
accuracy = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives + 1e-10)

# Print or use the metrics as needed
print(f"True Positives: {true_positives}")
print(f"False Positives: {false_positives}")
print(f"True Negatives: {true_negatives}")
print(f"False Negatives: {false_negatives}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {F1}")
print(f"Accuracy: {accuracy}")