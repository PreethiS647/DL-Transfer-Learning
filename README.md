# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## DESIGN STEPS
### STEP 1: 

Import required libraries and define image transforms.

### STEP 2: 

Load training and testing datasets using ImageFolder.


### STEP 3: 

Visualize sample images from the dataset.

### STEP 4: 

Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.

### STEP 5: 

Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.

### STEP 6: 

Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.



## PROGRAM

### Name: Preethi S

### Register Number: 212223230157

```python
import torch as t
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets,models
from torchvision.models import VGG19_Weights
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchsummary import summary

transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])

!unzip -qq ./chip_data.zip -d data
dataset_path='./data/dataset'
train_dataset=datasets.ImageFolder(root=f"{dataset_path}/train",transform=transform)
test_dataset=datasets.ImageFolder(root=f"{dataset_path}/test",transform=transform)

def show_sample_images(dataset,num_images=5):
  fig,axes=plt.subplots(1,num_images,figsize=(5,5))
  for i in range(num_images):
    image,label=dataset[i]
    image=image.permute(1,2,0)
    axes[i].imshow(image)
    axes[i].set_title(dataset.classes[label])
    axes[i].axis("off")
  plt.show()
# Show sample images from the training dataset
show_sample_images(train_dataset)

```
<img width="466" height="123" alt="image" src="https://github.com/user-attachments/assets/0b280c3a-b6a1-4940-bfc7-016a2e93938e" />
```
# Get the total number of samples in the training dataset
print(f"Total number of training samples: {len(train_dataset)}")

# Get the shape of the first image in the dataset
first_image, label = train_dataset[0]
print(f"Shape of the first image: {first_image.shape}")
```
<img width="525" height="44" alt="image" src="https://github.com/user-attachments/assets/9a06d887-b663-4328-bc3f-09be28546d60" />

```
# Get the total number of samples in the testing dataset
print("Number of testing samples:",len(test_dataset))

# Get the shape of the first image in the dataset
first_image1,label=test_dataset[0]
print("Image shape:",first_image1.shape)

```
<img width="400" height="52" alt="image" src="https://github.com/user-attachments/assets/b9c4b41b-0878-49dc-9eaf-7212adee4401" />

```
# Create DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```
```
## Step 2: Load Pretrained Model and Modify for Transfer Learning
# Load a pre-trained VGG19 model
from torchvision.models import VGG19_Weights
model = models.vgg19(weights=VGG19_Weights.DEFAULT)
```
```
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
from torchsummary import summary
```
```
# Print model summary
summary(model, input_size=(3, 224, 224))
```
<img width="523" height="333" alt="image" src="https://github.com/user-attachments/assets/ee306a8d-3499-427f-8236-95f8aa415355" />


## RESULT

VGG19 model was fine-tuned and tested successfully. The model achieved good accuracy with correct predictions on sample test images.
