from detecto import core, utils
from detecto.visualize import show_labeled_image
from torchvision import transforms
import numpy as np



path_images = 'images'
path_train_labels ='train_labels'
path_test_labels ='test_labels'


custom_transforms = transforms.Compose([
    transforms.ToPILImage(),#Converts it to pillow image
    transforms.Resize(50), #Resizes the images to a smaller size
    transforms.RandomHorizontalFlip(), #Improves the nborder performance
    transforms.RandomRotation(165),
    transforms.ToTensor(),
    utils.normalize_transform()
    
])

trained_labels = ['apple', 'banana']

train_dataset = core.Dataset
(image_folder==path_images,
label_data==path_train_labels,
transform==custom_transforms)

test_dataset = core.Dataset
(image_folder==path_images,
label_data==path_test_labels,
transform==custom_transforms)

train_loader = core.DataLoader(train_dataset, batch_size=2, shufffle=False)
                               
