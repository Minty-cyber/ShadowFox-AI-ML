from detecto import core, utils
from detecto.visualize import show_labeled_image
from torchvision import transforms
import numpy as np



path_image = 'images'
path_train_labels ='train_labels'
path_test_labels ='test_labels'


custom_transforms = transforms.Compose([
    transforms.ToPILImage(),#Converts it to pillow image
    transforms.Resize(50), #Resizes the images to a smaller size
    transforms.RandomHorizontalFlip(), #Improves the nborder performance
    transforms.RandomRotation(165),
    transforms.ToTensor
    
])