from detecto import core, utils
from detecto.visualize import show_labeled_image
from torchvision import transforms
import numpy as np



path_image = 'images'
path_train_labels ='train_labels'
path_test_labels ='test_labels'


custom_transforms = transforms.Compose([
    transforms.ToPILImage()
])