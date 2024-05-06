import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms
import cv2
import glob
import os

# TODO: check if normalization with mean and std is needed
transform = transforms.Compose([
    transforms.ToPILImage(), # -> PIL image
    transforms.Resize((512, 512)), # -> resize to 512x512
    transforms.RandomCrop((256,256)), # random crop to 256x256
    transforms.ToTensor()])

# write an infinite batch sampler
class InfiniteSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        while True:
            yield from torch.randperm(len(self.data_source)).tolist()

    def __len__(self):
        return 2**31

class coco_train_dataset(Dataset):
    # initialize the dataset
    def __init__(self, project_absolute_path, coco_dataset_relative_path = "datasets/coco_train_dataset/train2017"):
        # get the absolute path of the dataset
        dataset_absolute_path = os.path.join(project_absolute_path, coco_dataset_relative_path)


        # check if the dataset exists
        if not os.path.exists(dataset_absolute_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_absolute_path}")

        # load coco dataset paths from local directory
        self.coco_dataset_images_paths = sorted(glob.glob(os.path.join(dataset_absolute_path, "*.jpg")))

        # check if the dataset is empty
        if len(self.coco_dataset_images_paths) == 0:
            raise FileNotFoundError(f"No images found in the dataset at {dataset_absolute_path}\n\n!!!!!\nPlease download the dataset from http://images.cocodataset.org/zips/train2017.zip and extract it to the datasets directory.\n!!!!!\n")

    # return the length of the dataset
    def __len__(self):
        return len(self.coco_dataset_images_paths)

    # return the image at the given index
    def __getitem__(self, id):
        # load image
        img = cv2.imread(self.coco_dataset_images_paths[id])

        # convert the image to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # apply transformations
        img = transform(img)

        return img
    
class wikiart_dataset(Dataset):
    # initialize the dataset
    def __init__(self, project_absolute_path, wikiart_dataset_relative_path = "datasets/wikiart/**", wikiart_dataset_relative_path2 = "datasets/wikiart"):
        # get the absolute path of the dataset
        dataset_absolute_path2 = os.path.abspath(os.path.join(project_absolute_path, wikiart_dataset_relative_path2))
        dataset_absolute_path = os.path.abspath(os.path.join(project_absolute_path, wikiart_dataset_relative_path))
        # check if the dataset exists
        if not os.path.exists(dataset_absolute_path2):
            raise FileNotFoundError(f"Dataset not found at {dataset_absolute_path}\n\n!!!!!\nPlease download the dataset from https://drive.google.com/file/d/1vTChp3nU5GQeLkPwotrybpUGUXj12BTK/view and extract it to the datasets directory.\n!!!!!\n")
        
        # load wikiart dataset paths from local directory
        self.wikiart_dataset_images_paths = sorted(glob.glob(os.path.join(dataset_absolute_path, "*.jpg")))

        # check if the dataset is empty
        if len(self.wikiart_dataset_images_paths) == 0:
            raise FileNotFoundError(f"No images found in the dataset at {dataset_absolute_path}\n\n!!!!!\nPlease download the dataset from https://drive.google.com/file/d/1vTChp3nU5GQeLkPwotrybpUGUXj12BTK/view and extract it to the datasets directory.\n!!!!!\n")

    # return the length of the dataset
    def __len__(self):
        return len(self.wikiart_dataset_images_paths)

    # return the image at the given index
    def __getitem__(self, id):
        # load image
        img = cv2.imread(self.wikiart_dataset_images_paths[id])

        # convert the image to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # apply transformations
        img = transform(img)

        return img
