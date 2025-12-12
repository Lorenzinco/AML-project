from typing import Literal
from tqdm.auto import tqdm
import requests
import zipfile
import os
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, PILToTensor, Resize, RandomCrop
from aml_project.preprocess import Processor

from config import Config

COCO_PATH = os.getenv("COCO_PATH", "~/Downloads/coco")
COCO_PATH = os.path.expanduser(COCO_PATH)
if not os.path.exists(COCO_PATH):
    os.makedirs(COCO_PATH)

def download_coco():
    files_to_download = {
        "unlabeled2017.zip": "http://images.cocodataset.org/zips/unlabeled2017.zip",
        "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
        "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
        "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    }

    coco_finished_download_marker = os.path.join(COCO_PATH, ".finished")
    if os.path.exists(coco_finished_download_marker):
        return

    for filename, url in files_to_download.items():
        print(f"Downloading {filename} from {url}")
        filepath = os.path.join(COCO_PATH, filename)
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        if os.path.exists(filepath):
            if os.path.getsize(filepath) == total_size:
                print(f"{filename} already exists and is fully downloaded. Skipping.")
                continue
        block_size = 1024  # 1 Kibibyte
        with open(filepath, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                file.write(data)
                bar.update(len(data))
    # Unzip files
    for filename in files_to_download.keys():
        filepath = os.path.join(COCO_PATH, filename)
        print(f"Unzipping {filename}")
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            for name in zip_ref.namelist():
                if not os.path.exists(os.path.join(COCO_PATH, name)):
                    zip_ref.extract(name, COCO_PATH)
    # Create finished marker
    with open(coco_finished_download_marker, 'w') as f:
        f.write("COCO dataset download completed.")


class RandomSizeCrop:
    def __init__(self, min_size):
        self.min_size = min_size
    
    def __call__(self, img):
        max_size = min(img.shape[1:3])
        min_size = min(self.min_size, max_size)
        side = torch.randint(min_size, max_size+1, ())
        return RandomCrop((side, side))(img)

class ImageOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, config: Config, split: Literal["train", "validation"] = "train"):
        # download_coco() # not needed when using kaggle
        unlabeled_dir = os.path.join(COCO_PATH, "unlabeled2017")
        labeled_dir = os.path.join(COCO_PATH, "train2017")
        validation_dir = os.path.join(COCO_PATH, "val2017")
        self.image_paths = []


        match split:
            case "train":
                self.preprocess = Compose((PILToTensor(), RandomSizeCrop(128), Resize(config.resolution)))
                for root, _, files in os.walk(unlabeled_dir):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.image_paths.append(os.path.join(root, file))
                for root, _, files in os.walk(labeled_dir):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.image_paths.append(os.path.join(root, file))
            case "validation":
                self.preprocess = Compose((PILToTensor(), Resize(config.resolution))) # avoid randomness during validation
                for root, _, files in os.walk(validation_dir):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.image_paths.append(os.path.join(root, file))
            case _:
                raise ValueError("split must be either 'train' or 'validation'")
            
    def load_image(self, path):
        with Image.open(path) as im:
            im = im.convert("RGB")
            image = self.preprocess(im)   # safe: preprocess reads pixels while file is open
        return image

    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = self.load_image(image_path)
        return image

if __name__ == "__main__":
    config = Config()
    dataset = ImageOnlyDataset(config=config, split="train")

