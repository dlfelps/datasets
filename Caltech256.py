import torchvision
from torchvision.transforms import v2
import torch
import os
from pathlib import Path
from PIL import Image
import re

class CaltechNoBirds(torchvision.datasets.Caltech256):
  """`Caltech-256 <https://data.caltech.edu/records/nyy15-4j048>`_ Dataset.


  Args:
      root (str or ``pathlib.Path``): Root directory of dataset where directory
          ``caltech101`` exists or will be saved to if download is set to True.
      target_type (string or list, optional): Type of target to use, ``category`` or
          ``annotation``. Can also be a list to output a tuple with all specified
          target types.  ``category`` represents the target class, and
          ``annotation`` is a list of points from a hand-generated outline.
          Defaults to ``category``.
      transform (callable, optional): A function/transform that takes in a PIL image
          and returns a transformed version. E.g, ``transforms.RandomCrop``
      target_transform (callable, optional): A function/transform that takes in the
          target and transforms it.
      download (bool, optional): If true, downloads the dataset from the internet and
          puts it in root directory. If dataset is already downloaded, it is not
          downloaded again.


  ct256 = CaltechNoBirds('.', download=True, transform = v2.Compose([
    v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
    v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]), 
    target_transform=lambda x: 0)

          .. warning::

              To download the dataset `gdown <https://github.com/wkentaro/gdown>`_ is required.
  """

  def __init__(
        self,
        root: str,
        transform = None,
        target_transform= None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform, download=download)
        self.blacklist = [257, 114, 101, 113, 49, 60, 89, 118, 151, 152,158, 207]
        self.files = self.__getfiles__()
        self.class_matcher = re.compile(r'(\d+)_.+')


  def __getfiles__(self):
    # returns a list of non-blacklisted files

    valid_files = []
    for folder in Path(self.root).joinpath('256_ObjectCategories').iterdir():
      if int(folder.stem) not in self.blacklist:
        valid_files.extend(list(folder.glob('*.jpg')))

    return valid_files
  
  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    img = Image.open(self.files[idx])
    result = self.class_matcher.match(self.files[idx].stem)
    target = int(result.groups(1)[0])

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target
