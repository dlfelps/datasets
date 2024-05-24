import os
import os.path
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union
import re

from PIL import Image

from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import v2
import torch



class CUB200(VisionDataset):
    """`CUB-200-2011 <https://www.vision.caltech.edu/datasets/cub_200_2011/>`_ Dataset.


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

    cub = CUB200('.', download=True, is_test=False, transform = v2.Compose([
      v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
      v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
      v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

            .. warning::

                To download the dataset `gdown <https://github.com/wkentaro/gdown>`_ is required.
    """

    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_test: bool = False,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")


        self.root = Path(root).joinpath('CUB_200_2011')
        self.is_test=is_test
        self.transform = transform
        self.target_transform = target_transform
        self.ids = None
        self.files = None
        self.setup()

    def setup(self):
      ids = self.get_ids()
      id_image_path = self.root.joinpath('images.txt')
      with open(id_image_path, 'r') as f:
        lines = f.readlines()

      parser = image_parser_closure()
      pairs = list(map(parser, lines))

      self.ids = [k for k,_ in pairs if k in ids]
      self.files = [v for k,v in pairs if k in ids]


    def get_ids(self):

      test_train_split_path = self.root.joinpath('train_test_split.txt')
      with open(test_train_split_path, 'r') as f:
        lines = f.readlines()

      parser = split_parser_closure()
      pairs = map(parser, lines)
      training = filter(lambda x: x[1]==self.is_test, pairs)

      return list(map(lambda x: x[0], training))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(self.files[idx])
        target = self.ids[idx]

        if self.transform is not None:
          img = self.transform(img)

        if self.target_transform is not None:
          target = self.target_transform(target)

        return img, target


    def _check_integrity(self) -> bool:
        # can be more robust and check hash of files
        return os.path.exists(os.path.join(self.root, "CUB_200_2011"))


    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_and_extract_archive(
            "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1",
            self.root,
            filename="CUB_200_2011.tgz",
            md5='97eceeb196236b17998738112f37df78'
        )


    def extra_repr(self) -> str:
        return "Target type: {target_type}".format(**self.__dict__)
    

# parsing functions
def image_parser_closure():
  parse_attr = re.compile(r'(\d+) (.+)')
  p = Path('./CUB_200_2011/images')

  def match(input):
    result = parse_attr.match(input)
    return (int(result.group(1)), str(p.joinpath(result.group(2))))

  return match

def split_parser_closure():
  parse_two_numbers = re.compile(r'(\d+) (\d+)')

  def match(input):
    result = parse_two_numbers.match(input)
    return (int(result.group(1)), bool(int(result.group(2))))

  return match