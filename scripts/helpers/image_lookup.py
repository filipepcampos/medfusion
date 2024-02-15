from collections import OrderedDict
from tqdm import tqdm

import torch
from torchvision import transforms as T
from torchmetrics.image import StructuralSimilarityIndexMeasure
from medical_diffusion.data.datasets import MIMIC_CXR_Dataset

from PIL import Image

ds = MIMIC_CXR_Dataset(
    image_resize=256,
    augment_horizontal_flip=False,
    augment_vertical_flip=False,
    path_root = '/nas-ctm01/datasets/public/MEDICAL/MIMIC-CXR',
    split_path = '/nas-ctm01/homes/fpcampos/dev/diffusion/medfusion/data/mimic-cxr-2.0.0-split.csv'
)

transform = T.Compose([
    T.Resize(256),
    T.ToTensor(),
    T.Normalize(mean=0.5, std=0.5) # WARNING: mean and std are not the target values but rather the values to subtract and divide by: [0, 1] -> [0-0.5, 1-0.5]/0.5 -> [-1, 1]
])

sample_img = transform(Image.open("./results/MIMIC-CXR-JPG/samples/test_None_0.png").convert('RGB'))


ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
largest_ssim = 0
closest_img = None
for img in tqdm(ds):
    ssim_value = ssim(sample_img.unsqueeze(0), img["source"].unsqueeze(0))
    if ssim_value > largest_ssim:
        largest_ssim = largest_ssim
        closest_img = img

print(closest_img)