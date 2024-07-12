from medical_diffusion.data.datasets import AIROGSDataset


from pathlib import Path


path_out = Path("/mnt/hdd/datasets/eye/AIROGS/data_256x256/")
path_out.mkdir(parents=True, exist_ok=True)

ds = AIROGSDataset(
    crawler_ext="jpg",
    image_resize=256,
    image_crop=(256, 256),
    path_root="/mnt/hdd/datasets/eye/AIROGS/data/",  # '/home/gustav/Documents/datasets/AIROGS/dataset', '/mnt/hdd/datasets/eye/AIROGS/data/'
)

weights = ds.get_weights()

for img in ds:
    img["source"].save(path_out / f"{img['uid']}.jpg")
