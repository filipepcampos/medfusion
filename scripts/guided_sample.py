
from pathlib import Path
import torch 
from torchvision import utils 
import math 
from medical_diffusion.models.pipelines.diffusion_pipeline_debug import DiffusionPipeline
from medical_diffusion.data.datasets import MIMIC_CXR_Dataset
from medical_diffusion.data.datamodules import SimpleDataModule
from medical_diffusion.external.retrieval_model import get_retrieval_model

def rgb2gray(img):
    # img [B, C, H, W]
    return  ((0.3 * img[:,0]) + (0.59 * img[:,1]) + (0.11 * img[:,2]))[:, None]
    # return  ((0.33 * img[:,0]) + (0.33 * img[:,1]) + (0.33 * img[:,2]))[:, None]

def normalize(img):
    # img =  torch.stack([b.clamp(torch.quantile(b, 0.001), torch.quantile(b, 0.999)) for b in img])
    return torch.stack([(b-b.min())/(b.max()-b.min()) for b in img])




if __name__ == "__main__":
    path_out = Path.cwd()/'results/MIMIC-CXR-JPG/samples_11'
    path_out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    device = torch.device('cuda')

    # ------------ Load Model ------------
    # pipeline = DiffusionPipeline.load_best_checkpoint(path_run_dir)
    #pipeline = DiffusionPipeline.load_from_checkpoint('runs/2024_02_13_090955/last.ckpt')
    pipeline = DiffusionPipeline.load_from_checkpoint('last_identity.ckpt')
    pipeline.to(device)

    ds = MIMIC_CXR_Dataset(
        image_resize=256,
        augment_horizontal_flip=False,
        augment_vertical_flip=False,
        path_root = '/nas-ctm01/datasets/public/MEDICAL/MIMIC-CXR',
        split_path = '/nas-ctm01/homes/fpcampos/dev/diffusion/medfusion/data/mimic-cxr-2.0.0-split.csv'
    )

    dm = SimpleDataModule(
        ds_train = ds,
        batch_size=8, 
        # num_workers=0,
        pin_memory=True,
        # weights=ds.get_weights()
    ) 
    
    
    # --------- Generate Samples  -------------------
    steps = 250
    use_ddim = True 
    images = {}
    n_samples = 8

    retrieval_model = get_retrieval_model("/nas-ctm01/homes/fpcampos/dev/reidentification/anonymize/models/retrieval_model.pth") # TODO: Remove hard-coded path
    retrieval_model.to(device)

    k = next(iter(dm.train_dataloader()))
    x, y = k["source"].to(device), k["target"]

    guidance_scales = [-8, -2,  0, 2, 8]

    # Save original images
    utils.save_image(x, path_out/f'original.png', nrow=1, normalize=True, scale_each=True) # For 2D images: [B, C, H, W]

    # for cond in [0,1]:
    #     save_dict = {}
    #     for w in guidance_scales:
    #         torch.manual_seed(42)
    
    #         # --------- Conditioning ---------
    #         condition = torch.tensor([cond]*n_samples, device=device) if cond is not None else None 
    #         # un_cond = torch.tensor([1-cond]*n_samples, device=device)
    #         un_cond = None 

    #         x_identity = retrieval_model(x).cuda()

    #         # ----------- Run --------
    #         results = pipeline.sample(n_samples, (8, 32, 32), guidance_scale=8, condition=condition, un_cond=un_cond, steps=steps, use_ddim=use_ddim, identity_embedding=x_identity, identity_guidance_scale=w)
    #         # results = pipeline.sample(n_samples, (4, 64, 64), guidance_scale=1, condition=condition, un_cond=un_cond, steps=steps, use_ddim=use_ddim )

    #         # --------- Save result ---------------
    #         results = (results+1)/2  # Transform from [-1, 1] to [0, 1]
    #         results = results.clamp(0, 1)

    #         save_dict[w] = results

    #         #utils.save_image(results, path_out/f'test_{cond}_{w}.png', nrow=int(math.sqrt(results.shape[0])), normalize=True, scale_each=True) # For 2D images: [B, C, H, W]

    #         images[cond] = results
    #     # show images from different w side by side
    #     utils.save_image(torch.cat([save_dict[w] for w in guidance_scales], dim=0), path_out/f'test_{cond}.png', nrow=len(guidance_scales), normalize=True, scale_each=True)


    # diff = torch.abs(normalize(rgb2gray(images[1]))-normalize(rgb2gray(images[0]))) # [0,1] -> [0, 1]
    # # diff = torch.abs(images[1]-images[0])
    # utils.save_image(diff, path_out/'diff.png', nrow=int(math.sqrt(results.shape[0])), normalize=True, scale_each=True) # For 2D images: [B, C, H, W]
    
  
    save_dict = {}
    for w in guidance_scales:
        torch.manual_seed(42)

        # --------- Conditioning ---------
        condition = y.to(device)
        un_cond = None 

        x_identity = k["target2"].to(device)

        # ----------- Run --------
        results = pipeline.sample(n_samples, (8, 32, 32), guidance_scale=8, condition=condition, un_cond=un_cond, steps=steps, use_ddim=use_ddim, identity_embedding=x_identity, identity_guidance_scale=w)

        # --------- Save result ---------------
        results = (results+1)/2  # Transform from [-1, 1] to [0, 1]
        results = results.clamp(0, 1)

        save_dict[w] = results

    # show images from different w side by side
    utils.save_image(torch.cat([save_dict[w] for w in guidance_scales], dim=0), path_out/f'test.png', nrow=len(guidance_scales), normalize=True, scale_each=True)
