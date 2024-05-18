import torch
import torch.nn as nn
import numpy as np
from medical_diffusion.models.privacy.retrieval_model import get_retrieval_model

# Custom loss function
class PrivacyLoss(nn.Module):
    def __init__(self, reduction="none", margin=5, weight=10, **kwargs):
        super(PrivacyLoss, self).__init__()

        self.margin = margin
        self.weight = weight

        #lookup_table = torch.load("/nas-ctm01/homes/fpcampos/dev/reidentification/anonymize/outputs/lookup_table.pth") # TODO: Remove hard-coded path
        #self.training_identity_embeddings = torch.from_numpy(np.stack(list(lookup_table.values()), axis=0)).to("cuda")

        self.retrieval_model = get_retrieval_model("/nas-ctm01/homes/fpcampos/dev/reidentification/anonymize/models/retrieval_model.pth") # TODO: Remove hard-coded path
        self.retrieval_model.eval()

        self.l1 = nn.L1Loss(reduction=reduction)
        self.pairwise_distance = nn.PairwiseDistance(p=2)

        

    # def forward(self, x, y):

    #     embeddings = self.retrieval_model(x)

    #     # Calculate distance between embeddings and training identity embeddings
    #     distances = torch.cdist(embeddings, self.training_identity_embeddings, p=2)

    #     # Get minimum distance for each embedding
    #     min_distances = torch.min(distances, dim=1).values.to(x.device)

    #     # Get min between tensor and 0
    #     zeros = torch.zeros(min_distances.shape).to(x.device)

    #     # Calculate loss
    #     loss = self.mse(x, y) + torch.minimum(50 - min_distances, zeros) * 5 # TODO: Weight

    #     return loss
    
    def forward(self, x, y):
        synthetic_image_embeddings = self.retrieval_model(x)
        real_image_embeddings = self.retrieval_model(y)

        # Calculate distance between embeddings and training identity embeddings
        distances = self.pairwise_distance(synthetic_image_embeddings, real_image_embeddings)
        mean_distance = distances.mean()

        zeros = torch.zeros(mean_distance.shape).to(x.device)

        loss = self.l1(x, y) + torch.maximum(self.margin - mean_distance, zeros) * self.weight

        return loss
