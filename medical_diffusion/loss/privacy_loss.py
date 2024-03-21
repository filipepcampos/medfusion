import torch
import torch.nn as nn
import numpy as np
from medical_diffusion.models.privacy.retrieval_model import get_retrieval_model

# Custom loss function
class PrivacyLoss(nn.Module):
    def __init__(self, reduction=None, **kwargs):
        super(PrivacyLoss, self).__init__()

        lookup_table = torch.load("/nas-ctm01/homes/fpcampos/dev/reidentification/anonymize/outputs/lookup_table.pth") # TODO: Remove hard-coded path
        self.training_identity_embeddings = torch.from_numpy(np.stack(list(lookup_table.values()), axis=0)).to("cuda")

        self.retrieval_model = get_retrieval_model("/nas-ctm01/homes/fpcampos/dev/reidentification/anonymize/models/retrieval_model.pth") # TODO: Remove hard-coded path
        self.retrieval_model.eval()

        self.mse = nn.MSELoss()

    def forward(self, x, y):

        embeddings = self.retrieval_model(x)

        # Calculate distance between embeddings and training identity embeddings
        distances = torch.cdist(embeddings, self.training_identity_embeddings, p=2)

        # Get minimum distance for each embedding
        min_distances = torch.min(distances, dim=1).values.to(x.device)

        # Get min between tensor and 0
        zeros = torch.zeros(min_distances.shape).to(x.device)

        # Calculate loss
        loss = self.mse(x, y) + torch.minimum(50 - min_distances, zeros) * 5 # TODO: Weight

        return loss
