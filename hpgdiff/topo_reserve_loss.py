import torch
import torch.nn as nn
import math


class TopoReserveLoss(nn.Module):


    def __init__(self, image_size=64, decay_constant=5.0):

        super(TopoReserveLoss, self).__init__()
        self.image_size = image_size
        self.decay_constant = decay_constant


        self.num_diffusion_steps = math.ceil(math.log2(image_size))
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def time_weight(self, t, total_timesteps):


        return torch.exp(-self.decay_constant * t.float().to(t.device) / total_timesteps)

    def efficient_diffusion(self, source_mask, structure_pred):


        D_current = source_mask.clone()


        for _ in range(self.num_diffusion_steps):
            P_k = self.pool(D_current)
            D_next = D_current + (P_k * structure_pred)
            D_current = torch.clamp(D_next, 0, 1)

        return D_current

    def compute_topo_loss(self, pred_x0, support_mask, t, total_timesteps):


        structure_pred = (pred_x0 + 1.0) / 2.0
        structure_pred = torch.clamp(structure_pred, 0.0, 1.0)

        if structure_pred.shape[1] > 1:
            structure_pred = structure_pred.mean(dim=1, keepdim=True)


        D_final = self.efficient_diffusion(support_mask, structure_pred)


        disconnected_material = structure_pred * (1 - D_final)


        topo_loss = torch.mean(disconnected_material, dim=[1, 2, 3])


        time_weights = self.time_weight(t, total_timesteps)
        weighted_loss = time_weights * topo_loss


        return weighted_loss.mean()

    def forward(self, pred_x0, support_mask, t, total_timesteps):

        return self.compute_topo_loss(pred_x0, support_mask, t, total_timesteps)


def create_support_mask_from_bcs(bcs, threshold=0.5):

    support_mask = bcs
    support_mask = (support_mask > threshold).float()
    return support_mask
