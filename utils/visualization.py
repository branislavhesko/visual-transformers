import cv2
from einops import rearrange
import numpy as np
import torch

from modeling.vit import VIT


def get_attentions(model: VIT, image):
    model.reset()
    output = model(image)
    attention_mats = model.attention_store
    return attention_mats


# TODO: review
def visualize(att_mat, im):
    att_mat = torch.stack(att_mat).squeeze(1)

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=2)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    mask = cv2.resize(mask / mask.max(), tuple(im.shape[:2]))[..., np.newaxis]
    return mask


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    model_ = VIT(3, 768, 2, 6, 8, 224, 16, store_attention=True)
    image = torch.rand(1, 3, 224, 224)
    mats = get_attentions(model_, image)
    for idx in range(mats.shape[0]):
        plt.imshow(mats[idx, :, :].numpy())
        plt.show()
