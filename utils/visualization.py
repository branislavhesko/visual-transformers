from einops import rearrange
import torch

from modeling.vit import VIT


def get_attentions(model: VIT, image):
    output = model(image)
    attention_mats = model.attention_store
    if attention_mats is None:
        raise AttributeError("Attention store is None, VIT is not initialized to store attentions!!")

    attention_mats = torch.stack(attention_mats, dim=1).squeeze(0)
    attention_mats = torch.mean(attention_mats, dim=2)
    residual = torch.eye(attention_mats.size(1))
    #attention_mats = attention_mats + residual
    attention_mats = attention_mats / attention_mats.sum(dim=-1).unsqueeze(-1)
    return attention_mats


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    model_ = VIT(3, 768, 2, 6, 8, 224, 16, store_attention=True)
    image = torch.rand(1, 3, 224, 224)
    mats = get_attentions(model_, image)
    for idx in range(mats.shape[0]):
        plt.imshow(mats[idx, :, :].numpy())
        plt.show()
