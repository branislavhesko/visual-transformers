from modeling.swin_transformer import SwinTransformer
from modeling.vit import VIT
from modeling.xcit import XCIT


def vit_6_patch_16_heads_16_embed_1024(
        num_classes, image_shape=(224, 224), store_attention=False):
    return VIT(
        in_channels=3,
        embed_size=1024,
        num_classes=num_classes,
        num_layers=6,
        num_heads=16,
        image_shape=image_shape,
        patch_size=16,
        store_attention=store_attention
    )


def swin_small_window_7_embed_96(
        num_classes, image_shape=(224, 224), use_linear_pos_encoding=False):
    return SwinTransformer(
        num_classes=num_classes,
        window_size=7,
        img_shape=image_shape,
        shift_size=3,
        embed_dim=96,
        block_numbers=[2, 2, 6, 2],
        use_linear_pos_encoding=use_linear_pos_encoding,
        in_channels=3
    )


def xcit_6_patch_16_heads_8_embed_384(num_classes, use_linear_pos_encoding=True):
    return XCIT(
        num_classes,
        num_class_attention_layers=2,
        num_xcit_layers=6,
        num_heads=8,
        embed_size=384,
        use_pos_encoding=use_linear_pos_encoding,
        attention_dropout_rate=0.1,
        projection_dropout_rate=0.1,
        drop_path_rate=0.5,
        patch_size=16
        )