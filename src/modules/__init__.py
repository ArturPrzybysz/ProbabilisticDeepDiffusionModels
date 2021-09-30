from .dense import DenseModel
from .unet import UNetModel

# MODELS = {"dense": DenseModel, "unet": UNetModel}


def get_model(resolution, cfg):
    name = cfg.pop("name")
    if name != "unet":
        raise ValueError(f"Only 'unet' model supported.")
    return get_unet(resolution, **cfg)


def get_unet(
    resolution,
    in_channels,
    model_channels,
    num_res_blocks,
    attention_resolutions,
    dropout=0,
    channel_mult=(1, 2, 4, 8),
    conv_resample=True,
    dims=2,
    num_classes=None,
    use_checkpoint=False,
    num_heads=1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
):
    attention_ds = []
    for res in attention_resolutions:
        attention_ds.append(resolution // int(res))

    learn_sigma = False  # TODO

    return UNetModel(
        in_channels=in_channels,
        model_channels=model_channels,
        out_channels=(in_channels if not learn_sigma else in_channels * 2),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=num_classes,
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )
