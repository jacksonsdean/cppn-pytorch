import torch
from piq.feature_extractors import InceptionV3

@torch.no_grad()
def compute_feats(
        dist,
        feature_extractor: torch.nn.Module = None,
        device: str = 'cuda') -> torch.Tensor:
    r"""Generate low-dimensional image descriptors

    Args:
        loader: Should return dict with key `images` in it
        feature_extractor: model used to generate image features, if None use `InceptionNetV3` model.
            Model should return a list with features from one of the network layers.
        out_features: size of `feature_extractor` output
        device: Device on which to compute inference of the model
    """

    if feature_extractor is None:
        # print('WARNING: default feature extractor (InceptionNet V2) is used.')
        feature_extractor = InceptionV3()
    else:
        assert isinstance(feature_extractor, torch.nn.Module), \
            f"Feature extractor must be PyTorch module. Got {type(feature_extractor)}"
    feature_extractor.to(device)
    feature_extractor.eval()

    total_feats = []
    N = dist.shape[0]
    images = dist.float().to(device)

    # Get features
    features = feature_extractor(images)
    # TODO(jamil 26.03.20): Add support for more than one feature map
    assert len(features) == 1, \
        f"feature_encoder must return list with features from one layer. Got {len(features)}"
    total_feats.append(features[0].view(N, -1))

    return torch.cat(total_feats, dim=0)



from transformers import AutoFeatureExtractor, AutoModelForImageClassification
extractor = AutoFeatureExtractor.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10")
classifier = AutoModelForImageClassification.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10")