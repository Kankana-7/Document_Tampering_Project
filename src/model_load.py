from src.dtd import seg_dtd
import torch
import numpy as np
from albumentations import ToTensorV2
import torchvision

def load_segmentation_model(weights_path="/home/xelpmoc/Documents/DocTamperAPI/artifacts/seg_dtd_model_weights.pth", device="cpu"):
    model = seg_dtd(n_class=2)
    weights = torch.load(weights_path, map_location=device)
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    new_qtb = (
        np.array(
            [
                [2, 1, 1, 2, 2, 4, 5, 6],
                [1, 1, 1, 2, 3, 6, 6, 6],
                [1, 1, 2, 2, 4, 6, 7, 6],
                [1, 2, 2, 3, 5, 9, 8, 6],
                [2, 2, 4, 6, 7, 11, 10, 8],
                [2, 4, 6, 6, 8, 10, 11, 9],
                [5, 6, 8, 9, 10, 12, 12, 10],
                [7, 9, 10, 10, 11, 10, 10, 10],
            ],
            dtype=np.int32,
        )
        .reshape(64,)
        .tolist()
    )

    totsr = ToTensorV2()
    toctsr = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )

    return model, new_qtb, totsr, toctsr

