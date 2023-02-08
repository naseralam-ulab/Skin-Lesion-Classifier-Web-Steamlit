import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


def predict(image_path, model):

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    transforms = A.Compose([A.Resize(224 , 224), A.Normalize(mean, std), ToTensorV2()])
    
    img = Image.open(image_path)
    img = np.asarray(img)
    transformed_img = transforms(image=img)["image"]
    batch_t = torch.unsqueeze(transformed_img, 0)

    model.eval()
    out = model(batch_t)

    with open('isic_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
    
    prob = torch.nn.functional.softmax(out, dim=1)[0] * 8
    _, indices = torch.sort(out, descending=True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:1]]