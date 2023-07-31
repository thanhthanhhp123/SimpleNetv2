import torch
import torchvision.models as models

model = models.resnet50(pretrained=True)
print(model)