import torch.nn as nn
import torch.nn.functional as F
from  src.models.simclr import EcgResnet


class Classifier(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=2048, out_dim=2) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim)
        )
    
    def forward(self, x):
        x = self.head(x)
        return x
    

def simclr_finetuned_classifier():
    encoder = EcgResnet(
        in_length=2500, 
        in_channels=12, 
        n_grps=4, 
        N=4, 
        num_classes=0, 
        stride=[2,1,2,1], 
        dropout=0.3,
        first_width=32,
        dilation=100
    )
    classifier = Classifier(in_dim=2560, hidden_dim=2560, out_dim=2)
    # linear_classifier = nn.Linear(in_features=2560, out_features=2)

    return encoder, classifier
    