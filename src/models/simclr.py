import torch.nn as nn
import torch.nn.functional as F
import  src.models.ecgresnet as ecgresnet
import src.models.ecg_transformer as ecg_t


class AveragePool(nn.Module):
    def forward(self, x):
        signal_size = x.shape[-1]
        kernel = nn.AvgPool1d(signal_size)
        average_feature = kernel(x).squeeze(-1)
        return x, average_feature
    

class EcgResnet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.resnet = ecgresnet.ECGResNet(**kwargs)
        self.resnet.flatten = nn.Identity()
        self.resnet.fc1 = AveragePool()
        self.resnet.fc2 = AveragePool()
    
    
    def forward(self, x):
        _, (out, _) = self.resnet(x)
        return out

class EcgTransformer(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model    
    
    def forward(self, x):
        #x = self.model(x)
        #return x.flatten(1)
        x = self.model(x).transpose(0, 1)
        return x[0]


class SimCLR(nn.Module):
    def __init__(self, encoder, projection) -> None:
        super(SimCLR, self).__init__()
        self.encoder = encoder
        self.projection = projection

    def forward(self, x):
        return self.projection(self.encoder(x))



class Projection(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=2048, out_dim=256) -> None:
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
        return F.normalize(x, dim=1)
    

def simclr_ecgresnet(*args, **kwargs):
    
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
    projection = Projection(in_dim=2560, hidden_dim=2560, out_dim=512)

    model = SimCLR(encoder, projection)
    return encoder, projection, model


def simclr_ecgt(model_name = 'ecgt_tiny', patch_size=20, *args, **kwargs):
    if model_name == 'ecgt_base':
        ecgt = ecg_t.ecgt_base(patch_size)
    if model_name == 'ecgt_small':
        ecgt = ecg_t.ecgt_small(patch_size)
    else:
        ecgt = ecg_t.ecgt_tiny(patch_size)

    encoder = EcgTransformer(ecgt)

    embed_dim = ecgt.embed_dim #* ecgt.patch_embed.num_patches
    projection = Projection(in_dim=embed_dim, hidden_dim=2*embed_dim, out_dim=embed_dim)
    model = SimCLR(encoder, projection)

    return encoder, projection, model


def simclr_model(model_name = 'ecgresnet', *args, **kwargs):
    if model_name == 'ecgresnet':
        return simclr_ecgresnet(*args, **kwargs)
    else:
        return simclr_ecgt(model_name, *args, **kwargs)



