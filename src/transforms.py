import torch
import  torch.nn.functional as F
import random
import math


def make_transforms(num_transform, device):
    """Build a transformation path of `num_transform` squentially applied transforms
    
    @param num_transform (int) number of transforms
    @return transforms (List[torch->Transform]) random list of `num_transform` transforms
    """
    classes = [
        Identity(), 
        GaussianNoise({"scale": 5}), 
        GaussianBlur({"device": device}), 
        SobelDerivative({"device": device}),
        Scale(),
        TimeWarp({"stretch_factor": 0.1}),
        Reverse(),
        Invert(),
        RandWanderer({"amp": 5, "start_phase": 10, "end_phase": 100}),
        RandomResizedCrop({"n_samples": 500})
    ]
    transform_list = []
    for _ in range(num_transform):
        transform_list += [random.choice(classes)]
    return transform_list
    # return transforms.Compose(transform_list)



class Identity(object):
    def __call__(self, sample: torch.Any) -> torch.Any:
        """ Identify transformation

        @param sample (torch.FloatTensor of shape (W, C)): ecg sample
        @return same sample
        """
        return sample

class GaussianNoise(object):
    def __init__(self, params) -> None:
        self.scale = params.get("scale", 0.01)

    def __call__(self, sample: torch.Any) -> torch.Any:
        """ Add gaussian noise of variance [self.scale] and mean 0 to the sample

        @param sample (torch.FloatTensor of shape (W, C)): ecg sample
        @return noisy sample
        """
        noise = self.scale * torch.randn(sample.shape, device=sample.device)
        return sample + noise

class Blur(object):
    """ Apply a filter blur with kernel `kernel` to the sample """

    def __init__(self, params) -> None:
        kernel = params.get("kernel", [])
        device = params.get("device", torch.device('cpu'))
        ksize = len(kernel)
        self.conv = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=ksize, padding=ksize//2, bias=False).to(device)
        self.conv.weight.data = torch.nn.Parameter(torch.tensor([[kernel]], device=device))
        self.conv.weight.requires_grad = False

    def __call__(self, sample:  torch.Any) -> torch.Any:
        """ Apply a filter blur with kernel `kernel` to the sample

        @param sample (torch.FloatTensor of shape (W, C)): ecg sample
        @return blurred sample
        """
        if len(sample.shape) == 2: # Single item, change into a batch with 1 element
            sample.unsqueeze_(0)
        B, W, C = sample.shape
        return self.conv(sample.view(B, 1, W*C)).view(B, W, C)

class GaussianBlur(Blur):
    """ Apply a gaussian blur with kernel `kernel` to the sample """

    def __init__(self, params) -> None:
        params["kernel"] = params.get("kernel", [.1, .2, .4, .2, .1])
        super(GaussianBlur, self).__init__(params)
        
class SobelDerivative(Blur):
    """ Apply a sobel derivative filter blur with kernel `kernel` to the sample """

    def __init__(self, params) -> None:
        params["kernel"] = params.get("kernel", [1, .0, -1])
        super(SobelDerivative, self).__init__(params)

class Scale(object):
    def __init__(self, params={}):
        super().__init__()
        self.factor = params.get("max_factor", 5) * 10

    def __call__(self, x):
        factor = random.randint(1, self.factor) / 10
        return torch.mul(factor, x)
 
class TimeWarp(object):
    """Currently supports only stretching"""

    def __init__(self, params={}):
        super().__init__()
        self.ratio = params.get("stretch_factor", "random")
        if self.ratio == "random":
            self.ratio = random.uniform(0.1, 0.9)
        assert self.ratio > 0 and self.ratio < 1, "Stretch factor must be between 0 and 1"

    def __call__(self, x):
        is_batched = len(x.shape) == 3 # Single item, change into a batch with 1 element
        if not is_batched: 
            x.unsqueeze_(0)

        y = F.interpolate(F.interpolate(x, scale_factor=self.ratio, mode="linear"), size=x.shape[2], mode="linear")
        return y if is_batched else y.squeeze(0)
    
class Reverse(object):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return torch.flip(x, dims=[-1])   

class Invert(object):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return torch.neg(x)
    
class RandWanderer(object):
    def __init__(self, params):
        super().__init__()
        self.amp = params.get("amp")
        self.start_phase = params.get("start_phase")
        self.end_phase = params.get("end_phase")
        self.gn_scale = params.get("gn_scale", 5)

    def __call__(self, x):
        sn = torch.linspace(self.start_phase, self.end_phase, x.shape[-1], device=x.device)
        sn = self.amp * torch.sin(math.pi * sn / 180)
        gauss_noise = GaussianNoise({"scale": self.gn_scale})(x)
        return x + sn + gauss_noise

class RandomResizedCrop(object):
    def __init__(self, params):
        super().__init__()
        self.n_samples = params.get("n_samples")

    def __call__(self, x):
        is_batched = len(x.shape) == 3 # Single item, change into a batch with 1 element
        if not is_batched: 
            x.unsqueeze_(0)

        max_samples = x.shape[-1]
        start_idx = random.randint(0, max_samples - self.n_samples)    
        y = F.interpolate(x[..., start_idx : start_idx + self.n_samples], size=x.shape[2], mode="linear")
        return y if is_batched else y.squeeze(0)
        

# class RandomResizedCrop(object):
#     """Extract crop at random position and resize it to full size"""  

#     def __init__(self, params):
#         self.crop_ratio_range = params.get("crop_ratio_range", [.5, 1.])
#         self.output_size = params.get("output_size", 250)
    
#     def __call__(self, sample: torch.Any) -> torch.Any:
#         output = torch.full(sample[0].shape, float("inf")).type(sample[0].type())
#         timesteps, channels = output.shape
#         crop_ratio = random.uniform(*self.crop_ratio_range)
#         data, label = TRandomCrop(int(crop_ratio*timesteps))(sample)  # apply random crop
#         cropped_timesteps = data.shape[0]
#         indices = torch.sort((torch.randperm(timesteps-2)+1)[:cropped_timesteps-2])[0]
#         indices = torch.cat([torch.tensor([0]), indices, torch.tensor([timesteps-1])])
#         output[indices, :] = data  # fill output array randomly (but in right order) with values from random crop
        
#         # use interpolation to resize random crop
#         output = Tinterpolate(output, float("inf"))
#         return output, label 