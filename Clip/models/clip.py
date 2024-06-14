from torch import nn
from img_encoder import ImageEncoder

class clip(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.img_encoder = ImageEncoder()
        
    