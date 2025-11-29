from torch.utils.data import Dataset

class Config(Dataset):
    backbone: str = "facebook/dinov2-small"