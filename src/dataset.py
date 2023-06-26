import torch.utils.data
import json

from config.kw_config import Data

class KWDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str):
        super(KWDataset, self).__init__()
        self.data_path = data_path
        
        self.offset_list = self._get_offset_list(self.data_path)
        
    def __len__(self):
        return len(self.offset_list)
    
    def __getitem__(self, idx):
        offset = self.offset_list[idx]
        with open(self.data_path, 'r') as f:
            f.seek(offset)
            line = f.readline()
            try:
                entry = json.loads(line)
            except Exception as e:
                entry = {"title": "", "abstract": "", "kw": [], "fos": {}}
            return entry
    
    def _get_offset_list(self, path: str, chunk_size: int=2**10) -> list[int]:
        offsets = [0]
        with open(path, "rb") as file:
            chunk = file.readlines(chunk_size)
            while chunk:
                for line in chunk:
                    offsets.append(offsets[-1] + len(line))
                chunk = file.readlines(chunk_size)
        return offsets [:-1]
    
def _collate(x):
    """
    Collate function is needed for dataloader to load data with different input dimentions (texts have different len).
    Unfortunately it cannot be passed as lambda as it makes it local to this file which throws an error :c.
    """
    return x
    
def prepare_dataloaders(config: Data) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    train_data = KWDataset(data_path=config.path+"/train.txt")
    val_data = KWDataset(data_path=config.path+"/val.txt")
    test_data = KWDataset(data_path=config.path+"/test.txt")
    
    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=config.batch_size,
                                                   drop_last=False,
                                                   shuffle=True,
                                                   num_workers=config.num_workers,
                                                   collate_fn=_collate)
    
    val_dataloader = torch.utils.data.DataLoader(val_data,
                                                 batch_size=config.batch_size,
                                                 drop_last=False,
                                                 shuffle=False,
                                                 num_workers=config.num_workers,
                                                 collate_fn=_collate)
    
    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=config.batch_size,
                                                  drop_last=False,
                                                  shuffle=False,
                                                  num_workers=config.num_workers,
                                                  collate_fn=_collate)
    
    return train_dataloader, val_dataloader, test_dataloader
  
