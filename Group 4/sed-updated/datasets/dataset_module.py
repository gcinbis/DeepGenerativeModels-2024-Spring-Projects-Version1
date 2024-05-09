from .combined_dataset_train import CombinedDatasetTrain
from .combined_dataset_test import CombinedDatasetTest
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class DatasetModule(pl.LightningDataModule):
    def __init__(self,
                 num_workers, 
                 train_batch_size,
                 val_batch_size,
                 test_batch_size,
                 train_dataset_config,
                 val_dataset_config,
                 test_dataset_config
                 ):
        super().__init__()
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.train_dataset_config = train_dataset_config
        self.val_dataset_config = val_dataset_config
        self.test_dataset_config = test_dataset_config
        
    def setup(self, stage: str):
        if stage == 'training':
            self.train_dataset = CombinedDatasetTrain(**self.train_dataset_config)
        elif stage == 'test':
            self.test_dataset = CombinedDatasetTest(**self.test_dataset_config)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size,
                            shuffle=True, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size,
                          shuffle=False, num_workers=self.num_workers)