import hydra.core.config_store
import lightning as L 

from src.dataset import prepare_dataloaders, KWDataset
from config.kw_config import KWConfig
from src.model import KWModel

cs = hydra.core.config_store.ConfigStore().instance()
cs.store(name="kw_config", node=KWConfig)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: KWConfig):
    # train_dataloader, _, _ = prepare_dataloaders(cfg.data)
    data = KWDataset("data/processed/MATCHED_m36_a15.txt")
    import torch
    train_dataloader = torch.utils.data.DataLoader(data, batch_size=8, collate_fn=lambda x: x)
    wandb_logger = L.pytorch.loggers.WandbLogger(project=cfg.logging.wandb.project, save_dir=cfg.logging.wandb.local_path)    
    wandb_logger.experiment.config.update(cfg)
    
    model_checkpoint = L.pytorch.callbacks.ModelCheckpoint(dirpath=cfg.logging.lightning.local_path,
                                                           monitor="val_loss_epoch")
    
    early_stopping = L.pytorch.callbacks.EarlyStopping(monitor="val_loss_epoch", patience=cfg.trainer.stopping_patience)
    

    model = KWModel(cfg.model, batch_size=cfg.data.batch_size)
    trainer = L.Trainer(accelerator=cfg.trainer.accelerator,
                        devices=cfg.trainer.devices,
                        max_epochs=cfg.trainer.epoch_count,
                        callbacks=[model_checkpoint, early_stopping],
                        logger=wandb_logger)
    
    trainer.fit(model, train_dataloader)
    
    
if __name__ == "__main__":
    main()

