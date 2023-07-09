from dataclasses import dataclass

@dataclass
class Trainer():
    accelerator: str
    devices: int
    epoch_count: int
    stopping_patience: int

@dataclass
class NNet():
    pretrained_model_name_or_path: str

@dataclass
class Tokenizer():
    pretrained_model_name_or_path: str

@dataclass    
class HuggingFace():
    huggingface_model: str
    text_tokenizer_max_len: int
    kw_tokenizer_max_len: int
    huggingface_tokenizer: str
    huggingface_nnet: str
    nnet: NNet
    tokenizer: Tokenizer
    
@dataclass
class Optimizer():
    lr: float
    
@dataclass
class Scheduler():
    gamma: float
    
@dataclass 
class Loss():
    pass
    
@dataclass
class Model():
    huggingface: HuggingFace
    optimizer: Optimizer
    scheduler: Scheduler
    loss: Loss
    output_max_len: int
    test_output: str

@dataclass
class Data():
    path: str
    num_workers: int
    batch_size: int

@dataclass
class WandB():
    project: str
    local_path: str

@dataclass
class Lightning():
    local_path: str

@dataclass
class Logging():
    wandb: WandB
    lightning: Lightning

@dataclass
class KWConfig():
    logging: Logging
    model: Model
    data: Data
    trainer: Trainer