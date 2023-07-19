from typing import Any, Optional
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion
from transformers.tokenization_utils import PreTrainedTokenizer
from hydra.utils import instantiate
from datetime import datetime
import lightning as L
import evaluate
import torch

from transformers import get_linear_schedule_with_warmup

from config.kw_config import Model
from src.metrics import BertScore

class KWModel(L.LightningModule):
    def __init__(self, config: Model, batch_size: int):
        super(KWModel, self).__init__()
        self.config = config
        self.model = instantiate(config.huggingface.nnet)
        self.tokenizer: PreTrainedTokenizer = instantiate(config.huggingface.tokenizer)
        self.loss = instantiate(config.loss)
        self.rouge = evaluate.load("rouge")
        self.bert_score = BertScore()
        self.batch_size = batch_size
        
        _now = datetime.now().strftime("%H:%M:%S")
        self.test_output_path = config.test_output + _now + ".txt"
    
    def forward(self, x: list|str) -> str:
        tokenized_sample = self.tokenizer(x,
                                          truncation=True,
                                          return_tensors="pt")
        input_ids = tokenized_sample["input_ids"]

        if input_ids.dim() > 2:
            input_ids = input_ids.squeeze()
            
        model_output = self.model.generate(input_ids)
        
        decoded_output = self.tokenizer.decode(model_output[0], skip_special_tokens=True)
        
        return decoded_output
    
    def training_step(self, batch: list[dict], batch_idx: int) -> torch.Tensor:
        train_loss, _ = self.step(batch)
    
        self.log("train_loss", train_loss, on_epoch=True, batch_size=self.batch_size)

        return train_loss
    
    def validation_step(self, batch: list[dict], batch_idx: int) -> None:
        val_loss, logits = self.step(batch)
        
        keywords = [sample["kw"] for sample in batch]
        metrics = self.calculate_metrics(logits=logits, gt_keywords=keywords)
        metrics["val_loss"] = val_loss
        
        self.log_dict(metrics, on_epoch=True, batch_size=self.batch_size)
    
    def test_step(self, batch: list[dict], batch_idx: int) -> None:
        _, logits = self.step(batch)
        
        keywords = [sample["kw"] for sample in batch]
        metrics = self.calculate_metrics(logits=logits, gt_keywords=keywords)
        metrics = {"test_"+key: val for key, val in metrics.items()}
        
        self.save_test_output(logits=logits, gt_keywords=keywords)
        self.log_dict(metrics, on_epoch=True, batch_size=self.batch_size)
        
    def step(self, batch: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        texts = [' '.join(["Generate keywords: ", sample["title"], sample["abstract"]]) for sample in batch]
        encoded_text = self.tokenizer(texts, 
                                      return_tensors="pt", 
                                      max_length=self.config.huggingface.text_tokenizer_max_len, 
                                      padding="max_length", 
                                      truncation=True)
        
        keywords = [' , '.join(sample["kw"]) for sample in batch]
        

        encoded_keywords = self.tokenizer(keywords, 
                                        return_tensors="pt", 
                                        max_length=self.config.huggingface.kw_tokenizer_max_len, 
                                        padding="max_length", 
                                        truncation=True)

        encoded_keywords.input_ids[encoded_keywords.input_ids[:, :] == self.tokenizer.pad_token_id] = -100

        # This approach calculates loss twice but it still outperforms generate() / manual label shifting 
        logits = self.model(input_ids=encoded_text.input_ids.to("cuda:0"), 
                attention_mask=encoded_text.attention_mask.to("cuda:0"), 
                decoder_attention_mask=encoded_keywords.attention_mask.to("cuda:0"), 
                labels=encoded_keywords.input_ids.to("cuda:0")).logits

        loss = self.loss(logits.view(-1, logits.size(-1)), encoded_keywords.input_ids.view(-1).to("cuda:0"))

        return loss, logits
        
    def calculate_metrics(self, logits: torch.Tensor, gt_keywords: list[list[str]]) -> dict:
        metrics = dict()
        pred_tokens = torch.argmax(input=logits, dim=-1)
        predictions = self.tokenizer.batch_decode(pred_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        rouge = self.rouge.compute(predictions=predictions, references=gt_keywords)
        if rouge:
            metrics.update(rouge)
            
        bert_score  = self.bert_score.compute(predictions=predictions, gt_keywords=gt_keywords)
        if bert_score:
            metrics.update(bert_score)
            
        return metrics
    
    def save_test_output(self, logits: torch.Tensor, gt_keywords: list[list[str]]) -> None:
        pred_tokens = torch.argmax(input=logits, dim=-1)
        predictions = self.tokenizer.batch_decode(pred_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        with open(self.test_output_path, "a") as f:
            for pred, kw in zip(predictions, gt_keywords):
                sample = f"Predictions: {pred}\nGT: {kw}\n\n"
                f.write(sample)
    
    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, metric: Any | None) -> None:
        return scheduler.step(epoch=self.current_epoch)
        
    def configure_optimizers(self) -> dict:
        optimizer = instantiate(self.config.optimizer, params=self.parameters())
        # scheduler = instantiate(self.config.scheduler, optimizer=optimizer)
        return optimizer
        #return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss"}}
