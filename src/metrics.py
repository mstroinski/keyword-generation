from bert_score import BERTScorer
import torch

class BertScore(BERTScorer):
    def __init__(self, lang: str = "en", rescale_with_baseline: bool = True):
        super(BertScore, self).__init__(lang=lang, rescale_with_baseline=rescale_with_baseline)
        
    def compute(self, predictions: list[str], gt_keywords: list[list[str]]) -> dict[str, torch.Tensor]:
        predictions_list = [sample.split(",") for sample in predictions]
        precission_lst, recall_lst, f1_lst = [], [], []
        
        for sample_pred, sample_kw in zip(predictions_list, gt_keywords):
            extended_kw = [sample_kw for _ in range(len(sample_pred))]
            precision, recall, f1 = self.score(cands=sample_pred, refs=extended_kw)
            
            precission_lst.append(precision)
            recall_lst.append(recall)
            f1_lst.append(f1)
            
        precision_mean = torch.mean(input=torch.cat(precission_lst))
        recall_mean = torch.mean(input=torch.cat(recall_lst))
        f1_mean = torch.mean(input=torch.cat(f1_lst))
        
        return {"bertscore_precission": precision_mean, "bertscore_recall": recall_mean, "bertscore_f1": f1_mean}