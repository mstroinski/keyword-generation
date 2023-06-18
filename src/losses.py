import torch


class CrossEntropy(torch.nn.CrossEntropyLoss):
    def __init__(self):
        super(CrossEntropy, self).__init__()