import torch

class EvalPMD(torch.nn.Module):
    """
    网格逐点欧几里得距离(PMD)
    Init:
        None
    Args:
        pred_v: Tensor[BV,3]    预测点集
        gt_v: Tensor[BV,3]      正确点集
    Returns:
        eval: Tensor[1]         PMD指标
    """
    def __init__(self):
        super(EvalPMD, self).__init__()
        self.Eval = torch.nn.MSELoss(reduction='mean')

    def forward(self, pred_v, gt_v):
        return self.Eval(pred_v, gt_v)