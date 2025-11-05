import torch
import torch.nn as nn
import torch.nn.functional as F
'''
https://zhuanlan.zhihu.com/p/482153702
'''
class OrdinalRegressionLoss(nn.Module):

    def __init__(self, num_class,reg_weight=0, train_cutpoints=False):
        super().__init__()
        self.reg_weight = reg_weight
        self.num_classes = num_class
        num_cutpoints = self.num_classes - 1
        self.cutpoints = torch.arange(num_cutpoints).float() + 0.5
        
        self.cutpoints = nn.Parameter(self.cutpoints)
        if not train_cutpoints:
            self.cutpoints.requires_grad_(False)

    def forward(self, pred, label):
        mse = F.mse_loss(pred.float(),label.float())
        if label.ndim ==1:
            label = torch.unsqueeze(label,dim=1)
        if pred.ndim ==1:
            pred = torch.unsqueeze(pred,dim=1)

        sigmoids = torch.sigmoid(self.cutpoints.to(pred.device) - pred)
        link_mat = sigmoids[:, 1:] - sigmoids[:, :-1]
        link_mat = torch.cat((sigmoids[:, [0]], link_mat, (1 - sigmoids[:, [-1]]) ),dim=1)

        eps = 1e-15
        likelihoods = torch.clamp(link_mat, eps, 1 - eps)

        neg_log_likelihood = torch.log(likelihoods)

        loss = -torch.gather(neg_log_likelihood, 1, label).mean()
        if self.reg_weight > 0:
            loss = (loss + mse * self.reg_weight) / 2
        return loss
if __name__ == '__main__':
    num_classes = 10
    fun = OrdinalRegressionLoss(10,0.3).cuda()
    result = torch.tensor(range(9))[:,None].cuda()
    label = torch.tensor( [5] * 9 ).long().cuda()
    res = fun(result,label)
    print(res)
    # F.one_hot(binary_target, NUM_TARGET+1).cumsum(-1)[:, :, :-1].transpose(1, 2).squeeze(-1)