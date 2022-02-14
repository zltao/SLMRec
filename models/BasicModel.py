"""
BasicModel
"""
import torch
from torch import nn
from evaluator import ProxyEvaluator
import os

class BasicModel(nn.Module):
    def __init__(self,dataset, config):
        super(BasicModel, self).__init__()
        self.config = config
        self.dataset = dataset
        self.temp = self.config["temp"]
        self.evaluator = ProxyEvaluator(dataset,
                                        dataset.get_user_train_dict(),
                                        dataset.get_user_test_dict(),
                                        dataset.get_user_test_neg_dict(),
                                        metric=self.config["metric"],
                                        group_view=self.config["group_view"],
                                        top_k=self.config["topk"],
                                        batch_size=self.config["test_batch_size"],
                                        num_thread=self.config["num_thread"])
        self.infonce_criterion = nn.CrossEntropyLoss()
        
    def getFileName(self):
        
        suffix = self.config["suffix"]
        if not os.path.exists(self.config.path):
            os.mkdir(self.config.path)
        file = f"{self.config.recommender}-{self.config.dataset}-{suffix}.pth.tar"
        return os.path.join(self.config.path, file)
    
    def predict(sself, user_ids, candidate_items=None):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError
    def evaluate(self):
        return self.evaluator.evaluate(self)

    def getEmbedding(self, users, pos_items, neg_items):
        """
        return :
            users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
        """
        raise NotImplementedError
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        users_emb=torch.nn.functional.normalize(users_emb,dim=1)
        pos_emb=torch.nn.functional.normalize(pos_emb,dim=1)
        neg_emb=torch.nn.functional.normalize(neg_emb,dim=1)
 
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss 
    
    def infonce(self, users, pos):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), None)
        users_emb=torch.nn.functional.normalize(users_emb,dim=1)
        pos_emb=torch.nn.functional.normalize(pos_emb,dim=1)
        logits = torch.mm(users_emb,pos_emb.T)
        logits /= self.temp
        labels = torch.tensor(list(range(users_emb.shape[0]))).to(self.config.device)
        
        return  self.infonce_criterion(logits, labels) 
