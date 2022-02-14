import os
import sys
import time
import torch
import collections
import numpy as np
from torch import optim
from tensorboardX import SummaryWriter


# import dataloader
from data.dataset import Dataset
from models import *
from util import set_seed, UniformSample_original, minibatch, shuffle
from util.parse import parse_args, describe_args
from util.logger import Logger
from util.meter import Meter
from util.configurator import Configurator
from data import PairwiseSampler, PairwiseSamplerV2, PointwiseSamplerV2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

class Net:
    def __init__(self, args):
        self.config = args
        Logger.logger = Logger(name="_".join([self.config.recommender, self.config["data.input.dataset"], self.config.loss, self.config.suffix]),
                               show_in_console=False if self.config.verbose == 0 else True,
                               is_creat_log_file=self.config.create_log_file, path=self.config.log_path)
        Logger.info(args.params_str, "\n")
        self.config.device = torch.device("cuda:0" if torch.cuda.is_available() and not self.config.no_cuda else "cpu")
        write = SummaryWriter('log')
        self.sw_lock = False
        self.dataset = Dataset(self.config)

        # define model
        self.recommender = SLMRec(self.config, self.dataset).to(self.config.device)
        Logger.info(get_parameter_number(self.recommender))

        self.opt = optim.Adam(self.recommender.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

    def run(self):
        topk = str(self.config['topks'][0])
        meter_id = int(time.time())
        loss_meter = Meter("loss", id=meter_id)
        recall_meter = Meter("recall@" + topk, id=meter_id)
        precision_meter = Meter("precision@" + topk, id=meter_id)
        ndcg_meter = Meter("ndcg@" + topk, id=meter_id)
        # Not need to sample negative items
        data_iter = PointwiseSamplerV2(self.dataset, batch_size=self.config.batch_size, shuffle=True)
        hasNeg = False
        try:
            best_recall = 0
            best_epoch = 0
            best_result = ""
            model_name = self.recommender.getFileName()
            self.recommender.train()
            # self.recommender.train_mode = True
            for epoch in range(self.config.num_epoch):
                Logger.info('======================')
                Logger.info(f'EPOCH[{epoch}/{self.config.num_epoch}]')
                start = time.time()
                loss_meter.reset_time()
                
                batch_size = self.config.batch_size if self.config.batch_size != -1 else len(users)

                #  Meters
                batch_loss_meter = Meter(name="MultiLoss")
                batch_loss_meter.reset()
                if self.config.batch_size == -1:
                    loss = self.recommender.__getattribute__(self.config.loss)(users, pos_items)
                    self.opt.zero_grad()
                    loss.backward(retain_graph=True)
                    self.opt.step()
                    batch_loss_meter.update(val=loss.cpu().item())
                else:
                    for bat_users, bat_pos_items in data_iter:
                        bat_users = torch.tensor(bat_users).to(self.config.device)
                        bat_pos_items = torch.tensor(bat_pos_items).to(self.config.device)
                        batch_loss_meter.reset_time()
                        loss = self.recommender.__getattribute__(self.config.loss)(bat_users, bat_pos_items)
                        self.opt.zero_grad()
                        loss.backward(retain_graph=True)
                        self.opt.step()
                        batch_loss_meter.update(val=loss.cpu().item())
                        
                if (epoch + 1) % self.config["test_step"] == 0:
                    Logger.info(f'[TEST]')
                    recall_meter.reset_time()
                    precision_meter.reset_time()
                    ndcg_meter.reset_time()
                    current_result, buf = self.recommender.evaluate()
                    if current_result is not None:
                        recall_meter.update(val=current_result[1], epoch=epoch)
                        precision_meter.update(val=current_result[0], epoch=epoch)
                        ndcg_meter.update(val=current_result[2], epoch=epoch)
                        Logger.info("{}\t{}\t{}".format(recall_meter, ndcg_meter, precision_meter))
                        
                        if current_result[1] > best_recall and epoch != 0:
                            if self.config["save_flag"]:
                                # save model
                                Logger.info(f'[saved][EPOCH {epoch}]')
                                torch.save(self.recommender.state_dict(), model_name)
                            Logger.info("[Better Result]")
                            best_recall = current_result[1]
                            best_epoch = epoch
                            best_result = "[EPOCH {}]\n{}\t{}\t{}".format(epoch, recall_meter, ndcg_meter, precision_meter)
                        else:
                            if epoch - best_epoch > self.config.stop_cnt:
                                # stop training
                                break
                            
                loss_ = batch_loss_meter.avg
                loss_meter.update(val=loss_, epoch=epoch)
                Logger.info(f'[{loss_meter}]')
                Logger.info(f"[TOTAL TIME] {time.time() - start}")
        finally:
            if best_recall != 0:
                Logger.info("=>best_result:\n{}".format(best_result))
            if self.config["save_flag"]:
                loss_meter.save_history(self.config.path)
                ndcg_meter.save_history(self.config.path)
                precision_meter.save_history(self.config.path)
                recall_meter.save_history(self.config.path)


if __name__ == '__main__':
    # load cofig params
    is_windows = sys.platform.startswith('win')
    if is_windows:
        root_folder = 'E:/projects/SLMRec'
    else:
        root_folder = '/home/username/SLMRec/'

    args = Configurator(root_folder + "NeuRec.properties", default_section="hyperparameters")
    
    set_seed(args["seed"])
    egcn = Net(args)
    egcn.run()
