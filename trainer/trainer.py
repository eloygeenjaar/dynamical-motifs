import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 train_dataloader, valid_dataloader, num_epochs, run_type):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.train_dataloader = train_dataloader
        self.num_epochs = num_epochs
        self.run_type = run_type
        self.valid_dataloader = valid_dataloader
        self.do_validation = self.valid_dataloader is not None
        self.log_step = 10

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

        self.train_df = pd.DataFrame(
            np.zeros((self.num_epochs, len(metric_ftns) + 1)),
            index=list(range(1, num_epochs+1)), columns=['tr_loss'] + [f'tr_{m.__name__}' for m in self.metric_ftns])
        self.valid_df = pd.DataFrame(
            np.zeros((self.num_epochs, len(metric_ftns) + 1)),
            index=list(range(1, num_epochs+1)), columns=['va_loss'] + [f'va_{m.__name__}' for m in self.metric_ftns])

        self.lamb = 0.

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        num_s = 0
        for batch_idx, (data, (_, _), target) in enumerate(self.train_dataloader):
            data = data.to(self.device, non_blocking=True).float()
            target = target.to(self.device, non_blocking=True).long()

            self.optimizer.zero_grad(None)
            output = self.model(data)
            output['lambda'] = self.lamb
            loss = self.criterion(output, target)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            loss = loss.detach()

            self.lamb = min(1, self.lamb + self.lambda_step)
            num_s += data.size(0)

            self.train_metrics.update('loss', loss.detach())
            for met in self.metric_ftns:
                with torch.no_grad():
                    self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx, data.size(0)),
                    loss.detach()))

        log = self.train_metrics.result()
        log = {'tr_'+k : v for k, v in log.items()}
        self.train_df.loc[epoch, list(log.keys())] = [float(f) for f in log.values()]
        self.train_df.to_csv(self.checkpoint_dir / 'train.csv')
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            val_log = {'va_'+k : v for k, v in val_log.items()}
            self.valid_df.loc[epoch, list(val_log.keys())] = [float(f) for f in val_log.values()]
            self.valid_df.to_csv(self.checkpoint_dir / 'valid.csv')
            log.update(**val_log)

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        #fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        with torch.no_grad():
            for batch_idx, (data, (_, _), target) in enumerate(self.valid_dataloader):
                data = data.to(self.device, non_blocking=True).float()
                target = target.to(self.device, non_blocking=True).long()

                output = self.model(data)
                output['lambda'] = 1.
                loss = self.criterion(output, target)

                self.valid_metrics.update('loss', loss.detach())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

            #print(output['x_hat'][0].min(), output['x_hat'].max())
            #vminmax = max(output['x_hat'].detach().abs().max(), output['x_orig'].detach().abs().max())
            #axs[0].imshow(output['x_hat'][0].cpu().detach().numpy(), cmap='jet', vmin=-vminmax, vmax=vminmax)
            #axs[1].imshow(output['x_orig'][0].cpu().detach().numpy(), cmap='jet', vmin=-vminmax, vmax=vminmax)
            #plt.savefig(self.checkpoint_dir / 'rec.png')
            #plt.clf()
            #plt.close(fig)

        return self.valid_metrics.result()

    def _progress(self, batch_idx, batch_size):
        base = '[{}/{} ({:.0f}%)]'
        current = batch_idx * batch_size
        total = len(self.train_dataloader.dataset)
        return base.format(current, total, 100.0 * current / total)
