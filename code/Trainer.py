import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import tqdm
import torch.nn.functional as F
from FrankensteinDataset import FrankensteinDataSet


class Trainer:
    def __init__(
            self, model: nn.Module,
            train: FrankensteinDataSet,  val: FrankensteinDataSet,
            batch_size: int = 64, vocab_size: int = 50_000,
            file_path: str = "results.csv") -> None:
        self.file_path = file_path
        self.init_df()
        self.model = model.cuda()
        self.train_dl = train.dl
        self.val_dl = val.dl
        self.len_val = val.len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.configure_optimizer()

    def train_step(self, Xbatch, ybatch):
        self.optimizer.zero_grad()
        y_pred = self.model(Xbatch)
        loss = self.loss_fn(y_pred, ybatch)
        loss.backward()
        self.optimizer.step()
        if self.model.__class__.__name__ == 'FrankensteinsRNN':
            self.model.reset_hidden_state()
        return loss.item()

    def val_step(self, X_val, y_val):
        with torch.no_grad():
            y_pred_test = self.model(X_val)
            loss = self.loss_fn(y_pred_test, y_val)
            y_val = self.transform_target(y_val)
        return loss.item()

    def train(self, epochs: int):
        for epoch in range(1, epochs+1):
            train_loss = 0.
            with tqdm.tqdm(self.train_dl, unit="batch") as tepoch:
                for Xbatch, ybatch in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                    train_loss = self.train_step(Xbatch.cuda(), ybatch.cuda())
                    tepoch.set_postfix(loss=train_loss)
            val_loss = 0.
            for _, (X_val_batch, y_val_batch) in enumerate(self.val_dl):
                test_loss =self.val_step(X_val_batch.cuda(),
                                        y_val_batch.cuda())
                val_loss += test_loss
            val_loss /= self.len_val
            file = open(self.file_path, "a")
            file.write(f"{epoch}, {train_loss}, {val_loss}\n")


    def loss_fn(self, pred, target):
        target_ = self.transform_target(target)
        return F.cross_entropy(pred.view(-1, self.vocab_size), target_)

    def transform_target(self, target):
        target = target.view(-1)
        target_ = torch.zeros(
            target.shape[0], self.vocab_size
            ).cuda()
        target_[range(target_.shape[0]), target] = 1
        return target_

    def configure_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1e-4)

    def init_df(self):
        file = open(self.file_path, "a")
        file.write("Epoch, Train_Loss, Val_Loss\n")
