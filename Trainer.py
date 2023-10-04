import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import tqdm
import torch.nn.functional as F

class Trainer:
    def __init__(
            self, model: nn.Module,
            train_dl: DataLoader, val_dl: DataLoader,
            batch_size: int = 64, vocab_size: int=50_000,
            file_path: str = "results.csv") -> None:
        self.file_path = file_path
        self.init_df()
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.configure_optimizer()


    def train_step(self, Xbatch, ybatch):
        self.optimizer.zero_grad()
        y_pred = self.model(Xbatch)
        loss = self.loss_fn(y_pred, ybatch)
        loss.backward()
        self.optimizer.step()
        self.model.reset_hidden_state()
        return loss.item()

    def val_step(self, X_val, y_val):
        with torch.no_grad():
            y_pred_test = self.model(X_val)
            loss = self.loss_fn(y_pred_test, y_val)
            accuracy_test = (y_pred_test.round() == y_val).float().mean()
        return accuracy_test.item(), loss.item()

    def train(self, epochs:int):
        for epoch in range(1, epochs+1):
            train_loss = 0.
            with tqdm.tqdm(self.train_dl, unit="batch") as tepoch:
                for Xbatch, ybatch in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                    train_loss = self.train_step(Xbatch.cuda(), ybatch.cuda())
                    tepoch.set_postfix(loss=train_loss)
            val_loss = 0.
            val_acc = 0.
            for _, (X_val_batch, y_val_batch) in enumerate(self.val_dl):
                acc_test, test_loss =self.val_step(X_val_batch.cuda(),
                                        y_val_batch.cuda())
                val_loss += test_loss
                val_acc += acc_test
            val_loss /= len(self.val_dl)
            val_acc /= len(self.val_dl)
            file = open(self.file_path, "a")
            file.write(f"{epoch}, {train_loss}, {val_loss}, {val_acc}\n")

    def loss_fn(self, pred, target):
         return F.cross_entropy(pred.view(-1, self.vocab_size), target.view(-1))

    def configure_optimizer(self):
         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    
    def init_df(self):
        file = open(self.file_path, "a")
        file.write("Epoch, Train_Loss, Val_Loss, Test_Accuracy\n")