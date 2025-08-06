# utils/trainer.py
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from time import time
from logging import getLogger

class Trainer:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.logger = getLogger()
        self.device = torch.device(f'cuda:{config["gpu_id"]}' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.learning_rate = config.get('learning_rate', 1e-3)
        self.epochs = config.get('epochs', 100)
        self.clip_grad_norm = config.get('clip_grad_norm', None)
        self.patience = config.get('patience', 5)
        self.min_delta = config.get('min_delta', 1e-4)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.95)

        self.best_loss = float('inf')
        self.no_improve_count = 0

    def fit_pretrain(self):
        v_feat, t_feat = self.model.get_features()
        batch_size = self.config['train_batch_size']
        num_batches = (len(v_feat) + batch_size - 1) // batch_size

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            start_time = time()

            for batch_idx in range(num_batches):
                batch_v = v_feat[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(self.device)
                batch_t = t_feat[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(self.device)

                loss = self.model.calculate_loss_diff_batch(batch_t, batch_v)
                if isinstance(loss, tuple):
                    loss = sum(loss)

                if torch.isnan(loss):
                    self.logger.warning(f"NaN loss at epoch {epoch}, stopping.")
                    return

                self.optimizer.zero_grad()
                loss.backward()
                if self.clip_grad_norm:
                    clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
                self.optimizer.step()
                total_loss += loss.item()

            self.lr_scheduler.step()
            avg_loss = total_loss / num_batches
            elapsed = time() - start_time
            self.logger.info(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")

            if self.best_loss - avg_loss > self.min_delta:
                self.best_loss = avg_loss
                self.no_improve_count = 0
                self.logger.info(f"New best loss: {avg_loss:.4f}")
            else:
                self.no_improve_count += 1
                self.logger.info(f"No improvement. Patience: {self.no_improve_count}/{self.patience}")

            if self.no_improve_count >= self.patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

    @torch.no_grad()
    def infer(self):
        self.model.eval()
        v_feat, t_feat = self.model.get_features()
        batch_size = self.config['eval_batch_size']
        num_batches = (len(v_feat) + batch_size - 1) // batch_size

        total_predicted_v = []

        for batch_idx in range(num_batches):
            batch_v = v_feat[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(self.device)
            batch_t = t_feat[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(self.device)

            predicted_v, _ = self.model.calculate_loss_diff_batch(batch_t, batch_v)
            total_predicted_v.append(predicted_v)

        return torch.cat(total_predicted_v, dim=0)
