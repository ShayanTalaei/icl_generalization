from pathlib import Path

import torch
import torch.nn as nn

from src.models.base import SeqModel
from src.tasks.base import ICLTask


class Trainer:
    """Minimal training loop for ICL models."""

    def __init__(self, model: SeqModel, task: ICLTask, config):
        self.model = model
        self.task = task
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        if config.lr_schedule == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.num_steps
            )
        else:
            self.scheduler = None

        self.step = 0

    def train(self):
        self.model.train()
        cfg = self.config

        for step in range(1, cfg.num_steps + 1):
            self.step = step

            batch = self.task.sample_batch(cfg.batch_size, cfg.num_examples)
            xs = batch.xs.to(self.device)
            ys = batch.ys.to(self.device)

            y_preds = self.model(xs, ys)

            loss = nn.functional.mse_loss(y_preds, ys)
            query_loss = nn.functional.mse_loss(y_preds[:, -1, :], ys[:, -1, :])

            self.optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            if step % cfg.eval_every == 0:
                eval_loss = self.evaluate()
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"[step {step:>6d}/{cfg.num_steps}]  "
                    f"train_loss={loss.item():.4f}  "
                    f"query_loss={query_loss.item():.4f}  "
                    f"eval_query_loss={eval_loss:.4f}  "
                    f"lr={lr:.2e}"
                )

            if cfg.checkpoint_every > 0 and step % cfg.checkpoint_every == 0:
                self.save_checkpoint()

    @torch.no_grad()
    def evaluate(self, num_batches: int = 10) -> float:
        """Evaluate query-position MSE on fresh data."""
        self.model.eval()
        total_loss = 0.0
        cfg = self.config

        for _ in range(num_batches):
            batch = self.task.sample_batch(cfg.batch_size, cfg.num_examples)
            xs = batch.xs.to(self.device)
            ys = batch.ys.to(self.device)

            y_preds = self.model(xs, ys)
            total_loss += nn.functional.mse_loss(
                y_preds[:, -1, :], ys[:, -1, :]
            ).item()

        self.model.train()
        return total_loss / num_batches

    def save_checkpoint(self):
        path = Path(self.config.checkpoint_dir) / f"step_{self.step}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": self.step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        print(f"  -> checkpoint saved: {path}")
