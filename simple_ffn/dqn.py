"""
Deep Q-network
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Create a SummaryWriter, specifying the log directory
writer = SummaryWriter('logs')

NN_DEVICE = "cpu"


class ModelBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.in_features = kwargs.get("in_features")
        self.out_features = kwargs["out_features"]


class RunnerBase:
    def __init__(self, **kwargs):
        self.learning_rate = kwargs["learning_rate"]
        self.momentum = kwargs["momentum"]
        self.num_epochs = kwargs["num_epochs"]

    def print_epoch_info(self, current_epoch: int, loss_rate: float):
        print(f"[{current_epoch+1}/{self.num_epochs}] Training loss: {loss_rate:.6f}")


class DQNModel(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hidden_dimensions = self.in_features * 2
        self.net = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_dimensions),
            nn.Sigmoid(),
            nn.Linear(self.hidden_dimensions, self.out_features),
        )

    def forward(self, x):
        x = self.net(x)

        return x


class DQNRunner(RunnerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = DQNModel(in_features=2, out_features=2).to(NN_DEVICE)

        # Loss function
        self.criterion = nn.MSELoss().to(NN_DEVICE)

        # Optimizer
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def train(self, train_loader, test_loader):
        for epoch in range(self.num_epochs):
            self.model.train()

            total_loss = 0

            for inputs, targets in train_loader:
                outputs = self.model(inputs.unsqueeze(1))
                loss = self.criterion(outputs.squeeze(), targets)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            average_loss = total_loss / len(train_loader)

            writer.add_scalar('Loss/Train', average_loss, global_step=epoch)

            self.print_epoch_info(epoch, average_loss)

        self.save_model_state()

    def save_model_state(self):
        torch.save(self.model.state_dict(), "dqn_checkpoint.m")

    def eval(self):
        self.model.load_state_dict(torch.load("dqn_checkpoint.m"))
        self.model.eval()
