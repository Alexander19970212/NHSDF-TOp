import lightning as L
import torch
import torchvision.models as models
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR


class RSLossPredictorResNet50(L.LightningModule):
    def __init__(self, learning_rate=1e-4, beta=0.00002, warmup_steps=1000, max_steps=10000):
        super().__init__()
        # Load pre-trained ResNet50
        self.backbone = models.resnet50(pretrained=True)
        
        # Modify the first convolutional layer to accept 1-channel input
        self.backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify the final fully connected layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Identity()  # Remove the final FC layer

        self.learning_rate = learning_rate
        self.beta = beta
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        
        # Add your custom head for angle prediction
        self.head = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )

        self.criterion = torch.nn.MSELoss()
        
    def forward(self, x):
        # Ensure input is the correct shape (batch_size, 1, 40, 40)
        if x.shape[1] != 1 or x.shape[2] != 40 or x.shape[3] != 40:
            raise ValueError(f"Expected input shape (batch_size, 1, 40, 40), got {x.shape}")
        
        # Extract features using ResNet50
        features = self.backbone(x)
        # Predict angle using your custom head
        return self.head(features)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # Ensure y_hat and y have same shape before computing loss
        y_hat = y_hat.view(-1)  # Flatten predictions to match target shape
        y = y.view(-1)  # Flatten targets to match prediction shape
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # Ensure y_hat and y have same shape before computing loss
        y_hat = y_hat.view(-1)  # Flatten predictions to match target shape
        y = y.view(-1)  # Flatten targets to match prediction shape
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        
        # Extracting parameters from the optimizer config
        # warmup_steps = self.optimizer_params.get('warmup_steps', self.warmup_steps)
        # max_steps = self.optimizer_params.get('max_steps', self.max_steps)
        # Cosine annealing scheduler for the rest of the epochs
        # Warmup scheduler for the first few epochs


        warmup_lr_lambda = lambda step: (step + 1) / self.warmup_steps if step < self.warmup_steps else 1
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)

        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(self.max_steps - self.warmup_steps), eta_min=0)

        # Combining schedulers
        scheduler = {
            'scheduler': SequentialLR(
                optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[self.warmup_steps]
            ),
            'interval': 'step',
            'frequency': 1
        }

        return {
            "optimizer": optimizer,
            'lr_scheduler': scheduler
        }