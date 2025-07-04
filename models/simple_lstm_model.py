import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class SimpleLSTM(pl.LightningModule):
    """
    Simple LSTM model for RUL prediction.
    This is a simpler alternative to the TFT model that requires less memory.
    """
    def __init__(
        self,
        input_size: int = 24,  # Number of features
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        output_size: int = 1   # RUL prediction
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        # Model parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.output_size = output_size
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size]
            
        Returns:
            Output tensor of shape [batch_size, output_size]
        """
        # LSTM output: [batch_size, seq_len, hidden_size]
        lstm_out, _ = self.lstm(x)
        
        # Take the last time step
        lstm_out = lstm_out[:, -1]
        
        # Output layer
        output = self.fc(lstm_out)
        
        return output
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat.squeeze(), y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.mse_loss(y_hat.squeeze(), y)
        
        # Calculate metrics
        mae = F.l1_loss(y_hat.squeeze(), y)
        rmse = torch.sqrt(val_loss)
        
        # Calculate precision@25_cycles (percentage of predictions within 25 cycles of true RUL)
        precision_25 = torch.mean((torch.abs(y_hat.squeeze() - y) <= 25).float())
        
        # Log metrics
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_mae', mae, prog_bar=True)
        self.log('val_rmse', rmse, prog_bar=True)
        self.log('val_precision_25', precision_25, prog_bar=True)
        
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.mse_loss(y_hat.squeeze(), y)
        
        # Calculate metrics
        mae = F.l1_loss(y_hat.squeeze(), y)
        rmse = torch.sqrt(test_loss)
        
        # Calculate precision@25_cycles (percentage of predictions within 25 cycles of true RUL)
        precision_25 = torch.mean((torch.abs(y_hat.squeeze() - y) <= 25).float())
        
        # Log metrics
        self.log('test_loss', test_loss)
        self.log('test_mae', mae)
        self.log('test_rmse', rmse)
        self.log('test_precision_25', precision_25)
        
        return test_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        } 