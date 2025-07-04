import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Tuple, Union
import numpy as np


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network as described in the TFT paper.
    This is a key building block that allows for better gradient flow.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int = None, 
                 dropout: float = 0.1, context_size: int = None, layer_norm: bool = True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size if output_size is not None else input_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.layer_norm = layer_norm
        
        # Linear layers for the GRN
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.output_size)
        self.dropout = nn.Dropout(dropout)
        
        # Optional context projection
        if context_size is not None:
            self.context_projection = nn.Linear(context_size, hidden_size, bias=False)
        
        # Skip connection if input and output dimensions differ
        if self.input_size != self.output_size:
            self.skip_connection = nn.Linear(input_size, self.output_size)
        else:
            self.skip_connection = None
        
        # Gating layer
        self.gate = nn.Linear(input_size, self.output_size)
        
        # Layer normalization
        if layer_norm:
            self.layer_norm = nn.LayerNorm(self.output_size)
    
    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the GRN.
        
        Args:
            x: Input tensor [batch_size, ..., input_size]
            context: Optional context tensor [batch_size, ..., context_size]
            
        Returns:
            Output tensor [batch_size, ..., output_size]
        """
        # Main branch
        hidden = self.fc1(x)
        
        # Add context if provided
        if context is not None and self.context_size is not None:
            context_projection = self.context_projection(context)
            hidden = hidden + context_projection
            
        hidden = F.elu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)
        
        # Skip connection
        if self.skip_connection is not None:
            skip = self.skip_connection(x)
        else:
            skip = x
            
        # Gating mechanism
        gate = torch.sigmoid(self.gate(x))
        output = gate * hidden + (1 - gate) * skip
        
        # Apply layer normalization if specified
        if hasattr(self, 'layer_norm') and self.layer_norm:
            output = self.layer_norm(output)
            
        return output


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network from the TFT paper.
    This module selects the most relevant input variables at each time step.
    """
    def __init__(self, input_sizes: Dict[str, int], hidden_size: int, dropout: float = 0.1, 
                 context_size: int = None):
        super().__init__()
        self.input_sizes = input_sizes
        self.hidden_size = hidden_size
        self.context_size = context_size
        
        # Create a GRN for each input variable
        self.feature_grns = nn.ModuleDict({
            name: GatedResidualNetwork(
                input_size=size, 
                hidden_size=hidden_size, 
                output_size=hidden_size, 
                dropout=dropout
            ) for name, size in input_sizes.items()
        })
        
        # Create a GRN for the variable selection weights
        # The input size is the sum of all input variable sizes
        self.weight_grn = GatedResidualNetwork(
            input_size=sum(input_sizes.values()),
            hidden_size=hidden_size,
            output_size=len(input_sizes),
            dropout=dropout,
            context_size=context_size
        )
        
    def forward(self, inputs: Dict[str, torch.Tensor], context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Variable Selection Network.
        
        Args:
            inputs: Dictionary of input tensors {variable_name: tensor}
                   Each tensor has shape [batch_size, ..., input_size]
            context: Optional context tensor [batch_size, ..., context_size]
            
        Returns:
            Tuple of (processed_variables, attention_weights)
        """
        # Apply GRN to each input variable
        var_outputs = {name: grn(inputs[name]) for name, grn in self.feature_grns.items()}
        
        # Concatenate all inputs for the weight GRN
        combined = torch.cat(list(inputs.values()), dim=-1)
        
        # Get variable selection weights using the weight GRN
        weights = self.weight_grn(combined, context)
        weights = F.softmax(weights, dim=-1)
        
        # Weight the processed variables
        var_list = list(var_outputs.values())
        weighted_sum = torch.zeros_like(var_list[0])
        
        # Apply attention weights to each variable
        for i, var_tensor in enumerate(var_list):
            # Extract the weight for this variable
            var_weight = weights[..., i:i+1]
            # Apply the weight
            weighted_sum += var_weight * var_tensor
            
        return weighted_sum, weights


class TemporalFusionTransformer(pl.LightningModule):
    """
    Temporal Fusion Transformer model for time series forecasting.
    """
    def __init__(
        self,
        num_static_vars: int = 0,
        num_time_varying_categorical_vars: int = 0,
        num_time_varying_real_vars: int = 24,  # From metadata
        hidden_size: int = 32,  # Reduced from 64 to 32
        attention_head_size: int = 4,
        dropout: float = 0.1,
        num_lstm_layers: int = 1,  # Reduced from 2 to 1
        num_attention_heads: int = 2,  # Reduced from 4 to 2
        learning_rate: float = 1e-3,
        context_size: int = 32,  # Reduced from 64 to 32
        window_size: int = 30,  # From metadata
        output_size: int = 1,   # RUL prediction
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        # Model parameters
        self.num_static_vars = num_static_vars
        self.num_time_varying_categorical_vars = num_time_varying_categorical_vars
        self.num_time_varying_real_vars = num_time_varying_real_vars
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.num_lstm_layers = num_lstm_layers
        self.num_attention_heads = num_attention_heads
        self.learning_rate = learning_rate
        self.context_size = context_size
        self.window_size = window_size
        self.output_size = output_size
        
        # Initialize variable selection networks
        
        # For time-varying real variables
        if num_time_varying_real_vars > 0:
            self.time_varying_real_vsn = VariableSelectionNetwork(
                input_sizes={f'real_{i}': 1 for i in range(num_time_varying_real_vars)},
                hidden_size=hidden_size,
                dropout=dropout,
                context_size=context_size if num_static_vars > 0 else None
            )
        
        # LSTM encoder (past)
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # LSTM decoder (future)
        self.lstm_decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Gated residual networks for processing
        self.post_lstm_gate = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            dropout=dropout
        )
        
        self.post_attn_gate = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            dropout=dropout
        )
        
        # Final output layer
        self.output_layer = nn.Sequential(
            GatedResidualNetwork(
                input_size=hidden_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout
            ),
            nn.Linear(hidden_size, output_size)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the TFT model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, num_features]
            
        Returns:
            Output tensor of shape [batch_size, output_size]
        """
        batch_size, seq_len, num_features = x.shape
        
        # Process time-varying real variables
        # Reshape to [batch_size, seq_len, num_features, 1] for variable selection
        time_varying_real = x.unsqueeze(-1)
        
        # Create dictionary of inputs for the variable selection network
        real_inputs = {f'real_{i}': time_varying_real[..., i, :] for i in range(self.num_time_varying_real_vars)}
        
        # Apply variable selection network across time
        time_varying_embeddings = []
        variable_weights = []
        
        for t in range(seq_len):
            # Get inputs at current time step
            real_inputs_t = {k: v[:, t] for k, v in real_inputs.items()}
            
            # Apply variable selection
            embedding, weight = self.time_varying_real_vsn(real_inputs_t)
            
            time_varying_embeddings.append(embedding)
            variable_weights.append(weight)
            
        # Stack across time dimension
        time_varying_embeddings = torch.stack(time_varying_embeddings, dim=1)  # [batch_size, seq_len, hidden_size]
        variable_weights = torch.stack(variable_weights, dim=1)  # [batch_size, seq_len, num_vars]
        
        # LSTM encoding
        lstm_output, _ = self.lstm_encoder(time_varying_embeddings)
        
        # Apply gated residual network
        lstm_output = self.post_lstm_gate(lstm_output)
        lstm_output = self.norm1(lstm_output)
        
        # Self-attention layer
        attn_output, _ = self.multihead_attn(
            query=lstm_output,
            key=lstm_output,
            value=lstm_output
        )
        
        # Apply gated residual network to attention output
        attn_output = self.post_attn_gate(attn_output)
        attn_output = self.norm2(attn_output)
        
        # Take the last time step for prediction
        output = attn_output[:, -1]
        
        # Final output layer
        predictions = self.output_layer(output)
        
        return predictions
    
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