# Simple LSTM Model for RUL Prediction

This is a simpler alternative to the Temporal Fusion Transformer (TFT) model for Remaining Useful Life (RUL) prediction. The simple LSTM model requires less memory and computational resources while still providing reasonable performance.

## Model Architecture

The Simple LSTM model consists of:
- A single LSTM layer to process sequential sensor data
- A linear output layer for RUL prediction

This architecture is much simpler than the TFT model but is more suitable for environments with limited computational resources.

## Files

- `simple_lstm_model.py`: Implementation of the Simple LSTM model using PyTorch Lightning
- `train_simple_lstm.py`: Script for training the Simple LSTM model

## Usage

### Training

To train the Simple LSTM model, run:

```bash
python models/train_simple_lstm.py --max_epochs 10 --batch_size 32
```

Additional arguments:
- `--data_dir`: Directory containing the preprocessed data (default: 'transformer_data')
- `--batch_size`: Batch size for training and validation (default: 32)
- `--hidden_size`: Hidden size for the model (default: 32)
- `--dropout`: Dropout rate (default: 0.1)
- `--num_layers`: Number of LSTM layers (default: 1)
- `--learning_rate`: Learning rate (default: 1e-3)
- `--max_epochs`: Maximum number of epochs (default: 10)
- `--patience`: Patience for early stopping (default: 5)

### Full Pipeline

To run the full pipeline (data preparation, model training, and evaluation):

```bash
python models/run_pipeline.py
```

Additional arguments:
- `--window_size`: Window size for sequence data (default: 30)
- `--horizon`: Prediction horizon (default: 1)
- `--batch_size`: Batch size for training (default: 32)
- `--hidden_size`: Hidden size for the model (default: 32)
- `--max_epochs`: Maximum number of epochs (default: 10)
- `--skip_data_prep`: Skip data preparation step
- `--skip_training`: Skip model training step
- `--skip_evaluation`: Skip model evaluation step

## Performance

The Simple LSTM model achieves the following performance on the CMAPSS dataset:
- **RMSE (Root Mean Squared Error)**: ~71.97
- **MAE (Mean Absolute Error)**: ~70.58
- **Precision@25_cycles**: ~0.24 (24% of predictions are within 25 cycles of the true RUL)

While not as accurate as more complex models like TFT, this simple LSTM model provides a good baseline and can run on systems with limited resources. 