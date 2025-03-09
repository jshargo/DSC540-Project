import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from datetime import datetime
import re


class EnhancedTimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, attention_size=32, num_layers=2, output_size=1, dropout=0.3):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention_size = attention_size

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, attention_size),  # *2 for bidirectional
            nn.Tanh(),
            nn.Linear(attention_size, 1),
            nn.Softmax(dim=1)
        )

        # Fully connected layers with residual connections
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 32)
        self.fc3 = nn.Linear(32, output_size)

        # Use GroupNorm instead of BatchNorm (works with batch size of 1)
        self.bn1 = nn.GroupNorm(min(32, hidden_size), hidden_size)
        self.bn2 = nn.GroupNorm(min(8, 32), 32)

    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)

        # Initial hidden state
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)

        # Forward propagate the LSTM
        out, _ = self.lstm(x, (h0, c0))  # out shape: [batch_size, seq_length, hidden_size*2]

        # Attention mechanism
        attention_weights = self.attention(out)  # [batch_size, seq_length, 1]
        context_vector = torch.sum(attention_weights * out, dim=1)  # [batch_size, hidden_size*2]

        # Apply fully connected layers with residual connections and group normalization
        fc1_out = self.fc1(context_vector)
        fc1_out = self.bn1(fc1_out)  # GroupNorm works with any batch size
        fc1_out = self.relu(fc1_out)
        fc1_out = self.dropout(fc1_out)

        fc2_out = self.fc2(fc1_out)
        fc2_out = self.bn2(fc2_out)  # GroupNorm works with any batch size
        fc2_out = self.relu(fc2_out)
        fc2_out = self.dropout(fc2_out)

        output = self.fc3(fc2_out)

        return output


class NBAPointsPredictor:
    def __init__(self, player_name, csv_file=None, sequence_length=10, multiple_seq_lengths=False):
        """
        Initialize the NBA Points Predictor using CSV file data

        Args:
            player_name: Name of the player to predict points for
            csv_file: Path to the CSV file with player data (if None, will look for [PlayerName].csv)
            sequence_length: Number of previous games to use for prediction
            multiple_seq_lengths: Whether to use multiple sequence lengths
        """
        self.player_name = player_name
        self.csv_file = csv_file or f"{player_name.replace(' ', '')}.csv"
        self.sequence_length = sequence_length
        self.multiple_seq_lengths = multiple_seq_lengths

        if multiple_seq_lengths:
            self.sequence_lengths = [5, 10, 20]  # Use multiple sequence lengths
        else:
            self.sequence_lengths = [sequence_length]

        self.scaler = StandardScaler()
        self.model = None
        self.processed_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # For tracking experiments
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{player_name.replace(' ', '_')}_{self.timestamp}"

        # Create folders if they don't exist
        os.makedirs('models', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        os.makedirs('logs', exist_ok=True)

    def load_data(self):
        """
        Load player data from CSV file
        """
        try:
            # Load the CSV file
            df = pd.read_csv(self.csv_file)
            print(f"Successfully loaded data from {self.csv_file}")
            print(f"Data shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")

            # Display a sample of the data
            print("\nData sample:")
            print(df.head())

            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                print("\nMissing values by column:")
                print(missing_values[missing_values > 0])

            # Store the data
            self.raw_df = df
            return df

        except FileNotFoundError:
            print(f"Error: CSV file '{self.csv_file}' not found.")
            print(f"Make sure to run the scraper first to generate this file.")
            raise
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            raise

    def preprocess_data(self):
        """
        Preprocess the loaded CSV data for model training
        """
        if self.raw_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Make a copy to avoid modifying the original
        df = self.raw_df.copy()

        # Check data types and convert if needed
        numeric_columns = ['PTS', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%',
                           'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'MP', 'Rest']

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert minutes played to numeric if it's in the format "MM:SS"
        if 'MP' in df.columns:
            try:
                # Check if MP is in format "MM:SS"
                sample_mp = str(df['MP'].iloc[0]) if not pd.isna(df['MP'].iloc[0]) else ""
                if ':' in sample_mp:
                    df['MP'] = df['MP'].apply(lambda x:
                                              float(str(x).split(':')[0]) +
                                              float(str(x).split(':')[1]) / 60
                                              if pd.notna(x) and ':' in str(x) else x)
            except:
                print("Warning: Could not convert MP column to numeric minutes")

        # Create additional features

        # 1. Home/Away indicator (if not already in the CSV)
        if 'HomeGame' not in df.columns and '@' in df.values:
            # Check if there's a column that contains location info
            for col in df.columns:
                sample_values = df[col].astype(str).head(10).tolist()
                if any('@' in str(val) for val in sample_values):
                    print(f"Detected location in column '{col}', creating HomeGame feature")
                    df['HomeGame'] = df[col].apply(lambda x: 0 if str(x).strip() == '@' else 1)
                    break

        # 2. Feature engineering: Add rolling averages and trends
        # Ensure the data is sorted chronologically
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')

        # Create rolling features
        rolling_windows = [3, 5, 10]
        for window in rolling_windows:
            if len(df) > window:
                df[f'PTS_rolling_{window}'] = df['PTS'].rolling(window=window).mean()
                df[f'PTS_rolling_std_{window}'] = df['PTS'].rolling(window=window).std()

                # Offensive efficiency metrics
                if all(col in df.columns for col in ['PTS', 'FGA', 'FTA']):
                    df[f'PointsPerShot_rolling_{window}'] = df['PTS'].rolling(window=window).sum() / (
                                df['FGA'].rolling(window=window).sum() + 0.5 * df['FTA'].rolling(window=window).sum())

                # Last N games trend (positive or negative)
                df[f'PTS_trend_{window}'] = df['PTS'].rolling(window=window).apply(
                    lambda x: 1 if (x.iloc[-1] > x.iloc[0]) else -1, raw=False
                )

        # Fill NaN values in rolling features
        for col in df.columns:
            if 'rolling' in col or 'trend' in col:
                df[col] = df[col].fillna(df[col].mean() if not pd.isna(df[col].mean()) else 0)

        # Fill remaining NaN values
        df = df.fillna(0)

        # Store processed dataframe
        self.processed_df = df

        print(f"Processed data shape: {df.shape}")
        print(f"Final columns: {df.columns.tolist()}")

        return df

    def create_sequences(self, data, target_column='PTS'):
        """
        Create sequences from time series data for RNN input
        """
        # Define features to use
        feature_columns = [
            'PTS', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%',
            'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'MP', 'Opp', 'Rest'
        ]

        # Add home/away indicator if it exists
        if 'HomeGame' in data.columns:
            feature_columns.append('HomeGame')

        # Add rolling features if they exist
        rolling_features = [col for col in data.columns if 'rolling' in col or 'trend' in col]
        feature_columns.extend(rolling_features)

        # Keep only columns that actually exist in the data
        feature_columns = [col for col in feature_columns if col in data.columns]

        # Ensure target column is included
        if target_column not in feature_columns:
            feature_columns.append(target_column)

        # Store feature columns for later use
        self.feature_columns = feature_columns

        # Create a dataframe with only the selected features
        data_features = data[feature_columns].copy()

        # Final check for NaN values
        if data_features.isna().any().any():
            print("WARNING: There are still NaN values in the data. Setting them to 0.")
            data_features = data_features.fillna(0)

        # Scale features
        scaled_data = self.scaler.fit_transform(data_features)
        scaled_df = pd.DataFrame(scaled_data, columns=feature_columns)

        # If using multiple sequence lengths, create sequences for each length
        if self.multiple_seq_lengths:
            X_multi, y_multi = [], []

            for seq_length in self.sequence_lengths:
                X_seq, y_seq = self._create_single_sequence(scaled_df, seq_length, target_column)

                # Only add if we got valid sequences
                if len(X_seq) > 0:
                    X_multi.append(X_seq)
                    y_multi.append(y_seq)

            # Find the minimum number of sequences across all lengths
            min_sequences = min([len(x) for x in X_multi])

            # Trim all sequences to the same length
            X_multi = [x[-min_sequences:] for x in X_multi]
            y_multi = y_multi[-1][-min_sequences:]  # Use the target from the last sequence length

            # Combine sequences of different lengths
            X_combined = []
            for i in range(min_sequences):
                seq_features = []
                for seq_idx, seq_length in enumerate(self.sequence_lengths):
                    seq_features.append(X_multi[seq_idx][i])
                X_combined.append(np.concatenate(seq_features, axis=0))

            return np.array(X_combined), np.array(y_multi)
        else:
            # Use single sequence length
            return self._create_single_sequence(scaled_df, self.sequence_length, target_column)

    def _create_single_sequence(self, scaled_df, seq_length, target_column):
        """
        Helper method to create sequences for a single sequence length
        """
        X, y = [], []
        target_idx = scaled_df.columns.get_loc(target_column)

        for i in range(len(scaled_df) - seq_length):
            X.append(scaled_df.iloc[i:i + seq_length].values)
            y.append(scaled_df.iloc[i + seq_length, target_idx])

        if len(X) == 0:
            print(f"Warning: Could not create sequences with length {seq_length}. Not enough data.")
            return [], []

        return np.array(X), np.array(y)

    def build_model(self, input_size, hidden_size=64):
        """
        Build the LSTM model with attention mechanism

        Args:
            input_size: Number of features in the input data
            hidden_size: Size of the hidden layers (default: 64)
        """
        # Scale hidden size based on input size to prevent overfitting on small datasets
        if len(self.X_train) < 50:  # Very small dataset
            adjusted_hidden_size = min(hidden_size, 32)
            num_layers = 2
            dropout = 0.2
        elif len(self.X_train) < 100:  # Small dataset
            adjusted_hidden_size = min(hidden_size, 48)
            num_layers = 2
            dropout = 0.25
        else:  # Larger dataset
            adjusted_hidden_size = hidden_size
            num_layers = 3
            dropout = 0.3

        print(f"\nBuilding model with:")
        print(f"- Input size: {input_size}")
        print(f"- Hidden size: {adjusted_hidden_size}")
        print(f"- Layers: {num_layers}")
        print(f"- Dropout: {dropout}")

        model = AttentionLSTM(
            input_size=input_size,
            hidden_size=adjusted_hidden_size,
            attention_size=adjusted_hidden_size // 2,
            num_layers=num_layers,
            output_size=1,
            dropout=dropout
        ).to(self.device)

        print(model)
        self.model = model
        return model

    def train_model(self, epochs=200, batch_size=32, learning_rate=0.001, lr_scheduler=True, patience=25):
        """
        Train the model with early stopping and learning rate scheduling

        Args:
            epochs: Maximum number of training epochs
            batch_size: Training batch size
            learning_rate: Initial learning rate
            lr_scheduler: Whether to use learning rate scheduling
            patience: Number of epochs to wait for improvement before early stopping
        """
        if self.model is None or self.X_train is None:
            raise ValueError("Model not built or data not prepared")

        # Create dataset
        train_dataset = EnhancedTimeSeriesDataset(self.X_train, self.y_train)

        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)

        # For early stopping
        best_val_loss = float('inf')
        patience = 15  # More patience
        counter = 0

        # For storing training history
        train_losses = []
        val_losses = []

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)

        # Learning rate scheduler
        if lr_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )

        # Training loop with cross-validation
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            # Create a new set of train/val splits for each epoch
            # Add verbose output to diagnose early stopping
            epoch_val_losses = []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(train_dataset)):
                # Create train and validation sub-datasets
                train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
                val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

                # Ensure we don't have a batch size of 1 (min batch size of 2)
                effective_batch_size = min(batch_size, max(2, len(train_idx) // 10))
                val_batch_size = min(batch_size, max(2, len(val_idx) // 10))

                # Print batch size info for debugging
                if epoch == 0 and fold == 0:
                    print(f"Train set size: {len(train_idx)}, Validation set size: {len(val_idx)}")
                    print(f"Using batch sizes - Train: {effective_batch_size}, Validation: {val_batch_size}")

                train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, sampler=train_subsampler)
                val_loader = DataLoader(train_dataset, batch_size=val_batch_size, sampler=val_subsampler)

                # Train on this fold
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                    # Forward pass
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)

                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()

                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    optimizer.step()

                    train_loss += loss.item()

                # Validation on this fold
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        outputs = self.model(X_batch)
                        loss = criterion(outputs, y_batch)
                        val_loss += loss.item()

                # Update learning rate based on validation loss
                if lr_scheduler:
                    scheduler.step(val_loss)

                # Add fold validation loss to list for averaging
                epoch_val_losses.append(val_loss / len(val_loader))

            # Calculate average losses
            # Modified to avoid division by zero and provide more stable metrics
            num_folds = len(list(tscv.split(train_dataset)))
            avg_train_loss = train_loss / max(1, len(train_loader) * num_folds)
            avg_val_loss = sum(epoch_val_losses) / max(1, len(epoch_val_losses))

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            # Print progress with more detailed information
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

            # Early stopping
            if avg_val_loss < best_val_loss:
                improvement = (best_val_loss - avg_val_loss) / best_val_loss * 100
                best_val_loss = avg_val_loss
                counter = 0
                # Save the best model
                safe_filename = self.player_name.replace(" ", "_")
                torch.save(self.model.state_dict(), f'models/{safe_filename}_lstm_model_{self.timestamp}.pth')
                print(f'✓ Validation loss improved by {improvement:.2f}%, saving model')
            else:
                counter += 1
                print(f'✗ No improvement for {counter}/{patience} epochs')
                if counter >= patience:
                    print(f'Early stopping at epoch {epoch + 1} - No improvement for {patience} epochs')
                    break

        # Load the best model
        safe_filename = self.player_name.replace(" ", "_")
        try:
            self.model.load_state_dict(torch.load(f'models/{safe_filename}_lstm_model_{self.timestamp}.pth'))
        except FileNotFoundError:
            print(f"Warning: Could not find saved model file. Using current model state.")

        # Plot training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'{self.player_name} - Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'plots/{safe_filename}_training_loss_{self.timestamp}.png')

        return {'train_losses': train_losses, 'val_losses': val_losses}

    def evaluate_model(self):
        """
        Evaluate the model performance with visualizations
        """
        if self.model is None or self.X_test is None:
            raise ValueError("Model not built or data not prepared")

        # Create test dataset and dataloader
        test_dataset = EnhancedTimeSeriesDataset(self.X_test, self.y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Switch to evaluation mode
        self.model.eval()

        # Make predictions
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(y_batch.numpy())

        # Convert to numpy arrays
        y_pred = np.array(all_preds).flatten()
        y_test = np.array(all_targets).flatten()

        # Find the PTS index among the feature columns
        if hasattr(self, 'feature_columns') and 'PTS' in self.feature_columns:
            pts_idx = self.feature_columns.index('PTS')
        else:
            print("Warning: 'PTS' column not found in feature columns")
            pts_idx = 0

        # Create placeholder arrays with the correct size
        num_features = len(self.feature_columns) if hasattr(self, 'feature_columns') else len(self.X_test[0][0])
        inverse_placeholder = np.zeros((len(y_pred), num_features))
        inverse_actual_placeholder = np.zeros((len(y_test), num_features))

        # Set the values at the correct index
        inverse_placeholder[:, pts_idx] = y_pred
        inverse_actual_placeholder[:, pts_idx] = y_test

        # Inverse transform
        y_pred_actual = self.scaler.inverse_transform(inverse_placeholder)[:, pts_idx]
        y_test_actual = self.scaler.inverse_transform(inverse_actual_placeholder)[:, pts_idx]

        # Calculate metrics
        mse = mean_squared_error(y_test_actual, y_pred_actual)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_actual, y_pred_actual)
        r2 = r2_score(y_test_actual, y_pred_actual)

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }

        print("Model Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        # Additional metric: Percentage of predictions within 5 points
        within_5_points = np.mean(np.abs(y_pred_actual - y_test_actual) < 5) * 100
        print(f"Predictions within 5 points: {within_5_points:.2f}%")

        # Enhanced visualizations
        safe_filename = self.player_name.replace(" ", "_")

        # Plot actual vs predicted with improved styling
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_actual, 'b-', label='Actual Points', linewidth=2)
        plt.plot(y_pred_actual, 'r--', label='Predicted Points', linewidth=2)
        plt.title(f'{self.player_name} - Points Prediction', fontsize=14)
        plt.xlabel('Game Index', fontsize=12)
        plt.ylabel('Points', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f'plots/{safe_filename}_predictions_{self.timestamp}.png')

        # Plot a scatter plot with regression line
        plt.figure(figsize=(10, 10))

        # Add perfect prediction line
        min_val = min(min(y_test_actual), min(y_pred_actual))
        max_val = max(max(y_test_actual), max(y_pred_actual))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

        # Add scatter plot with color gradient by density
        scatter = plt.scatter(y_test_actual, y_pred_actual,
                              alpha=0.6, edgecolor='k', s=80)

        # Add regression line
        z = np.polyfit(y_test_actual, y_pred_actual, 1)
        p = np.poly1d(z)
        plt.plot(y_test_actual, p(y_test_actual), 'b-', label=f'Regression Line (y = {z[0]:.2f}x + {z[1]:.2f})')

        plt.title(f'{self.player_name} - Actual vs Predicted Points', fontsize=14)
        plt.xlabel('Actual Points', fontsize=12)
        plt.ylabel('Predicted Points', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f'plots/{safe_filename}_scatter_{self.timestamp}.png')

        # Error distribution histogram
        errors = y_pred_actual - y_test_actual
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True, bins=20)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title(f'{self.player_name} - Prediction Error Distribution', fontsize=14)
        plt.xlabel('Prediction Error (Predicted - Actual)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'plots/{safe_filename}_error_dist_{self.timestamp}.png')

        # Log detailed results
        with open(f'logs/{safe_filename}_results_{self.timestamp}.txt', 'w') as f:
            f.write(f"Model Evaluation for {self.player_name}\n")
            f.write(f"Timestamp: {self.timestamp}\n\n")
            f.write("Model Configuration:\n")
            f.write(f"Sequence Length: {self.sequence_length}\n")
            f.write(f"Multiple Sequence Lengths: {self.multiple_seq_lengths}\n")
            f.write(f"Feature Columns: {self.feature_columns}\n\n")
            f.write("Evaluation Metrics:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            f.write(f"Predictions within 5 points: {within_5_points:.2f}%\n\n")
            f.write("Detailed Predictions:\n")
            for i in range(len(y_test_actual)):
                f.write(
                    f"Game {i + 1}: Actual={y_test_actual[i]:.1f}, Predicted={y_pred_actual[i]:.1f}, Error={errors[i]:.1f}\n")

        return metrics

    def run_pipeline(self):
        """
        Run the complete modeling pipeline
        """
        print(f"Starting prediction pipeline for {self.player_name}")
        print(f"Experiment ID: {self.experiment_id}")

        # 1. Load data from CSV
        self.load_data()

        # 2. Preprocess the data
        processed_data = self.preprocess_data()

        # 3. Create sequences
        X, y = self.create_sequences(processed_data)
        print(f"Data shape after sequence creation: X: {X.shape}, y: {y.shape}")

        # Check if we have enough data
        if len(X) < 20:
            print(f"\nWARNING: Very small dataset size ({len(X)} samples)")
            print("The model may not train effectively. Consider:")
            print("1. Using a shorter sequence length")
            print("2. Adding more historical data")
            print("3. Reducing model complexity\n")

        # 4. Split the data using time series approach (last 20% for testing)
        # Ensure we have at least a few samples for testing
        min_test_size = min(int(len(X) * 0.2), max(5, int(len(X) * 0.1)))
        train_size = len(X) - min_test_size

        self.X_train, self.X_test = X[:train_size], X[train_size:]
        self.y_train, self.y_test = y[:train_size], y[train_size:]

        print(f"Training set size: {len(self.X_train)}, Testing set size: {len(self.X_test)}")

        # 5. Build the model with appropriate complexity based on dataset size
        input_size = self.X_train.shape[2]  # Number of features
        self.build_model(input_size)

        # 6. Train the model with flexible patience based on dataset size
        # Use longer patience for smaller datasets
        patience = 25 if len(self.X_train) < 50 else 15
        history = self.train_model(patience=patience)

        # 7. Evaluate the model
        metrics = self.evaluate_model()

        # 8. Predict next game
        next_game_prediction = self.predict_next_game()

        print(f"Analysis completed for {self.player_name}")
        print(f"Predicted points for next game: {next_game_prediction['prediction']:.1f}")
        print(
            f"90% confidence interval: [{next_game_prediction['lower_bound']:.1f}, {next_game_prediction['upper_bound']:.1f}]")

        return {
            'metrics': metrics,
            'history': history,
            'next_game_prediction': next_game_prediction['prediction'],
            'prediction_interval': (next_game_prediction['lower_bound'], next_game_prediction['upper_bound'])
        }

    def predict_next_game(self, include_opponent=None, is_home_game=None):
        """
        Predict points for next game with uncertainty estimates
        """
        if self.model is None:
            raise ValueError("Model not trained")

        # Get the most recent games
        processed_data = self.processed_df.copy()

        # Sort by date if available
        if 'Date' in processed_data.columns:
            processed_data = processed_data.sort_values('Date')

        # Get the most recent sequence_length games
        recent_games = processed_data.tail(self.sequence_length)

        if len(recent_games) < self.sequence_length:
            raise ValueError(f"Not enough recent games. Need {self.sequence_length}, but only have {len(recent_games)}")

        # Extract features
        if not hasattr(self, 'feature_columns'):
            raise ValueError("Feature columns not defined. Run pipeline first.")

        feature_columns = self.feature_columns
        available_cols = [col for col in feature_columns if col in recent_games.columns]

        # Create a copy of the last game to modify for the next game prediction
        next_game = recent_games.iloc[-1:].copy()

        # Update opponent if provided
        if include_opponent is not None and 'Opp' in available_cols:
            next_game['Opp'] = include_opponent

        # Update home/away if provided
        if is_home_game is not None and 'HomeGame' in available_cols:
            next_game['HomeGame'] = 1 if is_home_game else 0

        # Create sequence data (combine recent games with the modified next game)
        sequence_data = pd.concat([recent_games.iloc[1:], next_game], ignore_index=True)

        # Scale the data
        sequence_features = sequence_data[available_cols].copy()
        scaled_data = self.scaler.transform(sequence_features)

        # Reshape for model input
        X_pred = torch.tensor(scaled_data, dtype=torch.float32).reshape(1, self.sequence_length,
                                                                        len(available_cols)).to(self.device)

        # Predict - always set model to eval mode for prediction to disable dropout
        self.model.eval()
        with torch.no_grad():
            # Use model in evaluation mode to disable dropout and use running stats for normalization
            y_pred_scaled = self.model(X_pred).cpu().numpy()[0][0]

        # Find the index of PTS column
        if 'PTS' in available_cols:
            pts_idx = available_cols.index('PTS')
        else:
            print("Warning: 'PTS' column not found in available columns")
            pts_idx = 0

        # Create placeholder for inverse transform
        inverse_placeholder = np.zeros((1, len(available_cols)))
        inverse_placeholder[0, pts_idx] = y_pred_scaled

        # Inverse transform to get actual points
        y_pred_actual = self.scaler.inverse_transform(inverse_placeholder)[0, pts_idx]

        # Get prediction confidence interval using Monte Carlo dropout
        # We'll create a modified model with dropout always on for uncertainty estimation
        self.model.train()  # Set to train mode to enable dropout

        predictions = []
        for _ in range(20):  # Run 20 predictions with dropout enabled for better uncertainty estimate
            with torch.no_grad():
                pred = self.model(X_pred).cpu().numpy()[0][0]
                inverse_placeholder[0, pts_idx] = pred
                predictions.append(self.scaler.inverse_transform(inverse_placeholder)[0, pts_idx])

        # Set model back to eval mode for future predictions
        self.model.eval()

        lower_bound = max(0, np.percentile(predictions, 5))  # 5th percentile
        upper_bound = np.percentile(predictions, 95)  # 95th percentile

        print(f"Predicted points for {self.player_name}'s next game: {y_pred_actual:.1f}")
        print(f"90% confidence interval: [{lower_bound:.1f}, {upper_bound:.1f}]")

        if include_opponent is not None:
            print(f"Prediction is for a game against opponent {include_opponent}")
        if is_home_game is not None:
            print(f"Prediction is for a {'home' if is_home_game else 'away'} game")

        return {
            'prediction': y_pred_actual,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }

    def analyze_performance_factors(self):
        """
        Analyze how different factors affect the player's performance
        """
        if self.processed_df is None or len(self.processed_df) < 10:
            raise ValueError("Not enough processed data available")

        print(f"\nAnalyzing performance factors for {self.player_name}")
        df = self.processed_df.copy()

        # 1. Home vs Away performance
        if 'HomeGame' in df.columns:
            home_games = df[df['HomeGame'] == 1]
            away_games = df[df['HomeGame'] == 0]

            if len(home_games) > 0 and len(away_games) > 0:
                home_avg = home_games['PTS'].mean()
                away_avg = away_games['PTS'].mean()

                print(f"\nHome vs Away Performance:")
                print(f"Home games ({len(home_games)}): {home_avg:.1f} points per game")
                print(f"Away games ({len(away_games)}): {away_avg:.1f} points per game")
                print(f"Difference: {home_avg - away_avg:.1f} points")

                # Visualize
                plt.figure(figsize=(10, 6))
                sns.boxplot(x='HomeGame', y='PTS', data=df)
                plt.title(f'{self.player_name} - Home vs Away Performance')
                plt.xticks([0, 1], ['Away', 'Home'])
                plt.xlabel('Game Location')
                plt.ylabel('Points')
                plt.savefig(f'plots/{self.player_name.replace(" ", "_")}_home_away_{self.timestamp}.png')

        # 2. Performance against different opponents
        if 'Opp' in df.columns and len(df) >= 20:
            opp_performance = df.groupby('Opp')['PTS'].agg(['mean', 'count']).reset_index()
            # Only include opponents with at least 2 games
            opp_performance = opp_performance[opp_performance['count'] >= 2].sort_values('mean', ascending=False)

            if len(opp_performance) > 0:
                print("\nPerformance against different opponents:")
                for _, row in opp_performance.iterrows():
                    print(f"Opponent {int(row['Opp'])} ({row['count']} games): {row['mean']:.1f} points per game")

                # Visualize top and bottom 5 opponents
                plt.figure(figsize=(12, 8))

                # Get top 5 and bottom 5 opponents
                top_opps = opp_performance.head(5)
                bottom_opps = opp_performance.tail(5)
                plot_opps = pd.concat([top_opps, bottom_opps])

                # Create subplot
                ax = sns.barplot(x='Opp', y='mean', data=plot_opps)
                plt.title(f'{self.player_name} - Points vs Select Opponents')
                plt.xlabel('Opponent')
                plt.ylabel('Average Points')
                plt.xticks(rotation=45)

                # Add count labels on bars
                for i, p in enumerate(ax.patches):
                    ax.annotate(f"n={plot_opps.iloc[i]['count']}",
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='bottom')

                plt.tight_layout()
                plt.savefig(f'plots/{self.player_name.replace(" ", "_")}_vs_opponents_{self.timestamp}.png')

        # 3. Performance by rest days
        if 'Rest' in df.columns:
            # Group rest days into categories
            df['RestCategory'] = pd.cut(df['Rest'],
                                        bins=[-1, 1, 2, 3, 100],
                                        labels=['0-1 days', '2 days', '3 days', '4+ days'])

            rest_performance = df.groupby('RestCategory')['PTS'].agg(['mean', 'count']).reset_index()

            if len(rest_performance) > 0:
                print("\nPerformance by rest days:")
                for _, row in rest_performance.iterrows():
                    print(f"{row['RestCategory']} ({row['count']} games): {row['mean']:.1f} points per game")

                # Visualize
                plt.figure(figsize=(10, 6))
                sns.barplot(x='RestCategory', y='mean', data=rest_performance)
                plt.title(f'{self.player_name} - Performance by Rest Days')
                plt.xlabel('Rest Between Games')
                plt.ylabel('Average Points')
                plt.tight_layout()
                plt.savefig(f'plots/{self.player_name.replace(" ", "_")}_rest_days_{self.timestamp}.png')

        # 4. Performance trend over season
        if 'Date' in df.columns and 'PTS' in df.columns:
            plt.figure(figsize=(14, 7))

            # Sort by date
            df_sorted = df.sort_values('Date')

            # Plot points per game
            plt.plot(df_sorted['Date'], df_sorted['PTS'], 'b-', alpha=0.5)

            # Add trend line with LOWESS smoothing
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                y_smooth = lowess(df_sorted['PTS'].values, np.arange(len(df_sorted)), frac=0.3)
                plt.plot(df_sorted['Date'], y_smooth[:, 1], 'r-', linewidth=2, label='Trend')
            except:
                # If statsmodels not available, use a simple moving average
                plt.plot(df_sorted['Date'], df_sorted['PTS'].rolling(10, min_periods=1).mean(),
                         'r-', linewidth=2, label='10-game Moving Average')

            plt.title(f'{self.player_name} - Scoring Trend Over Season')
            plt.xlabel('Date')
            plt.ylabel('Points')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'plots/{self.player_name.replace(" ", "_")}_season_trend_{self.timestamp}.png')

        # 5. Most predictive features
        if hasattr(self, 'model') and self.model is not None and hasattr(self, 'X_train') and self.X_train is not None:
            print("\nAttempting to identify most important features for prediction...")

            try:
                # Create a baseline prediction
                feature_importance = {}

                # Skip this analysis if we don't have a properly trained model
                if not hasattr(self, 'feature_columns') or not hasattr(self, 'X_test') or len(self.X_test) == 0:
                    print("Skipping feature importance analysis - not enough data")
                    return

                # Identify which features are most important by permuting them
                # This is a simple approach - for a more robust analysis, consider SHAP values
                baseline_pred = self.evaluate_model()['MAE']

                for i, feature in enumerate(self.feature_columns):
                    # Create a copy of the test data
                    X_test_permuted = self.X_test.copy()

                    # Permute the feature across all sequence positions
                    for seq_pos in range(self.X_test.shape[1]):
                        # Shuffle the feature values for this position in sequence
                        X_test_permuted[:, seq_pos, i] = np.random.permutation(X_test_permuted[:, seq_pos, i])

                    # Convert to tensor
                    X_test_tensor = torch.tensor(X_test_permuted, dtype=torch.float32)
                    y_test_tensor = torch.tensor(self.y_test, dtype=torch.float32).reshape(-1, 1)

                    # Create dataset and dataloader
                    test_dataset = EnhancedTimeSeriesDataset(X_test_permuted, self.y_test)
                    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

                    # Make predictions
                    self.model.eval()
                    all_preds = []
                    all_targets = []

                    with torch.no_grad():
                        for X_batch, y_batch in test_loader:
                            X_batch = X_batch.to(self.device)
                            outputs = self.model(X_batch)
                            all_preds.extend(outputs.cpu().numpy())
                            all_targets.extend(y_batch.numpy())

                    # Calculate permuted MAE
                    y_pred = np.array(all_preds).flatten()
                    y_test = np.array(all_targets).flatten()

                    # Find the PTS index among the feature columns
                    pts_idx = self.feature_columns.index('PTS') if 'PTS' in self.feature_columns else 0

                    # Create placeholder arrays with the correct size
                    num_features = len(self.feature_columns)
                    inverse_placeholder = np.zeros((len(y_pred), num_features))
                    inverse_actual_placeholder = np.zeros((len(y_test), num_features))

                    # Set the values at the correct index
                    inverse_placeholder[:, pts_idx] = y_pred
                    inverse_actual_placeholder[:, pts_idx] = y_test

                    # Inverse transform
                    y_pred_actual = self.scaler.inverse_transform(inverse_placeholder)[:, pts_idx]
                    y_test_actual = self.scaler.inverse_transform(inverse_actual_placeholder)[:, pts_idx]

                    # Calculate MAE
                    permuted_mae = mean_absolute_error(y_test_actual, y_pred_actual)

                    # Feature importance is the difference in error when the feature is permuted
                    feature_importance[feature] = permuted_mae - baseline_pred

                # Sort features by importance
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

                print("\nFeature importance ranking (higher = more important):")
                for feature, importance in sorted_features[:10]:  # Show top 10
                    print(f"{feature}: {importance:.4f}")

                # Visualize top 10 features
                top_features = dict(sorted_features[:10])
                plt.figure(figsize=(12, 8))
                plt.bar(top_features.keys(), top_features.values())
                plt.title(f'{self.player_name} - Feature Importance')
                plt.xlabel('Feature')
                plt.ylabel('Importance (increase in MAE when permuted)')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(f'plots/{self.player_name.replace(" ", "_")}_feature_importance_{self.timestamp}.png')

            except Exception as e:
                print(f"Error in feature importance analysis: {e}")

        print("\nPerformance factor analysis completed")


# Example usage
if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='NBA Player Points Predictor')
    parser.add_argument('--player', type=str, required=True,
                        help='Player name (must match CSV filename without spaces)')
    parser.add_argument('--csv', type=str, help='Custom CSV file path (optional)')
    parser.add_argument('--seq_length', type=int, default=10, help='Sequence length (number of previous games)')
    parser.add_argument('--multi_seq', action='store_true', help='Use multiple sequence lengths')
    parser.add_argument('--opponent', type=int, help='Opponent team ID for next game prediction')
    parser.add_argument('--home', action='store_true', help='Set if next game is a home game')

    args = parser.parse_args()

    print(f"\n{'=' * 50}")
    print(f"NBA Points Predictor - Analyzing {args.player}")
    print(f"{'=' * 50}\n")

    # Initialize the predictor with command line arguments
    predictor = NBAPointsPredictor(
        args.player,
        csv_file=args.csv,
        sequence_length=args.seq_length,
        multiple_seq_lengths=args.multi_seq
    )

    try:
        # Run the complete pipeline
        print("\nTraining and evaluating model...")
        results = predictor.run_pipeline()

        # Analyze performance factors
        print("\nAnalyzing performance factors...")
        predictor.analyze_performance_factors()

        # Make specific prediction if opponent specified
        if args.opponent:
            print(f"\nPredicting points against opponent {args.opponent} ({'home' if args.home else 'away'} game)...")
            game_prediction = predictor.predict_next_game(
                include_opponent=args.opponent,
                is_home_game=args.home
            )

            print(f"\nPrediction for {args.player} vs opponent {args.opponent} ({'home' if args.home else 'away'}):")
            print(f"Points: {game_prediction['prediction']:.1f}")
            print(
                f"90% confidence interval: [{game_prediction['lower_bound']:.1f}, {game_prediction['upper_bound']:.1f}]")

        print(f"\n{'=' * 50}")
        print("Analysis completed successfully")
        print(f"{'=' * 50}\n")

    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback

        traceback.print_exc()