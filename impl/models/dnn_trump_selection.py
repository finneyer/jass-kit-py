import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- Configuration Constants ---
INPUT_SIZE = 37  # 36 cards + 1 forehand indicator
OUTPUT_SIZE = 7  # 7 trump options (4 suits, Obe, Ufe, Push)
HIDDEN_LAYER_1 = 128
HIDDEN_LAYER_2 = 64
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 32


class TrumpSelectionDNN(nn.Module):
    """
    Multilayer Perceptron for Jass trump selection.
    """

    def __init__(self):
        super(TrumpSelectionDNN, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_LAYER_1)
        self.fc2 = nn.Linear(HIDDEN_LAYER_1, HIDDEN_LAYER_2)
        self.fc3 = nn.Linear(HIDDEN_LAYER_2, OUTPUT_SIZE)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# --- 2. Data Loading and Preparation (UPDATED for your data) ---
def load_and_prepare_data(data_path):
    """
    Loads, preprocesses, and converts the Jass trump data into PyTorch DataLoaders.

    CRUCIAL STEP: Merges the 6 and 10 codes (both meaning PUSH) into a single code (6),
    ensuring the final target indices run cleanly from 0 to 6.
    """
    print(f"Loading data from: {data_path}")

    data = pd.read_csv(data_path, header=None)

    cards = [
        # Diamonds
        'DA', 'DK', 'DQ', 'DJ', 'D10', 'D9', 'D8', 'D7', 'D6',
        # Hearts
        'HA', 'HK', 'HQ', 'HJ', 'H10', 'H9', 'H8', 'H7', 'H6',
        # Spades
        'SA', 'SK', 'SQ', 'SJ', 'S10', 'S9', 'S8', 'S7', 'S6',
        # Clubs
        'CA', 'CK', 'CQ', 'CJ', 'C10', 'C9', 'C8', 'C7', 'C6'
    ]

    forehand = ['FH']
    user_col = ['user']
    trump_col = ['trump']  # Renamed for clarity in this scope

    data.columns = cards + forehand + user_col + trump_col

    # 1. Input Features (X)
    data[cards + forehand] = data[cards + forehand].astype(bool)
    X = data[cards + forehand].values.astype(np.float32)

    # 2. Target Variable (Y): Merge 6 and 10 to a single PUSH index (6)
    # The target array must contain continuous indices: 0, 1, 2, 3, 4, 5, 6
    # If the original data uses 10 for PUSH, replace 10 with 6.
    data['trump'] = data['trump'].replace(10, 6)

    # Ensure all remaining values are of integer type and convert to NumPy
    y = data['trump'].values.astype(np.int64)

    # Verify the output sizes
    print(f"Loaded {len(X)} total samples. X shape: {X.shape}, Y shape: {y.shape}")
    if X.shape[1] != INPUT_SIZE:
        raise ValueError(f"Input feature count mismatch: expected {INPUT_SIZE}, got {X.shape[1]}")
    # We check the max value is 6 (PUSH) and the min value is 0 (DIAMONDS)
    if np.max(y) != OUTPUT_SIZE - 1 or np.min(y) < 0:
        raise ValueError(
            f"Output class index range mismatch: expected 0-{OUTPUT_SIZE - 1}, got min={np.min(y)} max={np.max(y)}")

    # Split data (70% train, 30% test/validation)
    # stratify=y is essential for keeping the class distribution balanced
    X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Convert to PyTorch Tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    Y_train_tensor = torch.from_numpy(Y_train).long()
    X_val_tensor = torch.from_numpy(X_val).float()
    Y_val_tensor = torch.from_numpy(Y_val).long()

    # Create DataLoader objects
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    return train_loader, val_loader


# --- 3. Training Function (Same as before) ---
def train_model(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            Y_logits = model(X_batch)
            loss = criterion(Y_logits, Y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # Validation Phase (Monitoring Convergence)
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                Y_logits = model(X_batch)
                loss = criterion(Y_logits, Y_batch)
                val_loss += loss.item() * X_batch.size(0)

                # Calculate accuracy
                _, predicted = torch.max(Y_logits, 1)
                correct_predictions += (predicted == Y_batch).sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = correct_predictions / len(val_loader.dataset)

        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | '
                  f'Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}')

    print("Training finished.")


# --- 4. Main Execution and Saving ---
if __name__ == '__main__':
    # 1. Load Data
    # Ensure this path is correct for your file structure!
    data_path = Path(__file__).resolve().parent.parent / 'data/2018_10_18_trump.csv'
    train_loader, val_loader = load_and_prepare_data(data_path)

    # 2. Instantiate Model
    model = TrumpSelectionDNN()

    # Calculate and report parameter count (Exercise 8 requirement)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Architecture: 37 -> {HIDDEN_LAYER_1} -> {HIDDEN_LAYER_2} -> 7")
    print(f"Total trainable parameters: {num_params:,}")

    # 3. Train Model
    train_model(model, train_loader, val_loader)

    # 4. Save Weights (Best Practice)
    MODEL_DIR = Path("models")
    MODEL_DIR.mkdir(exist_ok=True)
    MODEL_PATH = MODEL_DIR / "dnn_trump_model.pth"

    # Save only the state dictionary for maximum flexibility and portability
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nSuccessfully saved model weights to: {MODEL_PATH}")