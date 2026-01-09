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
INPUT_SIZE = 37
OUTPUT_SIZE = 7
HIDDEN_LAYER_1 = 128
HIDDEN_LAYER_2 = 64
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 32


class TrumpSelectionDNN(nn.Module):

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


def load_and_prepare_data(data_path):
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
    trump_col = ['trump']

    data.columns = cards + forehand + user_col + trump_col

    data[cards + forehand] = data[cards + forehand].astype(bool)
    X = data[cards + forehand].values.astype(np.float32)

    data['trump'] = data['trump'].replace(10, 6)

    y = data['trump'].values.astype(np.int64)

    print(f"Loaded {len(X)} total samples. X shape: {X.shape}, Y shape: {y.shape}")
    if X.shape[1] != INPUT_SIZE:
        raise ValueError(f"Input feature count mismatch: expected {INPUT_SIZE}, got {X.shape[1]}")

    if np.max(y) != OUTPUT_SIZE - 1 or np.min(y) < 0:
        raise ValueError(
            f"Output class index range mismatch: expected 0-{OUTPUT_SIZE - 1}, got min={np.min(y)} max={np.max(y)}")

    X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    X_train_tensor = torch.from_numpy(X_train).float()
    Y_train_tensor = torch.from_numpy(Y_train).long()
    X_val_tensor = torch.from_numpy(X_val).float()
    Y_val_tensor = torch.from_numpy(Y_val).long()

    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    return train_loader, val_loader


def train_model(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
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

        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                Y_logits = model(X_batch)
                loss = criterion(Y_logits, Y_batch)
                val_loss += loss.item() * X_batch.size(0)

                _, predicted = torch.max(Y_logits, 1)
                correct_predictions += (predicted == Y_batch).sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = correct_predictions / len(val_loader.dataset)

        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | '
                  f'Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}')

    print("Training finished.")


if __name__ == '__main__':
    data_path = Path(__file__).resolve().parent.parent / 'data/2018_10_18_trump.csv'
    train_loader, val_loader = load_and_prepare_data(data_path)

    model = TrumpSelectionDNN()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Architecture: 37 -> {HIDDEN_LAYER_1} -> {HIDDEN_LAYER_2} -> 7")
    print(f"Total trainable parameters: {num_params:,}")

    train_model(model, train_loader, val_loader)

    MODEL_DIR = Path("models")
    MODEL_DIR.mkdir(exist_ok=True)
    MODEL_PATH = MODEL_DIR / "dnn_trump_model.pth"

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nSuccessfully saved model weights to: {MODEL_PATH}")