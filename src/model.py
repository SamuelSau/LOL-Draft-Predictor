import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
league_stats_file_path = os.path.join(PROJECT_ROOT, "stats/league_stats.json")
league_constants_file_path = os.path.join(PROJECT_ROOT, "constants/league_constants.json")

#safety check for opening the constants file
if not os.path.exists(league_stats_file_path):
    print(f"Error: {league_stats_file_path} not found. Please ensure it exists in the current directory.")
    sys.exit(1)

with open(league_constants_file_path, "r") as f:
    constants = json.load(f)

# ========== CONFIG ========== #
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
LEARNING_RATE = 0.0001
EPOCHS = 20
BATCH_SIZE = 32

MATCH_FILE = "league_games.npz"
ROLES = ["top", "jungle", "mid", "bot", "support"]
CHAMPION_LIST = constants["CHAMPION_LIST"]
NUM_CHAMPIONS = len(set(CHAMPION_LIST))
SAVE_PLOTS = True

# ========== LOAD DATA ========== #
def load_dataset_from_npz(path):
    """Load and extract league_games.npz if it exist"""
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    try:
        data = np.load(path)
        return data['X'], data['y']
    except Exception as e:
        print(f"Failed to load {path}: {str(e)}")
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

file_size_bytes = os.path.getsize(MATCH_FILE)
file_size_mb = file_size_bytes / (1024 * 1024)
file_size_gb = file_size_bytes / (1024 * 1024 * 1024)

print(f"File size of '{MATCH_FILE}': {file_size_mb:.2f} MB ({file_size_gb:.2f} GB)")

data = np.load(MATCH_FILE)

X, y = load_dataset_from_npz(MATCH_FILE)

# Label distribution diagnostic
unique, counts = np.unique(y, return_counts=True)
print(f"Label distribution: {dict(zip(unique, counts))}")

if len(X) == 0 or len(y) == 0:
    print("Error: Empty dataset. Exiting.")
    sys.exit(1)

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# ========== SPLIT DATA ========== #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

# ========== MODEL ========== #
class DraftWinPredictor(nn.Module):
    def __init__(self, input_size):
        super(DraftWinPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x)

input_size = X_train.shape[1]
model = DraftWinPredictor(input_size).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ========== TRAINING ========== #
train_losses = []
val_accuracies = []

start_train_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(X_train.size(0))
    epoch_loss = 0.0

    for i in range(0, X_train.size(0), BATCH_SIZE):
        indices = perm[i:i + BATCH_SIZE]
        batch_x, batch_y = X_train[indices], y_train[indices]
        batch_y = batch_y.view(-1, 1)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    train_losses.append(epoch_loss)

    model.eval()
    val_preds_all = []
    with torch.no_grad():
        for i in range(0, X_test.size(0), BATCH_SIZE):
            batch_x = X_test[i:i + BATCH_SIZE]
            batch_logits = model(batch_x)
            batch_probs = torch.sigmoid(batch_logits)
            val_preds_all.append(batch_probs)

        val_preds = torch.cat(val_preds_all, dim=0).cpu().view(-1)
        y_test_cpu = y_test.cpu().view(-1)
        val_preds_class = (val_preds > 0.5).float()
        val_accuracy = (val_preds_class == y_test_cpu).float().mean().item()
        val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# ========== FINAL EVALUATION ========== #
with torch.no_grad():
    preds_all = []
    for i in range(0, X_test.size(0), BATCH_SIZE):
        batch_x = X_test[i:i + BATCH_SIZE]
        batch_logits = model(batch_x)
        batch_probs = torch.sigmoid(batch_logits)
        preds_all.append(batch_probs)

    probs = torch.cat(preds_all, dim=0).cpu().view(-1)
    preds_class = (probs > 0.5).float()
    y_test_cpu = y_test.cpu().view(-1)
    accuracy = (preds_class == y_test_cpu).float().mean().item()
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")

    y_true = y_test.cpu().numpy().flatten()
    y_pred = preds_class.cpu().numpy().flatten()
    cm = confusion_matrix(y_true, y_pred)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Blue Loss", "Blue Win"]))


# ========== PREDICTION SAMPLE DEBUG ========== #
with torch.no_grad():
    sample = X_test[0].unsqueeze(0)  # Single match input
    output = model(sample)
    prob = torch.sigmoid(output).item()
    print("\nInspecting First Sample:")
    print(f"Predicted win probability for blue team: {prob:.4f}")
    print(f"Prediction: {'Blue Win ✅' if prob > 0.5 else 'Blue Loss ❌'}")

# ========== VISUALIZATION ========== #
os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS + 1), train_losses, marker='o')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS + 1), val_accuracies, marker='o', color='green')
plt.title('Validation Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)

plt.tight_layout()
plt.savefig('plots/training_metrics.png')
plt.close()

# Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Blue Loss", "Blue Win"],
            yticklabels=["Blue Loss", "Blue Win"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('plots/confusion_matrix.png')
plt.close()

print("\n📊 Visualizations saved to 'plots/'")

