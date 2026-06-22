import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("""\
# MediMap AI: Full Training Pipeline
This notebook documents the entire process of loading the original 41-disease dataset, preprocessing it, defining the neural network architecture, and running the training loop.
You can run this notebook directly to train your model on your GPU!
"""),
    
    nbf.v4.new_markdown_cell("## 1. Imports and Setup"),
    nbf.v4.new_code_cell("""\
import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast

# Ensure GPU is used if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Compute Device: {DEVICE}")
"""),

    nbf.v4.new_markdown_cell("## 2. Data Loading & Preprocessing"),
    nbf.v4.new_code_cell("""\
# Load dataset
df = pd.read_csv('../data/raw/tabular/dataset.csv')
df.columns = df.columns.str.strip()
label_col = "Disease"
df[label_col] = df[label_col].astype(str).str.strip()

# Extract symptoms
sym_cols = [c for c in df.columns if c.startswith("Symptom")]
X_df = pd.get_dummies(df[sym_cols], prefix="", prefix_sep="").groupby(level=0, axis=1).max()

# Drop any blank columns created by empty cells
if "" in X_df.columns:
    X_df.drop(columns=[""], inplace=True)
if "NaN" in X_df.columns:
    X_df.drop(columns=["NaN"], inplace=True)

X = X_df.values.astype(np.float32)

# Encode Labels
le = LabelEncoder()
y = le.fit_transform(df[label_col])

print(f"Data Shape: {X.shape[0]} patients, {X.shape[1]} symptoms, {len(le.classes_)} diseases")

# Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
"""),

    nbf.v4.new_markdown_cell("## 3. PyTorch Dataset & DataLoader"),
    nbf.v4.new_code_cell("""\
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(TabularDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TabularDataset(X_val, y_val), batch_size=32, shuffle=False)
"""),

    nbf.v4.new_markdown_cell("## 4. Model Architecture (Tabular Stream)"),
    nbf.v4.new_code_cell("""\
class SymptomNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.mlp(x)

model = SymptomNet(input_dim=X.shape[1], num_classes=len(le.classes_)).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()
"""),

    nbf.v4.new_markdown_cell("## 5. Training Loop"),
    nbf.v4.new_code_cell("""\
epochs = 5 # Just doing 5 for demonstration!
best_val_loss = float('inf')

for epoch in range(1, epochs + 1):
    # Training
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            val_loss += loss.item()
            
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            
    val_acc = correct / len(y_val) * 100
    print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f} - Val Loss: {val_loss/len(val_loader):.4f} - Val Acc: {val_acc:.2f}%")
""")
]

nb['cells'] = cells
os.makedirs('notebooks', exist_ok=True)
nbf.write(nb, 'notebooks/Model_Training_Pipeline.ipynb')
print("Training notebook generated at notebooks/Model_Training_Pipeline.ipynb")
