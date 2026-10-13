"""
Dermatology Disease Classification Model
- Trains the PyTorch model on the dermatology dataset
- Saves the state_dict for inference
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ──────────────────────── Model Definition ────────────────────────
class DermatologyClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(34, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


# ──────────────────────── Feature Columns ────────────────────────
COLUMNS = [
    'erythema', 'scaling', 'definite_borders', 'itching',
    'koebner_phenomenon', 'polygonal_papules', 'follicular_papules',
    'oral_mucosal_involvement', 'knee_and_elbow_involvement',
    'scalp_involvement', 'family_history', 'melanin_incontinence',
    'eosinophils_infiltrate', 'PNL_infiltrate', 'fibrosis_papillary_dermis',
    'exocytosis', 'acanthosis', 'hyperkeratosis', 'parakeratosis',
    'clubbing_rete_ridges', 'elongation_rete_ridges',
    'thinning_suprapapillary_epidermis', 'spongiform_pustule',
    'munro_microabcess', 'focal_hypergranulosis',
    'disappearance_granular_layer', 'vacuolisation_damage_basal_layer',
    'spongiosis', 'saw_tooth_appearance_retes', 'follicular_horn_plug',
    'perifollicular_parakeratosis', 'inflammatory_mononuclear_infiltrate',
    'band_like_infiltrate', 'age', 'class'
]

CLASS_NAMES = {
    0: "Psoriasis (Sedef Hastalığı)",
    1: "Seborrheic Dermatitis (Seboreik Dermatit)",
    2: "Lichen Planus (Liken Planus)",
    3: "Pityriasis Rosea (Gül Hastalığı)",
    4: "Chronic Dermatitis (Kronik Dermatit)",
    5: "Pityriasis Rubra Pilaris",
}


# ──────────────────────── Training Script ────────────────────────
if __name__ == "__main__":
    import os

    # Load dataset
    csv_path = os.path.join(os.path.dirname(__file__), "dermatology_databse_1.csv")
    df = pd.read_csv(csv_path, header=None, names=COLUMNS)

    # Handle missing values (age column may contain '?')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df.dropna()

    X = df[COLUMNS[:-1]].values
    y = df[COLUMNS[-1]].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long).squeeze() - 1
    y_test = torch.tensor(y_test, dtype=torch.long).squeeze() - 1

    # Initialize model
    model = DermatologyClassificationModel()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    epochs = 120
    for epoch in range(epochs):
        model.train()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        pred = torch.softmax(logits, dim=1).argmax(dim=1)
        acc = (pred == y_train).float().mean().item() * 100

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            test_logits = model(X_test)
            test_loss = loss_fn(test_logits, y_test)
            test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
            test_acc = (test_pred == y_test).float().mean().item() * 100

        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch:>3d} | "
                f"Loss: {loss:.5f} | Acc: {acc:.2f}% | "
                f"Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.2f}%"
            )

    print(f"\n[OK] Final Test Accuracy: {test_acc:.2f}%")

    # Save model state_dict
    save_path = os.path.join(os.path.dirname(__file__), "model_state_dict.pth")
    torch.save(model.state_dict(), save_path)
    print(f"[OK] Model saved to: {save_path}")
