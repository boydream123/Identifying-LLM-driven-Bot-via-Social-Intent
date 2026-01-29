# train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config import SIFTConfig
from dataset import MockSocialDataset
from models.sift_model import SIFT
from core.losses import SIFTLoss

def train():
    # 1. Initialize Configuration
    cfg = SIFTConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running SIFT on device: {device}")

    # 2. Prepare Data
    print("Initializing Dataset...")
    train_dataset = MockSocialDataset(num_samples=1000, seq_len=cfg.seq_len, input_dim=cfg.input_dim)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    # 3. Initialize Model and Loss
    print("Initializing Model...")
    model = SIFT(cfg).to(device)
    criterion = SIFTLoss(margin=cfg.separation_margin, lambda_val=cfg.lambda_loss)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)

    # 4. Training Loop
    model.train()
    print("Start Training...")
    
    for epoch in range(cfg.num_epochs):
        total_loss_avg = 0
        cls_loss_avg = 0
        man_loss_avg = 0
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            
            # Zero Gradients
            optimizer.zero_grad()
            
            # Forward Pass
            logits, alpha, beta = model(features)
            
            # Loss Calculation
            loss, l_cls, l_man = criterion(logits, alpha, beta, labels)
            
            # Backward Pass
            loss.backward()
            optimizer.step()
            
            # Logging
            total_loss_avg += loss.item()
            cls_loss_avg += l_cls.item()
            man_loss_avg += l_man.item()

        # Print Epoch Stats
        print(f"Epoch [{epoch+1}/{cfg.num_epochs}] "
              f"Total Loss: {total_loss_avg/len(train_loader):.4f} | "
              f"BCE: {cls_loss_avg/len(train_loader):.4f} | "
              f"Manifold: {man_loss_avg/len(train_loader):.4f}")

    print("Training Complete.")

if __name__ == "__main__":
    train()
