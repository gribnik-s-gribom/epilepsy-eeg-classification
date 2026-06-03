import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from .losses import FocalLoss

def train_epoch(model, loader, criterion, optimizer, clinical=False):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        if clinical:
            x, c, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            out = model(x, c)
        else:
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x)

        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total

def evaluate(model, loader, criterion, clinical=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    y_true, y_prob = [], []
    with torch.no_grad():
        for batch in loader:
            if clinical:
                x, c, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                out = model(x, c)
            else:
                x, y = batch[0].to(device), batch[1].to(device)
                out = model(x)

            total_loss += criterion(out, y).item() * x.size(0)
            prob = out.softmax(1)[:, 1]
            correct += (prob.argmax(1) == y).sum().item()
            total += y.size(0)
            y_true.extend(y.cpu().numpy())
            y_prob.extend(prob.cpu().numpy())

    return total_loss / total, correct / total, np.array(y_true), np.array(y_prob)

def train_model(model, train_ds, val_ds, name, clinical=False, config=None):
    if config is None:
        config = Config()

    sampler = WeightedRandomSampler(
        1.0 / np.bincount(train_ds.labels.numpy())[train_ds.labels.numpy()],
        len(train_ds), replacement=True
    )
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    best_val_loss = float('inf')
    for epoch in range(config.NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, clinical)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, clinical)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{name}_best.pth')

        print(f"{name} Epoch {epoch+1:2d}: Train Loss={train_loss:.4f} Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f} Acc={val_acc:.4f}")

    model.load_state_dict(torch.load(f'{name}_best.pth'))
    return model
