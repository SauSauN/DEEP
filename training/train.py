# training/train.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import yaml
import numpy as np
from tqdm import tqdm

from data.loader      import create_dataloaders
from models.fusion    import HybridFusionModel
from training.metrics import calculate_eer


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Une epoque d'entrainement complete.

    Pour chaque batch :
      1. Forward pass  -> logits
      2. Calcul loss   -> BCEWithLogitsLoss
      3. Backward pass -> gradients
      4. Clip gradients -> stabilite numerique
      5. Optimizer step -> mise a jour des poids

    Returns:
        loss moyenne sur l'epoque
    """
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        waveform = batch['waveform'].to(device)
        lfcc     = batch['lfcc'].to(device)
        labels   = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(waveform, lfcc).squeeze(1)  # (B,)
        loss    = criterion(outputs, labels)
        loss.backward()

        # Evite l'explosion des gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate(model, dev_loader, criterion, device):
    """
    Evaluation sur le dev set.

    Returns:
        (loss, eer, threshold_optimal)
    """
    model.eval()
    total_loss = 0
    all_scores = []
    all_labels = []

    for batch in tqdm(dev_loader, desc="Evaluating", leave=False):
        waveform = batch['waveform'].to(device)
        lfcc     = batch['lfcc'].to(device)
        labels   = batch['label'].to(device)

        outputs = model(waveform, lfcc).squeeze(1)
        loss    = criterion(outputs, labels)
        total_loss += loss.item()

        scores = torch.sigmoid(outputs)
        all_scores.extend(scores.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    eer, threshold = calculate_eer(np.array(all_scores), np.array(all_labels))
    return total_loss / len(dev_loader), eer, threshold


def main():
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Device : utilise GPU si disponible, sinon CPU
    device_str = config['training']['device']
    device     = torch.device(
        device_str if device_str == 'cpu' or torch.cuda.is_available() else 'cpu'
    )

    print(f"\nVoiceGuard - Entrainement")
    print(f"  Device  : {device}")
    print(f"  Epochs  : {config['training']['epochs']}")
    print(f"  Batch   : {config['training']['batch_size']}")

    # Chargement des donnees
    print("\nChargement des donnees...")
    train_loader, dev_loader = create_dataloaders(config)

    # Construction du modele
    print("\nConstruction du modele...")
    model = HybridFusionModel(
        lfcc_dim         = config['fusion']['lfcc_dim'],
        # CORRECTION : lit fusion.aasist_dim = 160 (et non aasist.graph_nodes = 128)
        aasist_embed_dim = config['fusion']['aasist_dim'],
        hidden_dim       = config['fusion']['hidden_dim'],
        dropout          = config['fusion']['dropout'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parametres : {n_params:,}")

    # Poids de classes
    # ASVspoof 2019 : ~9x plus de fakes que de reels
    # Sans pos_weight, le modele apprend a tout predire "spoof"
    n_real     = sum(1 for d in train_loader.dataset.data if d['label'] == 1)
    n_fake     = sum(1 for d in train_loader.dataset.data if d['label'] == 0)
    pos_weight = torch.tensor([n_fake / max(n_real, 1)], device=device)
    print(f"  Pos weight : {pos_weight.item():.2f}  (reels={n_real}, fakes={n_fake})")

    # Loss, optimiseur, scheduler
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(
        model.parameters(),
        lr           = config['training']['learning_rate'],
        weight_decay = 1e-4,
    )
    # Reduit le learning rate progressivement -> meilleure convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['epochs']
    )

    # Variables de suivi
    best_eer         = float('inf')
    best_threshold   = 0.5
    patience_counter = 0

    Path(config['paths']['checkpoints_dir']).mkdir(exist_ok=True)

    print(f"\nDebut entrainement - {config['training']['epochs']} epoques\n")

    for epoch in range(1, config['training']['epochs'] + 1):

        # Entrainement
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluation
        dev_loss, eer, threshold = evaluate(model, dev_loader, criterion, device)

        # Mise a jour du scheduler
        scheduler.step()

        print(f"Epoch {epoch:03d}/{config['training']['epochs']} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Dev Loss: {dev_loss:.4f} | "
              f"EER: {eer:.2f}%")

        # Sauvegarde si meilleur EER
        if eer < best_eer:
            best_eer       = eer
            best_threshold = threshold
            patience_counter = 0

            checkpoint_path = Path(config['paths']['checkpoints_dir']) / 'best_model.pth'
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'eer':                  eer,
                'threshold':            threshold,  # Sauvegarde pour detect.py
                'config':               config,
            }, checkpoint_path)
            print(f"  Meilleur modele sauvegarde (EER: {eer:.2f}%, seuil: {threshold:.4f})")

        else:
            patience_counter += 1
            if patience_counter >= config['training']['early_stopping_patience']:
                print(f"\nEarly stopping a l'epoque {epoch}")
                break

    print(f"\nEntrainement termine | Meilleur EER : {best_eer:.2f}%")
    print(f"Modele sauvegarde dans : {config['paths']['checkpoints_dir']}/best_model.pth")


if __name__ == '__main__':
    main()