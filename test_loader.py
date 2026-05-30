# test_loader.py
import yaml
import torch
from data.loader import create_dataloaders

print("=" * 60)
print("TEST DU CHARGEMENT DES DONNÉES")
print("=" * 60)

# Charger la configuration
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

print(f"\nData root: {config['paths']['data_root']}")
print(f"Batch size: {config['training']['batch_size']}")
print(f"Device: {config['training']['device']}")

# Créer les dataloaders
print("\nCréation des dataloaders...")
train_loader, dev_loader = create_dataloaders(config)

print(f"\nTrain: {len(train_loader)} batches")
print(f"Dev:   {len(dev_loader)} batches")

# Tester un batch
print("\nTest du premier batch...")
for batch in train_loader:
    print(f"   Waveform shape: {batch['waveform'].shape}")
    print(f"   LFCC shape: {batch['lfcc'].shape}")
    print(f"   Labels: {batch['label'][:5].tolist()}")
    print(f"   Labels uniques: {torch.unique(batch['label']).tolist()}")
    print(f"   Batch valide !")
    break

print("\n" + "=" * 60)
print("TEST RÉUSSI - Prêt pour l'entraînement !")
print("=" * 60)