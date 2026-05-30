# data/loader.py

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from .preprocessing import AudioPreprocessor
from features.lfcc  import LFCCExtractor


class ASVspoofDataset(Dataset):
    """
    Dataset PyTorch pour ASVspoof 2019 Logical Access.

    Format protocole :
        LA_0079 LA_T_1138215 - - bonafide   <- parts[-1] = "bonafide"
        LA_0001 LA_T_1000137 - A07 spoof    <- parts[-1] = "spoof"

    Labels :
        bonafide -> 1  (voix humaine authentique)
        spoof    -> 0  (voix synthetique / deepfake)
    """

    def __init__(self, protocol_file, audio_dir, preprocessor, lfcc_extractor):
        self.audio_dir      = Path(audio_dir)
        self.preprocessor   = preprocessor
        self.lfcc_extractor = lfcc_extractor
        self.data           = []

        print(f"Chargement : {Path(protocol_file).name}")

        with open(protocol_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue

                file_id    = parts[1]
                label_text = parts[-1]        # Toujours le dernier mot
                file_path  = self.audio_dir / f"{file_id}.flac"

                if file_path.exists():
                    label = 1 if label_text == 'bonafide' else 0
                    self.data.append({
                        'file_id':   file_id,
                        'file_path': file_path,
                        'label':     label,
                    })

        n_real = sum(1 for d in self.data if d['label'] == 1)
        n_fake = sum(1 for d in self.data if d['label'] == 0)
        print(f"OK : {len(self.data)} fichiers | authentiques={n_real} | deepfakes={n_fake}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item     = self.data[idx]
        waveform = self.preprocessor.process(str(item['file_path']))  # (1, T)
        lfcc     = self.lfcc_extractor.extract(waveform)              # (T', 60)

        return {
            'waveform': waveform,
            'lfcc':     lfcc,
            'label':    torch.tensor(item['label'], dtype=torch.float32),
        }


def collate_fn(batch):
    """
    Regroupe les samples en batch avec padding LFCC.
    INDISPENSABLE : sans elle torch.stack() crash sur des
    sequences LFCC de longueurs differentes.
    """
    waveforms = torch.stack([b['waveform'] for b in batch])  # (B, 1, T)
    labels    = torch.stack([b['label']    for b in batch])  # (B,)

    max_len  = max(b['lfcc'].shape[0] for b in batch)
    lfcc_dim = batch[0]['lfcc'].shape[1]

    padded = torch.zeros(len(batch), max_len, lfcc_dim)
    for i, b in enumerate(batch):
        padded[i, :b['lfcc'].shape[0], :] = b['lfcc']

    return {'waveform': waveforms, 'lfcc': padded, 'label': labels}


def create_dataloaders(config: dict):
    """Cree les DataLoaders train et dev depuis config.yaml."""

    preprocessor = AudioPreprocessor(
        target_length_sec = config['audio']['target_length_seconds'],
        sample_rate       = config['audio']['sample_rate'],
    )

    lfcc_extractor = LFCCExtractor(
        sample_rate = config['audio']['sample_rate'],
        n_filters   = config['lfcc']['n_filters'],
        n_ceps      = config['lfcc']['n_filters'],
        window_ms   = config['lfcc']['window_ms'],
        hop_ms      = config['lfcc']['hop_ms'],
    )

    data_root = Path(config['paths']['data_root'])
    proto_dir = data_root / config['paths']['protocols_dir']

    train_dataset = ASVspoofDataset(
        protocol_file  = proto_dir / 'ASVspoof2019.LA.cm.train.trn.txt',
        audio_dir      = data_root / config['paths']['train_dir'],
        preprocessor   = preprocessor,
        lfcc_extractor = lfcc_extractor,
    )

    dev_dataset = ASVspoofDataset(
        protocol_file  = proto_dir / 'ASVspoof2019.LA.cm.dev.trl.txt',
        audio_dir      = data_root / config['paths']['dev_dir'],
        preprocessor   = preprocessor,
        lfcc_extractor = lfcc_extractor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = config['training']['batch_size'],
        shuffle     = True,
        num_workers = 0,           # 0 = compatible Windows + CPU
        collate_fn  = collate_fn,  # INDISPENSABLE
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size  = config['training']['batch_size'],
        shuffle     = False,
        num_workers = 0,
        collate_fn  = collate_fn,
    )

    return train_loader, dev_loader