# Système Hybride de Détection de Deepfakes Vocaux

Ce projet est une solution logicielle avancée et modulaire dédiée à la classification et à la détection de falsifications vocales (clonage de voix et synthèse vocale par IA).

Développé sous l'écosystème **PyTorch**, ce système implémente une architecture hybride conçue pour le jeu de données officiel **ASVspoof 2019 Logical Access**.

Le modèle fusionne deux approches complémentaires :

* **Analyse end-to-end de l'audio brut** via un réseau neuronal profond basé sur l'architecture **AASIST** (*Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks*).
* **Analyse spectrale ciblée** via l'extraction de coefficients cepstraux sur échelle de fréquences linéaires (**LFCC**), particulièrement efficace pour capturer les artefacts à haute fréquence générés par les systèmes de synthèse vocale (*Text-to-Speech*).

---

## Architecture du Projet

```text
deepfake-voice-detection/
│
├── config.yaml
│   # Configuration globale (chemins, hyperparamètres)
│
├── requirements.txt
│   # Dépendances Python
│
├── README.md
│   # Documentation du projet
│
├── test_loader.py
│   # Diagnostic du pipeline de données
│
├── DATAFLAC/
│   ├── ASVspoof2019_LA_cm_protocols/
│   │   # Protocoles officiels et labels
│   │
│   ├── ASVspoof2019_LA_train/
│   │   # Données d'entraînement (~25 380 fichiers)
│   │
│   └── ASVspoof2019_LA_dev/
│       # Données de validation (~24 986 fichiers)
│
├── data/
│   ├── __init__.py
│   ├── loader.py
│   │   # Dataset PyTorch + collate_fn dynamique
│   └── preprocessing.py
│       # Prétraitement audio (normalisation, trim, padding cyclique)
│
├── features/
│   ├── __init__.py
│   └── lfcc.py
│       # Extraction des caractéristiques LFCC (filtres linéaires + DCT)
│
├── models/
│   ├── __init__.py
│   ├── aasist.py
│   │   # Encodeur RawNet2 (SincConv + blocs résiduels + GRU bidirectionnel)
│   │   # Couches HSGAL (Graph Attention Layers)
│   └── fusion.py
│       # Fusion AASIST + LFCC avec classification binaire
│
├── training/
│   ├── __init__.py
│   ├── train.py
│   │   # Boucle d'entraînement (early stopping, gradient clipping)
│   └── metrics.py
│       # EER (Equal Error Rate) et t-DCF
│
└── inference/
    └── detect.py
        # Détection sur fichier audio avec seuil optimal
```

---

## Spécifications Techniques

Le projet intègre plusieurs mécanismes destinés à améliorer la robustesse, la stabilité et les performances du système.

## Robustesse de Décodage Multiplateforme

L'utilisation systématique de :

```python
encoding = "utf-8"
```

lors de l'ouverture des fichiers élimine les problèmes de décodage observés sous Windows (CP1252).

## Indexation Fiable des Échantillons

Les protocoles ASVspoof sont analysés via :

```python
parts[-1]
```

afin de récupérer les labels (`bonafide` ou `spoof`) indépendamment du nombre de colonnes présentes dans les fichiers texte.

## Padding Temporel Dynamique

Une fonction personnalisée `collate_fn` permet de traiter des matrices LFCC de longueurs variables au sein d'un même batch.

Cette approche évite les erreurs de dimensions provoquées par :

* la suppression des silences ;
* les variations naturelles de durée des fichiers audio.

## Seuil de Décision Optimisé

Le script d'inférence charge automatiquement le seuil calculé lors de la validation au point **EER (Equal Error Rate)**.

Ce seuil remplace la valeur arbitraire `0.5` et reflète davantage le comportement statistique réel du modèle.

## Équilibrage de la Fonction de Perte

Le dataset ASVspoof présente une forte asymétrie :

* beaucoup plus d'exemples `spoof` (deepfake) ;
* beaucoup moins d'exemples `bonafide` (authentique).

Pour corriger ce biais :

```python
pos_weight = n_fake / n_real
```

est injecté dans :

```python
BCEWithLogitsLoss(pos_weight=...)
```

afin de pénaliser davantage les erreurs sur les voix authentiques.

## Padding Cyclique

Les signaux audio inférieurs à 4 secondes sont prolongés par répétition cyclique :

```text
ABCDEF → ABCDEFABCDEF...
```

Cette stratégie est généralement plus naturelle pour les réseaux neuronaux que le *zero-padding* :

```text
ABCDEF → ABCDEF000000...
```

## Gradient Clipping

Pour stabiliser l'entraînement et éviter l'explosion des gradients :

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=5.0
)
```

Cette technique est particulièrement utile pour les architectures profondes comme AASIST.

## Cosine Annealing Scheduler

Le taux d'apprentissage est ajusté automatiquement via :

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs
)
```

Cela permet d'affiner progressivement les poids du modèle.

## Early Stopping

L'entraînement s'arrête automatiquement si l'EER ne s'améliore plus pendant un certain nombre d'époques (*patience*).

Avantages :

* réduction du surentraînement (*overfitting*) ;
* diminution du temps de calcul ;
* meilleure généralisation.

## Vérification d'Existence des Fichiers

Chaque fichier audio est validé avant son ajout au dataset :

```python
if file_path.exists():
    self.data.append(...)
```

Cette vérification évite les erreurs `FileNotFoundError`.

## Cohérence des Dimensions du Modèle

| Composant          | Entrée      | Sortie      |
| ------------------ | ----------- | ----------- |
| SincConv           | (B, 1, T)   | (B, 64, T)  |
| GRU bidirectionnel | (B, T, 64)  | (B, T, 128) |
| HSGAL              | (B, T, 128) | (B, T, 128) |
| Projection AASIST  | (B, 128)    | (B, 160)    |
| Branche LFCC       | (B, T', 60) | (B, 128)    |
| Fusion             | (B, 288)    | (B, 1)      |

---

## Guide d'Utilisation

Le système fonctionne en **inférence différée** :

1. entraînement du modèle ;
2. sauvegarde des poids ;
3. audit indépendant de fichiers audio.

## 1. Installation

```bash
cd C:/Users/nelly/Desktop/DEEP

python -m venv venv

.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## 2. Validation du Pipeline

```bash
python test_loader.py
```

Si tout est correctement configuré :

```text
TEST RÉUSSI - Prêt pour l'entraînement !
```

## 3. Entraînement du Modèle

```bash
python training/train.py
```

Le moteur d'entraînement applique :

* Gradient Clipping ;
* CosineAnnealingLR ;
* BCE pondérée (`pos_weight`) ;
* Sauvegarde automatique du meilleur modèle.

À chaque époque sont calculés :

* la Loss ;
* l'EER ;
* les métriques de validation.

Le meilleur modèle est sauvegardé dans :

```text
checkpoints/best_model.pth
```

Le seuil optimal associé au meilleur EER est également conservé.

## 4. Détection sur un Fichier Audio

```bash
python inference/detect.py "chemin/vers/fichier.flac"
```

Le script :

1. charge le modèle entraîné ;
2. extrait les caractéristiques LFCC ;
3. applique l'inférence ;
4. affiche le verdict.

### Exemple de sortie

```text
Modele charge
  EER entrainement : 1.13%
  Seuil de decision : 0.6104
  Device : cpu

=======================================================
  Fichier : chemin/vers/fichier.flac
  Score   : 0.9412  (0=fake, 1=reel)
  Seuil   : 0.6104
  Verdict : AUTHENTIQUE (Bonafide)
=======================================================
```

---

## Métriques Académiques

## Equal Error Rate (EER)

L'**Equal Error Rate** correspond au point où :

* FAR (*False Acceptance Rate*) ;
* FRR (*False Rejection Rate*).

deviennent égaux :

```text
FAR = FRR
```

Dans les systèmes biométriques modernes, un EER inférieur à **2 %** est généralement considéré comme performant.

## Tandem Detection Cost Function (t-DCF)

La **t-DCF** est la métrique officielle du challenge ASVspoof.

Elle mesure l'impact réel du module anti-spoofing lorsqu'il est intégré à un système de vérification vocale.

Formule :

```python
dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
```

Les paramètres de coût privilégient généralement :

```text
c_miss >> c_fa
```

afin de minimiser le risque associé aux fraudes non détectées.

---

## Architecture AASIST — Détails Techniques

## SincConv (Filtres Sinc Paramétrables)

Les filtres SincConv sont des filtres passe-bande apprenables initialisés à partir de fréquences Mel.

```python
band_pass = 2 * band * sinc(t) * cos(2 * π * low * t)
```

### Avantages

* moins de paramètres ;
* meilleure interprétabilité ;
* meilleure généralisation.

## RawNet2 Encoder

L'encodeur RawNet2 combine :

* SincConv ;
* BatchNorm + ReLU ;
* 6 blocs résiduels ;
* GRU bidirectionnel.

## HSGAL (Hierarchical Spectro-Temporal Graph Attention Layer)

Chaque couche applique :

* Multi-Head Self-Attention ;
* LayerNorm ;
* Dropout.

Le mécanisme permet au modèle de se concentrer sur les régions temporelles les plus discriminantes.

---

## Caractéristiques LFCC

## Pourquoi LFCC plutôt que MFCC ?

| Caractéristique               | MFCC                | LFCC          |
| ----------------------------- | ------------------- | ------------- |
| Échelle des filtres           | Mel (logarithmique) | Linéaire      |
| Sensibilité hautes fréquences | Faible              | Élevée        |
| Artefacts TTS                 | Peu visibles        | Bien capturés |

Les systèmes TTS produisent généralement des artefacts perceptibles dans les hautes fréquences.

L'échelle linéaire des LFCC conserve une résolution constante sur tout le spectre, contrairement à l'échelle Mel.

## Pipeline d'Extraction

```text
Audio (1, T)
    ↓ STFT
Spectrogramme (n_freq, n_frames)
    ↓ Filtres linéaires
FilterBank (n_filters, n_frames)
    ↓ Log
LogEnergy (n_filters, n_frames)
    ↓ DCT
LFCC (n_ceps, n_frames)
    ↓ Transposition
LFCC (n_frames, n_ceps)
```

---

## Objectifs du Projet

Construire un système robuste capable de détecter :

* les voix clonées ;
* les synthèses Text-to-Speech (TTS) ;
* les attaques de spoofing vocal modernes ;

tout en conservant une faible erreur de classification sur les voix authentiques.

L'architecture hybride **AASIST + LFCC** combine :

* les capacités end-to-end d'AASIST pour capturer les anomalies temporelles globales ;
* la sensibilité spectrale des LFCC pour détecter les micro-artefacts haute fréquence.

