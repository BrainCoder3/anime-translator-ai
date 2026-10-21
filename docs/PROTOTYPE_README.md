# Anime Translator AI

> Prototype R&D de localisation et doublage automatique de vidéos japonaises vers le français.

Anime Translator AI est un pipeline IA qui vise à transformer automatiquement une vidéo japonaise en version doublée française.

Le projet est actuellement en développement actif.

## Objectif

Pipeline complet :

```text
Vidéo japonaise
→ Extraction audio
→ Transcription japonaise
→ Diarisation
→ Identification des personnages
→ Traduction japonais → français
→ Adaptation temporelle
→ Clonage / génération vocale
→ Séparation voix / musique / bruitages
→ Mixage
→ Vidéo française
```

## Fonctionnalités actuelles

- Transcription avec Faster-Whisper
- Diarisation avec Pyannote Audio
- Alignement transcription / locuteurs
- Mémoire persistante des personnages
- Embeddings et références vocales
- Traduction avec Qwen
- Contrôle qualité automatique
- Adaptation temporelle des dialogues
- TTS multilingue avec Chatterbox
- Contrôle automatique de la durée des voix
- Séparation audio avec Demucs
- Mixage et génération de la vidéo finale
- Cache des étapes déjà calculées

## Architecture

```text
anime-translator-ai/
├── translate_video.py
└── src/
    ├── core/
    ├── audio/
    ├── asr/
    ├── diarization/
    ├── characters/
    ├── translation/
    ├── tts/
    ├── separation/
    └── mix/
```

## Technologies

- Python
- PyTorch
- CUDA
- Faster-Whisper
- Pyannote Audio
- Qwen
- Hugging Face
- Chatterbox Multilingual TTS
- Demucs
- FFmpeg
- Google Colab

## État du projet

### Fonctionnel

- [x] transcription
- [x] diarisation
- [x] mémoire des personnages
- [x] traduction
- [x] génération vocale
- [x] adaptation temporelle
- [x] séparation audio
- [x] génération vidéo finale

### En développement

- [ ] utilisation automatique des sous-titres existants
- [ ] fusion sous-titres + ASR
- [ ] amélioration de la traduction contextuelle
- [ ] synchronisation plus stricte avec les lèvres
- [ ] meilleure gestion des émotions
- [ ] validation multi-épisodes
- [ ] véritable lip-sync vidéo

## Limites

Le projet est encore un prototype. La qualité peut varier selon l'audio, la traduction et la durée des dialogues.

## Avertissement

Projet développé à des fins d'apprentissage, de recherche et d'expérimentation. L'utilisation de contenus audiovisuels doit respecter les droits applicables.
