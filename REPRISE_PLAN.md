# CW Glyph - Plan de reprise

## Objectif

Transformer CW Glyph en decodeur Morse experimental mesurable, capable de
progresser sur du Morse reel heterogene sans confondre les erreurs de
segmentation, de classification et de reconstruction du texte.

Le projet doit rester glyph-first: on reconnait d'abord des formes temporelles
de caracteres Morse, puis on reconstruit le texte. Les etapes ci-dessous
servent a rendre cette approche debuggable et comparable.

## Principe directeur

Ne pas reentrainer a l'aveugle.

Chaque experience doit produire:

- la commande exacte;
- le commit Git;
- le checkpoint utilise ou produit;
- les parametres de segmentation;
- les metriques globales;
- des exemples d'erreurs inspectables segment par segment.

## Phase 0 - Infrastructure

Etat actuel:

- depot local: `projects/cw-glyph`;
- calcul GPU: pod RunPod A5000;
- connexion SSH locale: `ssh runpod-cw-glyph`;
- depot distant RunPod: `/workspace/cw-glyph`;
- environnement RunPod: `.venv` avec PyTorch CUDA deja fourni par l'image;
- smoke test RunPod: `65 passed`, mini checkpoint CUDA ecrit.

Regle:

- developpement leger et revue sur Ormind;
- entrainement, sweeps et evaluation lourde sur RunPod;
- donnees reelles et checkpoints dans `/workspace`, pas dans Git.

## Phase 1 - Baseline reproductible

But: obtenir une reference propre avant toute modification de modele.

Actions:

1. Entrainer un checkpoint synthetique `real` avec la commande du README.
2. Evaluer la classification isolee sur plusieurs regimes:
   - propre;
   - SNR 5-25 dB;
   - QSB;
   - QRN;
   - drift/filtres RX.
3. Evaluer la segmentation synthetique independamment du CNN.
4. Sauver les resultats JSON dans `outputs/experiments/<date>/`.

Critere de sortie:

- un checkpoint de reference;
- un rapport classification;
- un rapport segmentation;
- une commande unique pour reproduire les deux.

## Phase 2 - Corpus reel minimal

But: disposer d'un petit banc de test reel stable.

Structure attendue sur RunPod:

```text
/workspace/cw-glyph/livetests/
  source-name/
    sample.wav
    sample.txt
```

Actions:

1. Rassembler 5 a 20 extraits WAV courts avec labels texte.
2. Normaliser les labels avec les conventions existantes (`<SK>`, `<KN>`,
   ponctuation, espaces).
3. Definir un split manuel:
   - `debug`: quelques extraits pour iterer;
   - `validation`: mesure de progression;
   - `holdout`: jamais utilise pour regler les seuils.

Critere de sortie:

- `eval_livetest.py` tourne sur tous les fichiers;
- chaque fichier produit decoded text, CER et alignement.

## Phase 3 - Diagnostic segment par segment

But: savoir si chaque erreur vient de la segmentation ou du classifieur.

Actions:

1. Produire un rapport par WAV:
   - waveform;
   - spectrogramme;
   - enveloppe;
   - keydowns detectes;
   - segments candidats;
   - top-k CNN par segment;
   - alignement reference/hypothese.
2. Ajouter un resume JSON:
   - substitutions frequentes;
   - insertions;
   - suppressions;
   - segments rejetes par score ou duree;
   - unit_ms estime.
3. Identifier les cas dominants:
   - fusion de caracteres;
   - split d'un caractere;
   - faux keydown bruit;
   - dit perdu par QSB;
   - confusion CNN sur segment correct.

Critere de sortie:

- les 10 plus grosses erreurs sont inspectables visuellement;
- on peut classer les erreurs par cause.

## Phase 4 - Amelioration segmentation

But: reduire insertions, suppressions, splits et merges avant de toucher au CNN.

Pistes prioritaires:

1. Estimation locale du dot-unit, pas seulement globale.
2. Seuil adaptatif mieux calibre par fenetre.
3. Detection de regions actives plus robuste au QSB.
4. Split des segments trop longs avec cout base sur les gaps internes.
5. Estimation d'espace mot par units locaux.

Critere de sortie:

- baisse du CER reel sur validation;
- baisse separee des erreurs segmentation;
- pas de regression sur segmentation synthetique.

## Phase 5 - Amelioration modele

But: ameliorer la reconnaissance des glyphes quand les segments sont corrects.

Pistes:

1. Durcir le generateur synthetique avec les patterns observes en reel.
2. Tester `unit` vs `stretch` avec le meme protocole.
3. Comparer CNN 1D actuel avec une baseline spectrogramme 2D.
4. Calibrer les scores de confiance.
5. Eventuellement fine-tuner sur segments reels valides.

Critere de sortie:

- classification isolee robuste;
- top-k utile sur erreurs reelles;
- scores mieux calibres pour rejet/ambiguite.

## Phase 6 - Boucle de livraison

But: rendre le projet exploitable.

Actions:

1. Commande `train_reference`.
2. Commande `eval_reference`.
3. Commande `report_wav`.
4. Documentation courte pour RunPod.
5. Checkpoints et rapports ranges par date.

Critere de sortie:

- une nouvelle session peut reproduire les resultats en moins de 15 minutes;
- le README indique le chemin heureux actuel.

## Prochaine action

Mettre en place un protocole d'experience minimal:

```text
outputs/experiments/<timestamp>/
  command.txt
  environment.txt
  train.log
  eval_synth.json
  segmentation_synth.json
```

Puis lancer un premier checkpoint de reference sur RunPod.
