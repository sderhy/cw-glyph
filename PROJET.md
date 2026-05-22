# CW Glyph

## Idee

Construire un systeme de reconnaissance de caracteres Morse a partir de
segments audio courts, en s'inspirant de la reconnaissance de caracteres
manuscrits type EMNIST.

L'analogie de base est:

```text
EMNIST:
image d'un caractere manuscrit -> classifieur -> caractere

CW Glyph:
segment audio d'un caractere Morse -> classifieur -> caractere
```

Le but initial n'est pas de remplacer MorseFormer. Le but est de tester une
approche plus simple et plus interpretable: reconnaitre des caracteres Morse
isoles, puis reconstruire un texte a partir de segments.

## Hypothese

Le Morse porte surtout l'information dans une structure temporelle 1D:

- duree des keydowns;
- duree des silences internes;
- rapport dit/dah;
- rythme global de l'operateur.

Un petit CNN 1D applique a une enveloppe audio devrait donc pouvoir reconnaitre
les caracteres Morse, meme avec du bruit et des variations raisonnables.

## Pipeline vise

```text
audio continu
  -> detection de regions actives
  -> segmentation en caracteres
  -> extraction d'une enveloppe 1D normalisee
  -> CNN 1D
  -> classe caractere
  -> recomposition texte
```

La premiere version peut ignorer l'audio continu et travailler directement sur
des segments propres dont le label est connu.

## MVP

Le MVP doit repondre a une question simple:

> Peut-on reconnaitre un caractere Morse isole a partir de son enveloppe audio
> avec un petit modele 1D?

Perimetre du MVP:

- classes: `A-Z`, `0-9`, et quelques prosigns/ponctuations si disponibles;
- entree: segment audio mono contenant un seul caractere Morse;
- sortie: une classe caractere;
- donnees: generation synthetique avec `morse_synth`;
- modele: petit CNN 1D;
- metrique principale: accuracy par caractere;
- metriques secondaires: matrice de confusion, robustesse par SNR/WPM.

## Donnees synthetiques

Le projet peut reutiliser `morse_synth` comme generateur EMNIST-like audio.

Pour chaque classe, generer de nombreuses variantes:

- vitesse WPM;
- frequence audio;
- amplitude;
- rise/fall time;
- jitter d'element;
- jitter de gap;
- ratio dash/dot;
- bruit AWGN;
- QRN;
- QSB;
- filtre RX;
- silence avant/apres le caractere.

Exemple conceptuel:

```text
A -> ".-" -> 10 000 variantes audio -> label A
B -> "-..." -> 10 000 variantes audio -> label B
...
```

## Representation 1D

Le modele ne devrait pas necessairement consommer l'audio brut. Une entree plus
stable serait:

```text
audio
  -> filtrage autour de la porteuse si necessaire
  -> enveloppe amplitude/energie
  -> resampling vers une longueur fixe
  -> normalisation
  -> CNN 1D
```

Longueurs candidates:

- 128 echantillons d'enveloppe pour un classifieur tres compact;
- 256 ou 512 si les variations WPM/SNR demandent plus de resolution.

## Modele initial

Architecture volontairement petite:

```text
Input: [batch, 1, time]
  -> Conv1d + ReLU + BatchNorm
  -> Conv1d + ReLU + MaxPool
  -> Conv1d + ReLU + MaxPool
  -> global average pooling
  -> Linear
  -> logits caracteres
```

Un modele 2D sur spectrogramme peut etre teste plus tard, mais le point de
depart doit rester 1D pour exploiter la nature temporelle du Morse.

## Segmentation

La classification des caracteres isoles est probablement plus simple que la
segmentation. La segmentation doit donc etre traitee comme un second probleme.

Etapes possibles:

1. Entrainer le classifieur avec des segments parfaitement connus.
2. Tester avec des segments synthetiques decoupes automatiquement.
3. Ajouter une segmentation par enveloppe et seuil adaptatif.
4. Gerer les espaces entre mots.
5. Evaluer sur de l'audio continu.

## Structure proposee

```text
cw-glyph/
  PROJET.md
  pyproject.toml
  README.md
  morse_char_recognizer/
    __init__.py
    dataset.py
    features.py
    model.py
    segment.py
    decode.py
    metrics.py
  scripts/
    generate_dataset.py
    train.py
    eval.py
    decode_audio.py
  tests/
```

## Dependances probables

- `numpy`
- `scipy`
- `torch`
- `torchaudio` optionnel
- `scikit-learn` pour matrice de confusion et rapports
- `matplotlib` pour visualisations

Au debut, aucune carte NVIDIA n'est requise. Le projet doit pouvoir tourner sur
CPU ou Apple Silicon/MPS. CUDA devient utile seulement pour accelerer les
experiences.

## Questions ouvertes

- Faut-il entrainer sur l'enveloppe seule ou sur audio brut filtre?
- Quelle longueur fixe d'enveloppe donne le meilleur compromis?
- Faut-il normaliser le temps par WPM estime, ou laisser le modele apprendre
  les variations de vitesse?
- Quelles classes garder dans la premiere version?
- Quel niveau de bruit viser pour le MVP?
- Comment evaluer proprement la separation classification/segmentation?

## Premier plan d'action

1. Creer le squelette Python du projet.
2. Brancher `morse_synth` comme dependance locale ou copier seulement les
   briques necessaires.
3. Generer un dataset synthetique de caracteres isoles.
4. Implementer l'extraction d'enveloppe 1D.
5. Entrainer un petit CNN 1D.
6. Produire une matrice de confusion.
7. Identifier les confusions naturelles: `E/T`, `I/M`, `S/O`, `A/N`, etc.
8. Ajouter progressivement bruit, jitter et variation WPM.

## Critere de succes initial

Le MVP est interessant si:

- l'accuracy reste elevee sur caracteres isoles propres;
- les confusions sous bruit correspondent a des confusions Morse plausibles;
- le modele reste petit et rapide;
- la segmentation apparait comme le principal probleme restant, plutot que la
  classification elle-meme.
