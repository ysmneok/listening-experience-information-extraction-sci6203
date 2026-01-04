# Extraction d’information hybride pour la modélisation de l’expérience d’écoute (proof of concept)

Documentation pour reproduire :
- le pipeline d’analyses statistiques
- le pipeline hybride d’extraction d’information (règles + GLiNER2)

---

Exploring how listening experience is transformed into textual information in online music reviews. Drawing on controlled vocabularies and a corpus of 34,689 reviews online music reviews, it adopts an approach combining statistical analysis with an exploratory hybrid method that integrates rule-based patterns and a few-shot Named Entity Recognition (NER) based on a bidirectional transformer model (GLiNER2). This methodology enables the semi-automatic extraction of experiential domains (body, memory, place, social relations) by leveraging contextual information in documents. The findings show that perceptual descriptors do more than qualify sonic properties: they function as discursive anchors through which listening is mediated, apprehended through the active orientation of the subject toward what the sound affords for the regulation and modulation of lived experience.

---
## Pour reproduire les analyses

```bash
pip install -r requirements.txt
python run.py
```
---


## Vue d’ensemble du pipeline

Le projet est structuré autour de deux pipelines complémentaires, reposant sur un corpus commun:

- **Pipeline d’analyses statistiques** (`pipeline_statistical`)
- **Pipeline hybride d’extraction expérientielle** (`pipeline_experiential`)

---

## Données d’entrée

Le projet repose sur un corpus de 34 689 documents de critiques musicales en ligne, stockées au format JSON (texte des critiques et métadonnées fusionnés).Les métadonnées principales incluent la source, le genre, le texte de la critique et des identifiants de document. Le chargement du corpus comprend des vérifications de format, des contrôles d’intégrité de base et une normalisation des sources et des genres.

## Vocabulaires contrôlés

Deux lexiques construits manuellement structurent l’analyse :
- **Adjectifs perceptifs** (affectifs, expressifs, dynamiques)
- **Descripteurs musico-techniques** (dynamique, tempo, harmonie, articulation, etc.)

Ces vocabulaires sont conçus comme des outils analytiques. Ils permettent d’établir des profils de fréquence, de comparer des distributions par source et par genre, et d’ancrer l’interprétation des résultats statistiques. Chaque document de critique est vectorisé avec DictVectorizer de scikit-learn, produisant des matrices de fréquences par type de descripteur issu des vocabulaires contrôlés.


## Pipeline d’analyses statistiques (pipeline_statistical)

Ce pipeline opère à l’échelle du document et du corpus. Il comprend le chargement et le **filtrage des données**, **l’agrégation des descripteurs par source et par genre**, **l’analyse des distributions (genre × source)**, **des tests statistiques inférentiels (chi carré, résidus standardisés)**, ainsi qu’une **analyse multivariée exploratoire par analyse des correspondances**.

- **Les résultats sont exportés sous forme de tableaux et de figures dans le dossier outputs/.**
  
## Pipeline expérientiel : méthodes hybrides extraction d'information 

Ce pipeline vise l’identification de domaines de médiation de l’expérience d’écoute, au-delà des seuls descripteurs lexicaux.
- **Une première approche repose sur des règles lexicales (regex)**
- **En complément, une extraction d’entités en few-shot est réalisée avec le modèle GLiNER2** pour les catégories BODY, MEMORY, PLACE et PERSON, appliquée à un échantillon aléatoire stratifié (trois phrases par critique), avec des seuils spécifiques par catégorie et une stabilisation des taux par bootstrap.


---
## Structure du dépôt

```text
listening-experience-text-mining/
│
├── config.py
├── requirements.txt
├── run.py
│
├── pipeline_statistical/
│   ├── data/
│   │   └── corpus_loader.py
│   ├── analyses/
│   │   ├── genre_source_distribution.py
│   │   ├── descriptor_aggregation.py
│   │   ├── chi_square_analysis.py
│   │   ├── correspondence_analysis.py
│   │   ├── descriptor_profiles.py
│   │   └── diagnostics.py
│   └── main.py
│
├── pipeline_experiential/
│   ├── rulebased_patterns.py
│   ├── rulebased_amazonclassicalopera.py
│   └── gliner_extraction.py
│
├── Lexicons/
│   ├── perceptual_adjectives_complete.json
│   └── musico_technical_complete.json
│
└── outputs/
    ├── figures/
    └── tables/
