# Extraction d’information hybride pour la modélisation de l’expérience d’écoute (proof of concept)

Documentation pour reproduire :
- le pipeline d’analyses statistiques
- le pipeline hybride d’extraction d’information (règles + GLiNER2)

---

This project explores how listening experience is transformed into textual information in online music reviews. Drawing on controlled vocabularies, Pauline Oliveros’ Deep Listening framework, and a corpus of 34,689 reviews, it adopts a mixed-methods approach combining statistical analysis with a hybrid information extraction pipeline that integrates rule-based patterns and few-shot Named Entity Recognition (NER) using a bidirectional transformer model (GLiNER2).

The methodology enables the semi-automatic extraction of experiential domains such as body, memory, place, and social relations, by leveraging contextual information in documents. The results show that perceptual descriptors do more than qualify sonic properties: they function as discursive anchors through which listening is mediated, expressing how sound is apprehended as an active, situated experience.

The analysis also highlights structural limits of entity-centric approaches. While nominal and referential domains (e.g. PERSON, PLACE) are reliably captured by NER, domains such as BODY and MEMORY are more often expressed through predications, effects, and relations rather than stable entities. As a next step, the project will experiment with semantic role labeling for BODY and MEMORY in order to model event-like structures (actor–action–effect) and better capture music as an acting force on the perceiving subject.
Overall, the project treats online music reviews as informational inscriptions of listening experience, motivating a plural methodological infrastructure in which statistical, lexical, rule-based, and neural methods each make visible different facets of the same phenomenon.

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
- **Une première approche repose sur des règles lexicales (regex)** À des fins d'exploration, cette approche priorise le rappel (couverture) pour repérer des rôles référentiels génériques. Cette limite motive le recours à GLiNER2 en complément, afin d’identifier des mentions référentielles au-delà de la correspondance exacte. 
- **En complément, une extraction d’entités en few-shot est réalisée avec le modèle GLiNER2** l’extraction d’entités repose sur un échantillon aléatoire stratifié de 200 critiques par source, sous contrainte de longueur (450–650 jetons). Pour chaque critique, trois phrases ont été sélectionnées aléatoirement. L’extraction a été réalisée à l’aide du modèle GLiNER2 small v2.1, avec un seuil global de 0,45, ajusté par seuils spécifiques par catégorie (BODY = 0,50 ; MEMORY = 0,60 ; PLACE = 0,65 ; PERSON = 0,65). Les taux ont été stabilisés par bootstrap (1 000 itérations), l’ensemble du protocole étant rendu reproductible par l’utilisation d’un seed fixe (42).

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
│   ├── Lexicons/
│   │   ├── perceptual_adjectives_complete.json
│   │   └── musico_technical_complete.json
│   ├── outputs/
│      ├── figures/
│      └── tables/
│   ├── features/
│   │   ├── descriptor_extraction.py
│   │   ├── tokenization.py
│   │   └── vectorization_matrices.py
│   ├── reporting/
│   │   ├── descriptor_reports.py
│   │
├── pipeline_experiential/
│   ├── rulebased_patterns.py
│   ├── rulebased_amazonclassicalopera.py
│   └── gliner_extraction.py
