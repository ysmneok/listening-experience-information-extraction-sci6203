# Hybrid Information Extraction / Modeling the Listening Experience 

Documentation to reproduce:
- the statistical analysis pipeline
- the hybrid information-extraction pipeline (rules + GLiNER2)

---

This article investigates how listening experience is transformed into textual information in online music reviews. Drawing on controlled vocabularies and a corpus of 34,689 online music reviews, it adopts an approach combining statistical analysis with an exploratory hybrid method that integrates rule-based patterns and a few-shot Named Entity Recognition (NER) based on a bidirectional transformer model (GLiNER2). This methodology enables the semi-automatic extraction of experiential domains (body, memory, place, social relations) by leveraging contextual information in documents. The findings show that perceptual descriptors do more than qualify sonic properties: they function as discursive anchors through which listening is mediated, apprehended through the active orientation of the subject toward what the sound affords for the regulation and modulation of lived experience.

Following Pauline Oliveros's distinction between hearing and listening, and between focal and global modes of attention (Deep Listening, 2005), these descriptors are read as traces of differentiated attentional regimes. Editorial reviews lean toward a focal attention that isolates sonic objects and formal detail, while experiential reviews lean toward a global attention centered on atmosphere and overall effect — showing that the same shared vocabulary supports distinct ways of orienting attention to sound.

The analysis also highlights structural limits of entity-centric approaches. While nominal and referential domains (e.g. PERSON, PLACE) are reliably captured by NER, domains such as BODY and MEMORY are more often expressed through predications, effects, and relations rather than stable entities. As a next step, we will experiment with semantic role labeling for BODY and MEMORY in order to model event-like structures (actor–action–effect) and better capture music as an acting force on the perceiving subject. Overall, the project treats online music reviews as informational inscriptions of listening experience, motivating a plural methodological infrastructure in which statistical, lexical, rule-based, and neural methods each make visible different facets of the same phenomenon.

---
## Pour reproduire les analyses

```bash
pip install -r requirements.txt
python run.py
```
---


## Pipeline Overview
 
The project is organized around two complementary pipelines built on a shared corpus:
- **Statistical analysis pipeline** (`pipeline_statistical`)
- **Hybrid experiential extraction pipeline** (`pipeline_experiential`)
---
 
## Input Data
 
The project is based on a corpus of 34,689 online music review documents, stored in JSON format (review text and metadata merged). The main metadata fields include the source, the genre, the review text, and document identifiers. Corpus loading includes format checks, basic integrity controls, and normalization of sources and genres.
 
## Controlled Vocabularies
 
Two manually built lexicons structure the analysis:
- **Perceptual adjectives** (affective, expressive, dynamic)
- **Musico-technical descriptors** (dynamics, tempo, harmony, articulation, etc.)
These vocabularies are designed as analytical tools. They make it possible to establish frequency profiles, compare distributions by source and by genre, and ground the interpretation of the statistical results. Each review document is vectorized with scikit-learn's `DictVectorizer`, producing frequency matrices by descriptor type drawn from the controlled vocabularies.
 
## Statistical Analysis Pipeline (`pipeline_statistical`)
 
This pipeline operates at the document and corpus level. It covers data loading and **filtering**, **aggregation of descriptors by source and by genre**, **distribution analysis (genre × source)**, **inferential statistical tests (chi-square, standardized residuals)**, and an **exploratory multivariate analysis via correspondence analysis**.
 
- **Results are exported as tables and figures to the `outputs/` folder.**
## Experiential Pipeline: Hybrid Information-Extraction Methods (`pipeline_experiential`)
 
This pipeline aims to identify domains of mediation of the listening experience, beyond lexical descriptors alone.
 
- **A first approach relies on lexical rules (regex).** For exploratory purposes, this approach prioritizes recall (coverage) in order to capture generic referential roles. This limitation motivates the complementary use of GLiNER2, in order to identify referential mentions beyond exact matching.
- **In addition, few-shot entity extraction is carried out with the GLiNER2 model.** Entity extraction is based on a stratified random sample of 200 reviews per source, under a length constraint (450–650 tokens). For each review, three sentences were randomly selected. Extraction was performed using GLiNER2 small v2.1, with a global threshold of 0.45, adjusted by category-specific thresholds (BODY = 0.50; MEMORY = 0.60; PLACE = 0.65; PERSON = 0.65). Rates were stabilized by bootstrap (1,000 iterations), with the entire protocol made reproducible through the use of a fixed seed (42).
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
