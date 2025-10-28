# Projet IA5 — Étude : classification des textes par époque

*Mathéo GUILBERT*

---

## 1. Contexte et problématique

L'objectif principal de cette étude est d'évaluer si l'on peut reconnaître, de façon fiable et robuste, l'époque d'un texte uniquement à partir de son vocabulaire et de sa sémantique. 

Je cherche à répondre à la question : "Est-il possible de reconnaître l'époque d'un texte grâce à ses mots et à sa construction ?".

Dans un second temps, j'explore la faisabilité de "traduire" un texte d'une époque à une autre.

## 2. Démarche

Pour répondre à cette problématique, ma démarche a été la suivante :

1.  **Exploration et Prétraitement :** Analyse du jeu de données, nettoyage des textes et création d'étiquettes de "périodes" (par tranches de 10 et 50 ans).

2.  **Premier essaie :** Une première expérimentation utilisant TF-IDF avec SGDClassifier.

3.  **Second essaie :** Une seconde approche utilisant des *embeddings* de phrases pour capturer la sémantique du texte.

4.  **Analyse et Comparaison :** Comparaison des performances des deux modèles.

## 3. Les données et Prétraitement

### 3.1. Source

Le jeu de données utilisé est `PleIAs/French-PD-Books` ([voir sur Hugging Face](https://huggingface.co/datasets/PleIAs/French-PD-Books)). Il contient 289 000 livres environ, avec, pour chaque ouvrage, les données suivantes :

* file_id : id du fichier
* ocr :
* title : titre du livre
* date : date de publication, peut être une année simple ou un interval
* author : nom de l'autheur et ses dates de naissance et décès
* page_count : nombre de page du livre
* word_count : nombre de mots du livre
* character_count : nombre de personnages mentionnés dans le libre (fictifs ou réels)
* complete_text : texte entier du livre

**A noter :** les textes proviennent de scans OCR. Ils contiennent donc beaucoup de bruit : éléments de pagination (numéros de page, en‑têtes, pieds de page), sauts de ligne et retours à la ligne, coupures de mots au passage de ligne, caractères d'échappement, et parfois des erreurs d'OCR. Une étape de prétraitement robuste est donc indispensable avant toute modélisation..

#### Les dates

Les dates ne sont pas toutes homogènes, on retrouve les formats suivants :

* une année : 1860
* un interval d'années : 1929-1931
* interval d'années avec valeur manquante : 1876-???? ou ????-1876

#### La taille du jeu de données

Le jeu de données est très volumineux. On voit sur le graphique ci-dessous une forte concentration de textes sur certaines périodes.

![Répartition des textes dans le temps](images/repartition-exemples-dans-le-temps.png)

Compte tenu du temps de prétraitement et d'entraînement, j'ai effectué mes expérimentations sur un **sous-ensemble de 5000 textes**.

### 3.2. Prétraitement des textes

Deux fonctions de nettoyage ont été testées. La première (`clean_text_old`) était une tentative de nettoyage en profondeur à base de regex.

```py
# Ancienne version
def clean_text(example):
    text = example["complete_text"]
    date = example.get("date", None)

    # Si la date contient un "-", on essaie d'extraire l'année connue (format "1234-????" ou "????-1234") ou moyenne des deux années
    if "-" in str(date) and date is not None:
        parts = str(date).split("-")
        if (parts[1].isdigit() and len(parts[1]) == 4) and parts[0] == "????":
            date = str(parts[1])
        else:
            date = str(parts[0])

    # Retirer les numéros de page
    text = re.sub(r"[—\-–]\s*\d+\s*[—\-–]", " ", text)
    
    # Corriger les apostrophes et guillemets échappés
    text = text.replace("\\'", "'")
    text = text.replace("\\\"", "\"")
    text = text.replace("\\n", " ")
    text = text.replace("\\r", " ")
    text = text.replace("\\t", " ")
    
    # Corriger les mots coupés (pattern plus précis)
    text = re.sub(r'([a-zàâäæçéèêëïîôùûüœ])\s+([a-zàâäæçéèêëïîôùûüœ]{2,})', 
                  r'\1\2', text)
    
    # Corriger les cas avec plusieurs espaces
    text = re.sub(r'([a-zàâäæçéèêëïîôùûüœ])\s{2,}([a-zàâäæçéèêëïîôùûüœ])', 
                  r'\1\2', text)
    
    # Normaliser les espaces multiples
    text = re.sub(r"\s+", " ", text)
    
    # Nettoyer les caractères spéciaux
    text = re.sub(r"[^\w\s\.,;:\?!'\-\"«»À-ÖØ-öø-ÿœŒ]", " ", text)
    
    # Re-normaliser après nettoyage
    text = re.sub(r"\s+", " ", text)
    
    # Corriger la ponctuation
    text = re.sub(r"\s+([,.\?!;:])", r"\1", text)
    text = re.sub(r"([,.\?!;:])\s*([,.\?!;:])", r"\1\2", text)
    
    text = text.strip()
    return {"text": text, "date": str(date)}
```

Ici, les traitements appliqués sont :

* suppression des sauts de ligne et des retours à la ligne, remplacement par des espaces,
* élimination des caractères non alphabétiques (en conservant les lettres accentuées françaises),
* suppression partielle des numéros de page

La suppression des numéros de pages est une étape délicate car leur mise en forme dépends de l'ouvrage et de l'éditeur le plus souvent. J'ai quand même repéré un écriture récurente : -- [NUMERO DE PAGE] --. J'ai supprimé ces cas là.

Cette fonction s'est avérée complexe et pas nécessairement plus performante.

Pour l'approche par embeddings, j'ai opté pour une version simplifiée, se concentrant sur la miniscule, la suppression de la ponctuation et des stopwords.

```py
def clean_text(example):
    """
        Nettoie le texte d'entrée
    """
    text = example["complete_text"]
    date = example.get("date", None)

    # Nettoyage de la date
    if "-" in str(date) and date is not None:
        parts = str(date).split("-")
        if (parts[1].isdigit() and len(parts[1]) == 4) and parts[0] == "????"
            date = str(parts[1])
        else:
            date = str(parts[0])

    # Nettoyage de texte
    text = (text.replace("\\\\n", " ")
                .replace("\\\\r", " ")
                .replace("\\\\t", " "))
    
    text = text.lower()

    text = re.sub(r"[^a-zàâäæçéèêëïîôùûüœ\\s]", " ", text)
    
    text = re.sub(r"\\s+", " ", text).strip()

    words = text.split()
    filtered_words = [word for word in words if word not in french_stopwords]
    text = " ".join(filtered_words)

    return {"text": text, "date": str(date)}
```

Les stopwords utilisés ont été générés par Gemini après lui avoir donné les nuages de mots qui viendrons [plutard dans le rapport](#62-analyse-qualitative-nuages-de-mots) :

```py
french_stopwords = set([
    'a', 'ai', 'aie', 'aient', 'aies', 'ait', 'alors', 'as', 'au', 'aucun', 'aura', 'aurai', 'auraient', 'aurais', 'aurait', 'auras', 'aurez', 'auriez', 'aurions', 'aurons', 'auront', 'aussi', 'autre', 'aux', 'avaient', 'avais', 'avait', 'avant', 'avec', 'avez', 'aviez', 'avions', 'avoir', 'avons', 'ayant', 'ayez', 'ayons',
    'bon',
    'c', 'ce', 'ceci', 'cela', 'ces', 'cet', 'cette', 'chaque', 'comme', 'comment',
    'd', 'dans', 'de', 'des', 'deux', 'donc', 'dont', 'du',
    'elle', 'en', 'encore', 'es', 'est', 'et', 'etaient', 'etais', 'etait', 'etant', 'ete', 'etes', 'etiez', 'etions', 'etre', 'eu', 'eue', 'eues', 'eurent', 'eus', 'eusse', 'eussent', 'eusses', 'eussiez', 'eussions', 'eut', 'eux', 'eûmes', 'eût', 'eûtes',
    'fait', 'fais', 'faisaient', 'faisais', 'faisait', 'faisant', 'faire', 'faites', 'fasse', 'fassent', 'fasses', 'fassiez', 'fassions', 'faut', 'fi', 'font', 'force', 'furent', 'fus', 'fusse', 'fussent', 'fusses', 'fussiez', 'fussions', 'fut', 'fûmes', 'fût', 'fûtes',
    'hors',
    'i', 'ici', 'il', 'ils',
    'j', 'je',
    'l', 'la', 'le', 'les', 'leur', 'leurs', 'lui',
    'm', 'ma', 'mais', 'me', 'mes', 'moi', 'mon',
    'n', 'ne', 'ni', 'nos', 'notre', 'nous',
    'on', 'ont', 'ou', 'où',
    'par', 'pas', 'pendant', 'peu', 'peut', 'peux', 'plus', 'point', 'pour', 'pourquoi',
    'qu', 'quand', 'que', 'quel', 'quelle', 'quelles', 'quels', 'qui',
    's', 'sa', 'sans', 'se', 'sera', 'serai', 'seraient', 'serais', 'serait', 'seras', 'serez', 'seriez', 'serions', 'serons', 'seront', 'ses', 'soi', 'soient', 'sois', 'soit', 'sommes', 'son', 'sont', 'soyez', 'soyons', 'suis', 'sur',
    't', 'ta', 'te', 'tes', 'toi', 'ton', 'tous', 'tout', 'tu', 'un', 'une',
    'va', 'vers', 'voici', 'voilà', 'vos', 'votre', 'vous',
    'y', 'à'
])
```

### 3.3. Création des labels

Pour la classification, j'ai groupé les textes par périodes de 10 ans et 50 ans.

```py
def create_period_label(example, period_length=50):
     """
          Crée une étiquette de période basée sur l'année de publication.
     """
     try:
          year = int(example['date'])
          start_year = (year // period_length) * period_length
          end_year = start_year + period_length - 1

          return {"period": f"{start_year}-{end_year}"}
     except (ValueError, TypeError):
          return {"period": None}
     
dataset_with_labels = cleaned_ds.map(create_period_label)
```

## 4\. Méthodes de classification (Première approche)

Ma première approche a servi de base. J'ai utilisé un `TfidfVectorizer` pour transformer le texte en vecteurs de fréquence de mots, puis entraîné un `SGDClassifier`.

```py
def train_and_evaluate_tfidf(dataset, period_length_value):
    
    # Séparer en ensembles d'entraînement (80%) et de test (20%)
    train_test_split = dataset_with_labels.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split['train']
    test_dataset = train_test_split['test']

    # Extraire les textes et les labels pour une utilisation plus simple
    train_texts = [ex['text'] for ex in train_dataset]
    test_texts = [ex['text'] for ex in test_dataset]
    train_labels = [ex['decade'] for ex in train_dataset]
    test_labels = [ex['decade'] for ex in test_dataset]
    
    # Vectorisation
    # Dans cette premiere etape le pretraitement n'exclus pas les stopwords
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words=None, min_df=3, max_df=0.85)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # Entraînement
    sgd_classifier = SGDClassifier(loss='log_loss', random_state=42, max_iter=1000, tol=1e-3)
    sgd_classifier.fit(X_train_tfidf, y_train)
    
    return sgd_classifier, tfidf_vectorizer, X_test, y_test
```

## 5\. Classification sémantique par Embeddings (Seconde approche)

L'objectif est de transformer chaque texte en un vecteur qui représente son sens. Des textes sémantiquement similaires auront des vecteurs proches.

Le modèle choisi est [dangvantuan/sentence-camembert-base](https://huggingface.co/dangvantuan/sentence-camembert-base), il spécialisé pour le français.

### 5.2. Implémentation

Le code ci-dessous, illustre le chargement du modèle d'embedding et l'entraînement du classifieur :

```py
# Charger le modèle
model = SentenceTransformer("dangvantuan/sentence-camembert-base")

# Génération des embeddings
# train_texts et test_texts sont créés lors du prétraitement
X_train_embeddings = model.encode(train_texts, show_progress_bar=True)
X_test_embeddings = model.encode(test_texts, show_progress_bar=True)

# Entrainement
sgd_classifier_emb = SGDClassifier(
    loss='log_loss', 
    random_state=42, 
    max_iter=1000, 
    tol=1e-3
)
sgd_classifier_emb.fit(X_train_embeddings, train_labels)
```

