from typing import List, Dict, Tuple
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import csr_matrix


def build_attribute_matrices(
    counts_list: List[Dict[str, int]],
    binary_list: List[Dict[str, int]]
) -> Tuple[csr_matrix, csr_matrix, list]:
    """
    Construit les matrices fréquentielle et binaire
    dans un espace d’attributs strictement identique.
    """
    vectorizer = DictVectorizer(sparse=True)

    # Fit on union to avoid silent feature loss
    vectorizer.fit(counts_list + binary_list)

    X_counts = vectorizer.transform(counts_list)
    X_binary = vectorizer.transform(binary_list)

    # Enforce binary values defensively
    X_binary.data[:] = 1

    feature_names = vectorizer.get_feature_names_out()

    return X_counts, X_binary, feature_names
