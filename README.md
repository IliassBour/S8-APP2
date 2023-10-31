# S8-APP2


Le fichier problematique.py exécute le code pour le prétraitement des 
données, rajouter des labels aux images et ensuite, l'exécution des 3 
classifacateurs implémentés dans l'app sur les données prétraités
soit : Bayes, K-ppv (plus proche voisin), RNN (résaux de neuronnes)

Le fichier ImageCollection.py contient tout le code pour le prétraitement 
des données. La fonction prep_images() effectue pour chaque image l'analyse
de luminosité maximum, l'erreur standart RGB, l'erreur standart XYZ et le
bruit. Les fonctions nécessaires pour obtenir chaque objet de l'analyse. 
La fonction image_display() présente les images dont les indexes sont envoyé 
en paramètre. La fonction view_histogram() contient le code pour voir les 
différents histogrammes des images dont les indexes sont envoyés en paramètre.
 
Le fichier ClassificationData.py contien le code pour ajouter des labels 
aux données reçu par la classe ClassificationData et retourne par des fonctions
get les statistiques des données et les frontières.

Le fichier classifiers.py contients les classes pour effectué l'entrainement, 
la validation et les tests pour les 3 classificateurs. La classe
BayesClassifier contient le code pour effectuer l'entrainement du classificateur
et les prédiction pour le classificateur Bayes. La classe BayesClassify_APP2
contient les hyperparamètre pour le classificateur Bayes et la visualisation
de graphique de la répartition des données. La classe PPVClassifier contient 
le code pour effectuer l'entrainement du classificateur et les prédiction pour 
le classificateur K-ppv. La classe PPVClassify_APP2 contient les hyperparamètre 
pour le classificateur K-ppv et la visualisation de graphique de la répartition 
des données. La classe NNClassifier contient le code pour effectuer l'entrainement 
du classificateur et les prédiction pour le classificateur RNN. La classe 
RNNClassify_APP2 contient les hyperparamètre pour le classificateur RNN et la 
visualisation de graphique de la répartition des données.

Le fichier analysis.py contient les fonctions pour traiter les données pour 
l'entrainement ou pour tester les données.
