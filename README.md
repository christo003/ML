Le but de ce travail est de trouver un modèle à partir des données de regression_data.csv qui fit au mieux avec les données de yXtest.cvs.

Différentes tentatives ont été faites. 

Celle contribuant au modèle final sont expliqué dans les premières lignes de ce fichier. 

Le fichier preprocess1.py prends les données de regression_data.csv crée un fichier regression_data1.npz dans lequel le train et le target sont bien distinct.
Puis une figure contenant les histogrammes construit à partir des paires de chaque charactéristiques et y est généré et sauvegarder dans visu_feature.png

Le fichier estimate_reg_RERFs.py estime les paramètres du modèle RERFs avec cross validation puis génère le fichier parameters.npz contenant les paramètre median des meilleures parmètre selectionner à chaque shuffle. (attention le run de ce prgramme peut être long il faut changer les paramètre num_lasso, num_cv , num_n_estimators, num_max_features, num_max_depth, etc 

Le ficher compare.py recupere le meilleure paramètres trouver avec estimate_reg_RERFs dans parameters.npz et fait une cross validation pour comparer Ridge regression avec RERFs il génère par la suite le fichier compare.png montrant accuracy de la validatoin à travers la cross validaton.

Le fichier modèle.py prend les meilleures paramètres selectionnée auparavant (dans parameters.npz) et génère le fichier model.npz.
 
Le fichier preprocess_test.py génère le fichier test.npz pour accéder plus facilement au données test et test_target.

Le fichier test.py test le model trouver dans model.npz et le test sur les données de test.npz il génère également deux figure montrant la prediction et la vraies valeurs. 



*****************************tentative non utilisé pour le modèle final**************************************************

Le fichier estimate_reg_lasso.py estimes les paramètres d'un modèle linéaire en cross validant différents lasso path puis génère la figure reg_cv.png et le ficher regression_data2.npz contenant le train sans les features selectionnées par Lasso et le résidu comme target  .

Le fichier random_forest_sklearn.py estime les paramètres d'un arbre de regression avec CV en regardant les out of bag. Puis génère une figure représentant le out of bag et mse pred (non sauvegarder dans un fichier).

Le fichier regression_tree.py construit un arbre from scratch d'un toy dataset. Ce fichier a été constuit dans le but de faire Random Forest masi faute de temps j'ai du utilisé la version déjà faite de sklearn. Ce qui est dommage car je n'ai pas compris comment il utilise les out of bag pour selectionner les bon arbres. Il génère également un plot sympatique à observer ( non sauvegarder dans un fichier)

Il est intéressant de constater dans regression_tree.py que le point de départ pour la construction d'une branche est crutial. Si celui ci est bien choisit, l'arbre devient équillibré et obtient des meilleures résultats dans le residu mais aussi visuellement.
