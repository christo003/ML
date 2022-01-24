The preprocess1.py file takes the data from regression_data.csv creates a regression_data1.npz file in which the train and the target are distinct.

Then a figure containing the histograms built from the pairs of each feature and is generated there and saved in visu_feature.png


The estimate_reg_RERFs.py file estimates the parameters of the RERFs model with cross validation then generates the parameters.npz file containing the median parameters of the best parameters selected at each shuffle. (be careful, the run of this program can be long, you must change the parameters num_lasso, num_cv , num_n_estimators, num_max_features, num_max_depth, etc.



The compare.py file retrieves the best parameter found with estimate_reg_RERFs in parameters.npz and does a cross validation to compare Ridge regression with RERFs it then generates the compare.png file showing accuracy of the validation through the cross validation.



The model.py file takes the best parameters selected before (in parameters.npz) and generates the model.npz file.

 

The preprocess_test.py file generates the test.npz file for easier access to test and test_target data.



The file test.py tests the model found in model.npz and the test on the data in test.npz it also generates two figures showing the prediction and the true values.




######################################
try explanation
########################
The estimate_reg_lasso.py file estimates linear model’s paramters by cross validating different lasso paths then generates the reg_cv.png figure and the regression_data2.npz file containing the train without the features selected by Lasso and the residue as target .

The random_forest_sklearn.py file estimates the parameters of a regression tree with CV by looking at the out of bag. Then generates a figure representing the out of bag and mse pred (not saved in a file).

The regression_tree.py file builds a tree from scratch of a toy dataset. This file was built in order to make Random Forest but for lack of time I had to use the version already made by sklearn. Which is a shame because I didn't understand how he uses the out of bag to select the right trees. It also generates a nice plot to observe (not saved in a file)

Interestingly, in regression_tree.py, the starting point for building a branch is crucial. If this one is well chosen, the tree becomes balanced and obtains better results in the residue but also visually.



######################
FRENCH
####################






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
