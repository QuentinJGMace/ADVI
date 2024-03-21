# Structure du repo

- advi.py : ne marche pas, outdated
- advi2.py : version qui marche
    - Contient 2 classes : ModelParam2, ADVI2
    - ModelParam2 : sert à gérer les paramètres qu'apprend l'advi (peut les sampler, retrouner leur distribution et contient leur moyenne + std)
    - ADVI2 : algo en lui même, méthode fit qui permet de fit un modèle à des données
- ppca.py : Contient les modèles pour la ppca (**PPCA_model, PPCA_with_ARD_model**), tous les paramètres et les calculs importants pour ADVI (log_vraissemblance en particulier)
- test.ipynb, test.py : fichiers de test avec un petit jeu de donnée en 2D pour voir si ça fonctionne.
- data_exploration.ipynb : Fichier ou j'ai lu et mis en forme les données sur les taxis

# Ajout d'un nouveau modèle
Le modèle doit avoir les même méthodes que dans le fichier ppca.py. C'est à dire :
## Attributs
- self.num_parameters : la dimension totale des paramètres (i.e mean + sigma en 2 dimension --> num_parameters = 4)
- self.named_params (en vrai pas besoin) : des clés pour chacun des paramètres
- self.dim_parameters : dictionnaire {clé de paramètre : dimension du paramètre}
- self.key_pos : les clés des paramètres qui sont passés au log (typiquement std)

## Méthodes
- dist(\*\*params) : retourne la distribution du modèle selon les paramètres données
- rsample(n, \*\*params) : sample n points selon self.dist(\*\*params)
- theta_from_zeta(zeta) : retourne le vecteur de paramètre $\theta = T^{-1}(\zeta)$
- grad_inv_T: OSEF, plus utilisée, c'est calculé par pytorch
- log_det(zeta) : retourne $log(|det(J_{T^{-1}}(\zeta))|)$
- log_prob(x, theta, full_data_size) : retourne la log vraissemblance des données selon le paramètre theta. Full data size est la taille totale du jeu de données si x ne représente qu'un sous ensemble.
