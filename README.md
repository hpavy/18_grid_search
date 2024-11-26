Tentative de faire un grid search pour avoir de la meilleure manière possible l'écoulement autour d'un cylindre 
On appelle onyx_hug pour tourné sur onyx_hug, onyx_fikr, gpu_rte
On enregistre tout. On fait tourner sur deux heures et on prendra ceux qui convergent le mieux. (Sur 24h = 36 simulations)
On fait une technique pour essayer d'avoir de meilleurs points de collocations.
On touche au nombre de points sur le cylindre aux bords, sur la taille du réseau (entre 10 et 15 couches et entre 32 et 100 neurons) au lr init, au nb_point_pde et au batch size, au nb de point sur la frontière du cylindre.

