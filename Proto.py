class Puissance4:
    def __init__(self):
        self.lignes = 6
        self.colonnes = 7
        self.grille = [[0 for _ in range(self.colonnes)] for _ in range(self.lignes)]
        self.joueur_actuel = 1  # 1 pour le joueur 1, 2 pour le joueur 2

    def afficher_grille(self):
        for ligne in self.grille:
            print('|'.join(str(cell) for cell in ligne))
            print('-' * (self.colonnes * 2 - 1))

    def jouer_coup(self, colonne):
        if colonne < 0 or colonne >= self.colonnes or self.grille[0][colonne] != 0:
            return False  # Coup invalide

        for ligne in reversed(range(self.lignes)):
            if self.grille[ligne][colonne] == 0:
                self.grille[ligne][colonne] = self.joueur_actuel
                break
        return True

    def changer_joueur(self):
        self.joueur_actuel = 2 if self.joueur_actuel == 1 else 1

    def est_gagnant(self, joueur):
        # Vérification des lignes
        for ligne in range(self.lignes):
            for col in range(self.colonnes - 3):
                if all(self.grille[ligne][col + i] == joueur for i in range(4)):
                    return True

        # Vérification des colonnes
        for col in range(self.colonnes):
            for ligne in range(self.lignes - 3):
                if all(self.grille[ligne + i][col] == joueur for i in range(4)):
                    return True

        # Vérification des diagonales montantes
        for ligne in range(3, self.lignes):
            for col in range(self.colonnes - 3):
                if all(self.grille[ligne - i][col + i] == joueur for i in range(4)):
                    return True

        # Vérification des diagonales descendantes
        for ligne in range(self.lignes - 3):
            for col in range(self.colonnes - 3):
                if all(self.grille[ligne + i][col + i] == joueur for i in range(4)):
                    return True

        return False

    def est_plein(self):
        return all(self.grille[0][col] != 0 for col in range(self.colonnes))

    def jouer(self):
        while True:
            self.afficher_grille()
            try:
                colonne = int(input(f"Joueur {self.joueur_actuel}, choisissez une colonne (0-{self.colonnes - 1}): "))
            except ValueError:
                print("Entrée invalide. Veuillez entrer un nombre.")
                continue

            if not self.jouer_coup(colonne):
                print("Colonne pleine ou invalide. Choisissez une autre colonne.")
                continue

            if self.est_gagnant(self.joueur_actuel):
                self.afficher_grille()
                print(f"Félicitations! Le joueur {self.joueur_actuel} a gagné!")
                break

            if self.est_plein():
                self.afficher_grille()
                print("Match nul! La grille est pleine.")
                break

            self.changer_joueur()

# Pour jouer :
jeu = Puissance4()
jeu.jouer()
