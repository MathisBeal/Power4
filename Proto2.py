import math
import random

class Puissance4:
    def __init__(self):
        self.lignes = 6
        self.colonnes = 7
        self.grille = [[0 for _ in range(self.colonnes)] for _ in range(self.lignes)]
        self.joueur_humain = 1
        self.joueur_ordi = 2
        self.joueur_actuel = 1  # Le joueur humain commence

    def afficher_grille(self):
        for ligne in self.grille:
            print('|'.join(str(cell) for cell in ligne))
            print('-' * (self.colonnes * 2 - 1))

    def jouer_coup(self, colonne, joueur):
        if colonne < 0 or colonne >= self.colonnes or self.grille[0][colonne] != 0:
            return False

        for ligne in reversed(range(self.lignes)):
            if self.grille[ligne][colonne] == 0:
                self.grille[ligne][colonne] = joueur
                break
        return True

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

    def obtenir_coups_valides(self):
        return [col for col in range(self.colonnes) if self.grille[0][col] == 0]

    def evaluer_position(self, joueur):
        # Fonction d'évaluation simple : +100 pour 4 en ligne, +10 pour 3 en ligne, +1 pour 2 en ligne
        score = 0
        # Évaluation horizontale, verticale, diagonale (vous pouvez l'améliorer)
        for ligne in self.grille:
            for col in range(self.colonnes - 3):
                groupe = ligne[col:col + 4]
                score += self.evaluer_groupe(groupe, joueur)
        # À vous de rajouter évaluation diagonale et verticale
        return score

    def evaluer_groupe(self, groupe, joueur):
        score = 0
        adversaire = self.joueur_humain if joueur == self.joueur_ordi else self.joueur_ordi
        if groupe.count(joueur) == 4:
            score += 100
        elif groupe.count(joueur) == 3 and groupe.count(0) == 1:
            score += 10
        elif groupe.count(joueur) == 2 and groupe.count(0) == 2:
            score += 1
        if groupe.count(adversaire) == 3 and groupe.count(0) == 1:
            score -= 80  # Réduire les chances de l'adversaire
        return score

    def minimax(self, profondeur, alpha, beta, maximiser):
        coups_valides = self.obtenir_coups_valides()
        est_plein = self.est_plein()
        if profondeur == 0 or self.est_gagnant(self.joueur_humain) or self.est_gagnant(self.joueur_ordi) or est_plein:
            if self.est_gagnant(self.joueur_ordi):
                return (None, 100000000000)
            elif self.est_gagnant(self.joueur_humain):
                return (None, -100000000000)
            elif est_plein:
                return (None, 0)
            else:
                return (None, self.evaluer_position(self.joueur_ordi))

        if maximiser:
            valeur_max = -math.inf
            meilleure_colonne = random.choice(coups_valides)
            for col in coups_valides:
                grille_temp = [row[:] for row in self.grille]
                self.jouer_coup(col, self.joueur_ordi)
                nouveau_score = self.minimax(profondeur - 1, alpha, beta, False)[1]
                self.grille = grille_temp
                if nouveau_score > valeur_max:
                    valeur_max = nouveau_score
                    meilleure_colonne = col
                alpha = max(alpha, valeur_max)
                if alpha >= beta:
                    break
            return meilleure_colonne, valeur_max
        else:
            valeur_min = math.inf
            meilleure_colonne = random.choice(coups_valides)
            for col in coups_valides:
                grille_temp = [row[:] for row in self.grille]
                self.jouer_coup(col, self.joueur_humain)
                nouveau_score = self.minimax(profondeur - 1, alpha, beta, True)[1]
                self.grille = grille_temp
                if nouveau_score < valeur_min:
                    valeur_min = nouveau_score
                    meilleure_colonne = col
                beta = min(beta, valeur_min)
                if alpha >= beta:
                    break
            return meilleure_colonne, valeur_min

    def tour_ordinateur(self):
        colonne, _ = self.minimax(4, -math.inf, math.inf, True)  # Profondeur 4 pour l'exploration
        self.jouer_coup(colonne, self.joueur_ordi)
        print(f"L'ordinateur joue dans la colonne {colonne}.")

    def jouer(self):
        while True:
            self.afficher_grille()
            if self.joueur_actuel == self.joueur_humain:
                try:
                    colonne = int(input(f"Joueur {self.joueur_humain}, choisissez une colonne (0-{self.colonnes - 1}): "))
                except ValueError:
                    print("Entrée invalide. Veuillez entrer un nombre.")
                    continue

                if not self.jouer_coup(colonne, self.joueur_humain):
                    print("Colonne pleine ou invalide. Choisissez une autre colonne.")
                    continue
            else:
                self.tour_ordinateur()

            if self.est_gagnant(self.joueur_actuel):
                self.afficher_grille()
                print(f"Félicitations! Le joueur {self.joueur_actuel} a gagné!")
                break

            if self.est_plein():
                self.afficher_grille()
                print("Match nul! La grille est pleine.")
                break

            self.joueur_actuel = 2 if self.joueur_actuel == 1 else 1

# Pour jouer :
jeu = Puissance4()
jeu.jouer()
