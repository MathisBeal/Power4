import pygame
import sys
import math
import random
import matplotlib.pyplot as plt


class Puissance4:
    def __init__(self, grid=None, player=None):
        self.lignes = 6
        self.colonnes = 7
        self.grille = [[0 for _ in range(self.colonnes)] for _ in range(self.lignes)]
        self.joueur_humain = 1  # Devient l'IA RL
        self.joueur_ordi = 2    # Minimax
        self.joueur_actuel = self.joueur_humain  # IA RL commence
        self.largeur = 700
        self.hauteur = 600
        self.rayon = int(self.largeur / self.colonnes / 2 - 5)
        self.scores_humain = []
        self.scores_ordi = []
        self.coup_count = []

        # Initialiser Pygame
        pygame.init()
        self.fenetre = pygame.display.set_mode((self.largeur, self.hauteur))
        pygame.display.set_caption("Puissance 4")
        self.couleurs = {1: (255, 0, 0), 2: (255, 255, 0), 0: (0, 0, 0)}  # Rouge, Jaune, Vide

        if not grid is None:
            self.grille = grid

        if not player is None:
            self.joueur_actuel = player

    def tour_ia_rl(self, ia_rl_action):
        coups_valides = self.obtenir_coups_valides()

        if ia_rl_action in coups_valides:
            if self.jouer_coup(ia_rl_action, self.joueur_humain):
                print(f"L'IA RL joue dans la colonne {ia_rl_action}.")
                return True
            else:
                print(f"Coup invalide de l'IA RL : {ia_rl_action}.")
                return False
        else:
            ia_rl_action = random.choice(coups_valides)
            if self.jouer_coup(ia_rl_action, self.joueur_humain):
                print(f"L'IA RL joue dans la colonne {ia_rl_action} (choisi aléatoirement).")
                return True
            else:
                print(f"Coup invalide de l'IA RL : {ia_rl_action}.")
                return False


    def obtenir_etat_grille(self):
        return [row[:] for row in self.grille]

    def mise_a_jour_scores(self):
        score_humain = self.evaluer_position(self.joueur_humain)
        score_ordi = self.evaluer_position(self.joueur_ordi)

        self.scores_humain.append(score_humain)
        self.scores_ordi.append(score_ordi)
        self.coup_count.append(len(self.scores_humain) + len(self.scores_ordi))

    def afficher_grille(self):
        self.fenetre.fill((0, 0, 255))  # Fond bleu pour la grille
        for ligne in range(self.lignes):
            for col in range(self.colonnes):
                pygame.draw.circle(self.fenetre, self.couleurs[self.grille[ligne][col]],
                                   (col * self.largeur // self.colonnes + self.largeur // (2 * self.colonnes),
                                    (ligne + 1) * self.hauteur // self.lignes - 40),
                                   self.rayon)
        pygame.display.update()
        self.mise_a_jour_scores()

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
            score -= 80
        return score

    def evaluer_position(self, joueur):
        score = 0
        for ligne in self.grille:
            for col in range(self.colonnes - 3):
                groupe = ligne[col:col + 4]
                score += self.evaluer_groupe(groupe, joueur)
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
        colonne, _ = self.minimax(2, -math.inf, math.inf, True)
        self.jouer_coup(colonne, self.joueur_ordi)
        print(f"L'ordinateur joue dans la colonne {colonne}.")

    def afficher_graphique(self):
        pygame.quit()
        plt.plot(self.coup_count, self.scores_humain, label='IA RL', color='red')
        plt.plot(self.coup_count, self.scores_ordi, label='Minimax', color='yellow')
        plt.title("Évolution des scores")
        plt.xlabel("Nombre de coups")
        plt.ylabel("Score")
        plt.legend()
        plt.grid()
        plt.show()

    def jouer_automatique(self, ia_rl_action):
        while True:
            self.afficher_grille()

            if self.joueur_actuel == self.joueur_ordi:
                self.tour_ordinateur()
                if self.est_gagnant(self.joueur_ordi):
                    self.afficher_grille()
                    print("Minimax gagne!")
                    pygame.time.wait(2000)
                    return "Minimax"
                self.joueur_actuel = self.joueur_humain

            elif self.joueur_actuel == self.joueur_humain:
                valid = self.tour_ia_rl(ia_rl_action)
                if not valid:
                    print(f"Action invalide de l'IA RL: {ia_rl_action}.")
                    return "Invalid"
                if self.est_gagnant(self.joueur_humain):
                    self.afficher_grille()
                    print("IA RL gagne!")
                    pygame.time.wait(2000)
                    return "RL"
                self.joueur_actuel = self.joueur_ordi

            if self.est_plein():
                self.afficher_grille()
                print("Match nul!")
                pygame.time.wait(2000)
                return "Draw"
