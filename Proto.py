import pygame
import sys

class Puissance4:
    def __init__(self):
        self.lignes = 6
        self.colonnes = 7
        self.grille = [[0 for _ in range(self.colonnes)] for _ in range(self.lignes)]
        self.joueur_actuel = 1  # 1 pour le joueur 1, 2 pour le joueur 2
        self.largeur = 700
        self.hauteur = 600
        self.rayon = int(self.largeur / self.colonnes / 2 - 5)

        # Initialiser Pygame
        pygame.init()
        self.fenetre = pygame.display.set_mode((self.largeur, self.hauteur))
        pygame.display.set_caption("Puissance 4")
        self.couleurs = {1: (255, 0, 0), 2: (255, 255, 0), 0: (0, 0, 0)}  # Rouge, Jaune, Vide

    def afficher_grille(self):
        # Dessiner la grille en bleu
        self.fenetre.fill((0, 0, 255))  # Fond bleu pour la grille
        for ligne in range(self.lignes):
            for col in range(self.colonnes):
                # Dessiner les cercles noirs pour les trous vides et les pions rouge/jaune
                pygame.draw.circle(self.fenetre, self.couleurs[self.grille[ligne][col]],
                                   (col * self.largeur // self.colonnes + self.largeur // (2 * self.colonnes),
                                    (ligne + 1) * self.hauteur // self.lignes-40),
                                   self.rayon)
        pygame.display.update()

    def jouer_coup(self, colonne):
        if colonne < 0 or colonne >= self.colonnes or self.grille[0][colonne] != 0:
            return False  # Coup invalide

        for ligne in reversed(range(self.lignes)):  # Commencer par le bas
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
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos_x = event.pos[0]
                    colonne = pos_x // (self.largeur // self.colonnes)

                    if self.jouer_coup(colonne):
                        if self.est_gagnant(self.joueur_actuel):
                            self.afficher_grille()
                            print(f"Félicitations! Le joueur {self.joueur_actuel} a gagné!")
                            pygame.time.wait(2000) 
                            pygame.quit()
                            sys.exit()

                        if self.est_plein():
                            self.afficher_grille()
                            print("Match nul! La grille est pleine.")
                            pygame.time.wait(2000)  
                            pygame.quit()
                            sys.exit()

                        self.changer_joueur()

            self.afficher_grille()


jeu = Puissance4()
jeu.jouer()
