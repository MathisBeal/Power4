import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt  # Importation pour afficher un graphique
from Proto import Puissance4

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def preprocess_grid(grid):
    # Convertit la grille en un format 1D pour le réseau
    return np.array(grid).flatten()

    

def detecter_menace(jeu, joueur_adverse):
    """
    Détecte si l'adversaire peut compléter une ligne, une colonne, ou une diagonale de 4 avec un prochain coup.
    Retourne une liste de coups qui bloquent ces menaces.
    """
    menaces = []

    # Vérification des lignes
    for ligne in range(jeu.lignes):
        for col in range(jeu.colonnes - 3):
            groupe = [jeu.grille[ligne][col + i] for i in range(4)]
            if groupe.count(joueur_adverse) == 3 and groupe.count(0) == 1:
                # L'adversaire peut gagner dans cette ligne, donc on doit bloquer ici
                for i in range(4):
                    if groupe[i] == 0:
                        menaces.append(col + i)

    # Vérification des lignes de 2 adverses
    for ligne in range(jeu.lignes):
        for col in range(jeu.colonnes - 2):
            groupe = [jeu.grille[ligne][col + i] for i in range(3)]
            if groupe.count(joueur_adverse) == 2 and groupe.count(0) == 1:
                for i in range(3):
                    if groupe[i] == 0:
                        menaces.append(col + i)

    # Vérification des colonnes
    for col in range(jeu.colonnes):
        for ligne in range(jeu.lignes - 3):
            groupe = [jeu.grille[ligne + i][col] for i in range(4)]
            if groupe.count(joueur_adverse) == 3 and groupe.count(0) == 1:
                # L'adversaire peut gagner dans cette colonne, donc on doit bloquer ici
                for i in range(4):
                    if groupe[i] == 0:
                        menaces.append(col)

    # Vérification des diagonales montantes et descendantes
    for ligne in range(jeu.lignes - 3):
        for col in range(jeu.colonnes - 3):
            # Diagonale descendante
            groupe = [jeu.grille[ligne + i][col + i] for i in range(4)]
            if groupe.count(joueur_adverse) == 3 and groupe.count(0) == 1:
                # L'adversaire peut gagner dans cette diagonale descendante, donc on doit bloquer ici
                for i in range(4):
                    if groupe[i] == 0:
                        menaces.append(col + i)

            # Diagonale montante
            groupe = [jeu.grille[ligne + 3 - i][col + i] for i in range(4)]
            if groupe.count(joueur_adverse) == 3 and groupe.count(0) == 1:
                # L'adversaire peut gagner dans cette diagonale montante, donc on doit bloquer ici
                for i in range(4):
                    if groupe[i] == 0:
                        menaces.append(col + i)

    return list(set(menaces))  # Supprimer les doublons


def detecter_menaces_potentielles(jeu, joueur_adverse):
    """
    Détecte si l'adversaire pourrait créer une ligne gagnante dans deux coups.
    Retourne une liste de colonnes où l'IA devrait jouer pour bloquer ces menaces.
    """
    menaces_potentielles = []

    # Obtenir toutes les colonnes valides
    colonnes_valides = jeu.obtenir_coups_valides()

    for col in colonnes_valides:  # Parcours des colonnes valides uniquement
        # Simule un coup pour l'adversaire
        jeu.simuler_coup(col, joueur_adverse)  # L'adversaire joue dans cette colonne
        menaces = detecter_menace(jeu, joueur_adverse)  # Vérifie les menaces immédiates après ce coup

        if menaces:
            menaces_potentielles.append(col)  # Cette colonne mène à une menace potentielle

        jeu.annuler_coup(col)  # Annule le coup simulé

    return list(set(menaces_potentielles))  # Supprime les doublons




def calculate_reward(jeu, joueur_rl, joueur_minimax, action, action_history):
    """
    Calculer la récompense pour l'IA RL, prenant en compte le blocage des menaces de l'adversaire.
    """
    reward = 0

    # Détecter les menaces de l'adversaire
    menaces = detecter_menace(jeu, joueur_minimax)

    # Si l'adversaire menace de gagner, bloquer cette menace
    if action in menaces:
        reward += 3.0  # Bloquer une menace

    # Récompenser la prévention de menaces plus faibles (lignes de 2 adverses)
    for ligne in range(jeu.lignes):
        for col in range(jeu.colonnes - 2):  # Ajustement pour une ligne de 2
            groupe = [jeu.grille[ligne][col + i] for i in range(3)]
            if groupe.count(joueur_minimax) == 2 and groupe.count(0) == 1:
                reward += 1.0  # Récompenser le blocage d'une ligne de 2

    # Récompenser l'attaque (lignes de 3 pions RL avec 1 case vide)
    for ligne in range(jeu.lignes):
        for col in range(jeu.colonnes - 3):
            groupe = [jeu.grille[ligne][col + i] for i in range(4)]
            if groupe.count(joueur_rl) == 3 and groupe.count(0) == 1:
                reward += 2.0  # Construire une ligne gagnante est toujours utile

    # Récompenser la contribution à une ligne de 4 pour l'IA RL
    for ligne in range(jeu.lignes):
        for col in range(jeu.colonnes - 3):
            groupe = [jeu.grille[ligne][col + i] for i in range(4)]
            if groupe.count(joueur_rl) == 3 and groupe.count(0) == 1:
                reward += 1.0 # L'IA construit une ligne gagnante

    # Vérifier les résultats finaux
    if jeu.est_gagnant(joueur_rl):
        reward += 5.0  # Victoire de l'IA RL
    elif jeu.est_gagnant(joueur_minimax):
        reward -= 5.0  # Défaite contre Minimax
    elif jeu.est_plein():
        reward += 0.0  # Match nul

    return reward




def train_rl_vs_minimax(episodes=500, alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
    input_size = 6 * 7
    output_size = 7
    q_network = QNetwork(input_size, output_size)

        # Charger un modèle pré-entraîné si disponible
    load = input("Charger un modèle pré-entraîné ? (Y/n): ").strip().upper()
    if load == "Y":
        filename = input("Nom du fichier du modèle sauvegardé : ").strip()
        try:
            q_network.load_state_dict(torch.load(filename))
            print(f"Modèle chargé depuis le fichier {filename}.")
        except FileNotFoundError:
            print(f"Fichier {filename} introuvable. Entraînement depuis zéro.")



    optimizer = optim.Adam(q_network.parameters(), lr=alpha)
    loss_fn = nn.MSELoss()

    rewards_history = []

    for episode in range(episodes):
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))  # Decay epsilon over episodes
        print(f"--- Partie {episode + 1} ---")
        jeu = Puissance4()
        state = preprocess_grid(jeu.obtenir_etat_grille())
        state = torch.tensor(state, dtype=torch.float32)

        if episode % 2 == 0:
            joueur_rl = jeu.joueur_humain
            joueur_minimax = jeu.joueur_ordi
        else:
            joueur_rl = jeu.joueur_ordi
            joueur_minimax = jeu.joueur_humain

        done = False
        action_history = []
        total_reward = 0

        while not done:
            valid_actions = jeu.obtenir_coups_valides()


            # Détecter les menaces immédiates
            menaces = detecter_menace(jeu, joueur_minimax)

            # Si des menaces potentielles existent
            menaces_potentielles = detecter_menaces_potentielles(jeu, joueur_minimax)

            # Prioriser les menaces immédiates
            if menaces:
                action = np.random.choice([m for m in menaces if m in valid_actions])
            elif menaces_potentielles:  # Ensuite, traiter les menaces anticipées
                action = np.random.choice([m for m in menaces_potentielles if m in valid_actions])
            else:
                if np.random.rand() < epsilon:  # Exploration
                    action_weights = [1.0 / (1 + action_history.count(a)) for a in valid_actions]
                    action_weights = np.array(action_weights) / sum(action_weights)
                    action = np.random.choice(valid_actions, p=action_weights)
                else:  # Exploitation
                    q_values = q_network(state)
                    valid_q_values = torch.tensor([q_values[a].item() for a in valid_actions])
                    action = valid_actions[torch.argmax(valid_q_values).item()]


            result = jeu.jouer_automatique(action)
            action_history.append(action)

            reward = calculate_reward(jeu, joueur_rl, joueur_minimax, action, action_history)

            if result == "RL":
                reward += 2.0
                done = True
            elif result == "Minimax":
                reward -= 2.0
                done = True
            elif result == "Draw":
                reward += 0.0
                done = True
            elif result == "Invalid":
                reward -= 1.0
                done = True

            next_state = preprocess_grid(jeu.obtenir_etat_grille())
            next_state = torch.tensor(next_state, dtype=torch.float32)

            if not done:
                next_q_values = q_network(next_state)
                target = reward + gamma * torch.max(next_q_values).item()
            else:
                target = reward

            q_values = q_network(state)
            loss = loss_fn(q_values[action], torch.tensor([target], dtype=torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            total_reward += reward

        avg_reward = total_reward / len(jeu.obtenir_coups_valides())
        rewards_history.append(avg_reward)

        print(f"Partie {episode + 1} terminée.")

    plt.plot(range(episodes), rewards_history, label='Récompense moyenne par épisode')
    plt.xlabel('Épisodes')
    plt.ylabel('Récompense moyenne')
    plt.title('Progression de l\'IA RL dans le jeu Puissance 4')
    plt.legend()
    plt.show()


     # Sauvegarder le modèle si l'utilisateur le souhaite
    save = input("Sauvegarder le modèle ? (Y/n): ").strip().upper()
    if save == "Y":
        filename = input("Nom du fichier pour sauvegarder le modèle : ").strip()
        torch.save(q_network.state_dict(), filename)
        print(f"Modèle sauvegardé sous le nom {filename}.")



train_rl_vs_minimax(episodes=100)
