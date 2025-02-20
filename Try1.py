import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt  # Importation pour afficher un graphique
from Proto import Puissance4
import random
import time

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
    Détecte si l'adversaire peut gagner immédiatement et retourne les colonnes bloquantes.
    """
    menaces = []

    for col in jeu.obtenir_coups_valides():
        jeu.simuler_coup(col, joueur_adverse)
        if jeu.est_gagnant(joueur_adverse):  # Minimax gagne immédiatement
            print(f"⚠️ Menace détectée à la colonne {col} ! ⚠️")
            menaces.append(col)
        jeu.annuler_coup(col)

    if menaces:
        print(f"🚨 Menaces détectées : {menaces}")
    return [menaces[0]] if menaces else []





def detecter_menaces_potentielles(jeu, joueur_adverse):
    """
    Vérifie si un coup permettrait à l'adversaire de gagner en 2 tours.
    """
    menaces_potentielles = []

    for col in jeu.obtenir_coups_valides():
        jeu.simuler_coup(col, joueur_adverse)  # Minimax joue un coup
        if jeu.est_gagnant(joueur_adverse):  # Vérification si Minimax gagne en 1 coup
            jeu.annuler_coup(col)
            continue  # On ne regarde pas plus loin, c'est déjà une menace immédiate

        # Minimax joue un second coup après l'IA RL
        for next_col in jeu.obtenir_coups_valides():
            jeu.simuler_coup(next_col, joueur_adverse)
            if jeu.est_gagnant(joueur_adverse):  # Minimax gagne après 2 tours
                menaces_potentielles.append(col)
            jeu.annuler_coup(next_col)

        jeu.annuler_coup(col)

    return menaces_potentielles



def obtenir_coups_valides(self):
    """ Retourne une liste des colonnes où un coup est possible """
    return [col for col in range(self.colonnes) if self.grille[0][col] == 0]


def calculate_reward(jeu, joueur_rl, joueur_minimax, action, action_history):
    reward = 0

    # Détecter les menaces de l'adversaire
    menaces = detecter_menace(jeu, joueur_minimax)
    menaces_potentielles = detecter_menaces_potentielles(jeu, joueur_minimax)

    if action in menaces:
        reward += 6.0
        print(f"✅ Récompense élevée pour avoir bloqué une menace en colonne {action}.")

    if action in menaces_potentielles:
        reward += 3.5
        print(f"🔄 Récompense modérée pour anticipation en colonne {action}.")

    # 🏆 Vérifier si l'IA RL gagne immédiatement
    for col in jeu.obtenir_coups_valides():
        jeu.simuler_coup(col, joueur_rl)
        if jeu.est_gagnant(joueur_rl):
            reward += 10.0
            print(f"🏆 Récompense maximale pour une victoire immédiate en colonne {col}.")
            jeu.annuler_coup(col)
        jeu.annuler_coup(col)

    if jeu.est_gagnant(joueur_rl):
        reward += 5.0
    elif jeu.est_gagnant(joueur_minimax):
        reward -= 5.0
    elif jeu.est_plein():
        reward += 0.0

    print(f"🔎 DEBUG: Action {action}, Récompense attribuée: {reward}")
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
    moyenne_rewards = []  # Stocker la moyenne des récompenses tous les 10 épisodes
    successful_actions = set()  # Actions qui ont mené à des victoires
    failed_actions = set()  # Actions qui ont mené à des défaites

    for episode in range(episodes):
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode) * random.uniform(0.8, 1.2))
        print(f"--- Partie {episode + 1} ---")
        jeu = Puissance4()
        state = preprocess_grid(jeu.obtenir_etat_grille())
        state = torch.tensor(state, dtype=torch.float32)

        # 🔀 Alterner aléatoirement qui commence la partie
        if random.random() < 0.5:
            joueur_rl = jeu.joueur_humain
            joueur_minimax = jeu.joueur_ordi
            premier_joueur = "IA RL"
        else:
            joueur_rl = jeu.joueur_ordi
            joueur_minimax = jeu.joueur_humain
            premier_joueur = "Minimax"

        print(f"🌀 Cette partie, {premier_joueur} commence en premier.")

        # 🌀 Ajouter un coup aléatoire au début pour varier les parties
        if random.random() < 0.5:
            random_col = random.choice(jeu.obtenir_coups_valides())
            jeu.jouer_coup(random_col, joueur_minimax)
            print(f"🎲 Minimax commence en jouant aléatoirement dans la colonne {random_col}")

        if random.random() < 0.5:
            random_col = random.choice(jeu.obtenir_coups_valides())
            jeu.jouer_coup(random_col, joueur_rl)
            print(f"🎲 IA RL commence en jouant aléatoirement dans la colonne {random_col}")

        done = False
        action_history = []
        total_reward = 0

        while not done:
            valid_actions = jeu.obtenir_coups_valides()
            action = None

            # 🔥 Vérification des menaces immédiates
            menaces = detecter_menace(jeu, joueur_minimax)
            if menaces:
                action = menaces[0]  
                print(f"🛑 IA RL bloque la menace en colonne {action} !")
            else:
                menaces_potentielles = detecter_menaces_potentielles(jeu, joueur_minimax)

            # 🏆 Vérifier si l'IA RL peut gagner immédiatement
            if action is None:
                for col in jeu.obtenir_coups_valides():
                    jeu.simuler_coup(col, joueur_rl)
                    if jeu.est_gagnant(joueur_rl):
                        print(f"🏆 IA RL joue dans {col} pour gagner immédiatement !")
                        action = col
                        jeu.annuler_coup(col)
                        break
                    jeu.annuler_coup(col)

            # 🎯 Sélection du meilleur coup basé sur Q-learning
            if action is None:  
                q_values = q_network(state)
                best_action = max(valid_actions, key=lambda a: q_values[a].item())
                action = best_action
                print(f"🧠 L'IA RL choisit la colonne {action} basée sur Q-learning.")

            # 📌 Vérifier si l'IA doit s'adapter
            if action is None:
                if successful_actions:
                    action = random.choice(list(successful_actions.intersection(valid_actions)))
                elif failed_actions:
                    action = random.choice([a for a in valid_actions if a not in failed_actions])
                else:
                    action = random.choice(valid_actions)

            # 📌 Exécuter l'action
            result = jeu.jouer_automatique(action)
            time.sleep(0.5)
            action_history.append(action)

            # 🔥 Calcul de la récompense
            reward = calculate_reward(jeu, joueur_rl, joueur_minimax, action, action_history)

            # 📌 Mise à jour des états et apprentissage du réseau
            if result == "RL":
                reward += 5.0
                done = True
            elif result == "Minimax":
                reward -= 5.0
                done = True
            elif result == "Draw":
                reward += 2.0
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

            if result == "RL":
                successful_actions.update(action_history)
            elif result == "Minimax":
                failed_actions.update(action_history)

        avg_reward = total_reward / max(1, len(jeu.obtenir_coups_valides()))
        rewards_history.append(avg_reward)

        # 📌 Calculer la moyenne tous les 10 épisodes
        if (episode + 1) % 10 == 0:
            avg_last_10 = sum(rewards_history[-10:]) / 10
            moyenne_rewards.append(avg_last_10)

        print(f"✅ Partie {episode + 1} terminée. Récompense moyenne : {avg_reward:.2f}")

    # 📊 Affichage des progrès
    plt.plot(range(episodes), rewards_history, label='Récompense moyenne par épisode')
    plt.plot(range(10, episodes + 1, 10), moyenne_rewards, label='Moyenne des 10 derniers épisodes', linestyle='--')
    plt.xlabel('Épisodes')
    plt.ylabel('Récompense moyenne')
    plt.title('Progression de l\'IA RL dans le jeu Puissance 4')
    plt.legend()
    plt.show()

    # 🔹 Sauvegarde du modèle
    save = input("Sauvegarder le modèle ? (Y/n): ").strip().upper()
    if save == "Y":
        filename = input("Nom du fichier pour sauvegarder le modèle : ").strip()
        torch.save(q_network.state_dict(), filename)
        print(f"💾 Modèle sauvegardé sous le nom {filename}.")

train_rl_vs_minimax(episodes=30)
