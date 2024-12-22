from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import random

app = Flask(__name__)

# Charger et préparer les données
def load_and_prepare_data():
    global route_mapping, airline_mapping, reward_matrix, Q, num_routes, num_airlines
    # Charger les données
    data = pd.read_csv('C:/Users/hedia/Desktop/projet fac/origins_destinations_airlines_df_RL.csv')

    # Colonnes des retards
    delay_columns = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']

    # Ajouter une colonne 'TotalDelays' si elle n'existe pas
    if 'TotalDelays' not in data.columns:
        data['TotalDelays'] = data[delay_columns].sum(axis=1)

    # Nettoyer les Routes
    valid_routes = data[['Origin', 'Dest']].drop_duplicates().reset_index(drop=True)
    route_mapping = {f"{row['Origin']}-{row['Dest']}": idx for idx, row in valid_routes.iterrows()}
    airlines = data['Airline'].unique()
    airline_mapping = {airline: idx for idx, airline in enumerate(airlines)}

    # Réduction des données à 3377 lignes
    target_size = 3377
    valid_data = data.sample(n=target_size, random_state=42).reset_index(drop=True)

    # Recalcul des routes valides
    reduced_routes = valid_data[['Origin', 'Dest']].drop_duplicates().reset_index(drop=True)
    route_mapping = {f"{row['Origin']}-{row['Dest']}": idx for idx, row in reduced_routes.iterrows()}

    # Initialisation des dimensions et matrice de récompenses
    num_routes = len(route_mapping)
    num_airlines = len(airline_mapping)
    reward_matrix = np.full((num_routes, num_airlines), np.nan)

    # Remplir la matrice de récompenses
    for _, row in valid_data.iterrows():
        route_name = f"{row['Origin']}-{row['Dest']}"
        if route_name in route_mapping:
            route_index = route_mapping[route_name]
            airline_index = airline_mapping[row['Airline']]
            reward_matrix[route_index, airline_index] = -row['TotalDelays']

    # Remplir les NaN et normaliser
    reward_matrix = np.nan_to_num(reward_matrix, nan=-5000)
    max_delay = valid_data['TotalDelays'].max()
    reward_matrix = reward_matrix / max_delay

    # Initialiser la table Q
    Q = np.zeros_like(reward_matrix)

# Entraîner le modèle RL
def train_q_learning():
    global Q
    alpha = 0.5  # Taux d'apprentissage
    gamma = 0.95  # Facteur de discount
    epsilon = 0.2  # Exploration vs Exploitation
    num_episodes = 5000  # Nombre d'épisodes

    # Boucle d'apprentissage
    for episode in range(num_episodes):
        epsilon = max(0.01, epsilon * 0.99)
        route = random.choice(range(num_routes))
        for _ in range(num_airlines):
            if np.random.uniform(0, 1) < epsilon:
                airline = random.choice(range(num_airlines))
            else:
                airline = np.argmax(Q[route, :])
            reward = reward_matrix[route, airline]
            next_action = np.argmax(Q[route, :])
            Q[route, airline] += alpha * (reward + gamma * Q[route, next_action] - Q[route, airline])

# Recommandation pour une route donnée
def get_best_airline(origin, destination):
    route_name = f"{origin}-{destination}"
    if route_name not in route_mapping:
        return f"La route '{route_name}' est introuvable."
    route_index = route_mapping[route_name]
    best_airline_index = np.argmax(Q[route_index, :])
    best_airline = [airline for airline, idx in airline_mapping.items() if idx == best_airline_index][0]
    best_score = Q[route_index, best_airline_index]
    return f"Pour la route '{route_name}', la meilleure compagnie aérienne est '{best_airline}'."

# Route principale pour la page d'accueil
@app.route('/')
def index():
    return render_template('index.html')

# API pour obtenir une recommandation
@app.route('/recommend', methods=['POST'])
def recommend():
    origin = request.form.get('origin')
    destination = request.form.get('destination')
    if not origin or not destination:
        return jsonify({"error": "Veuillez fournir une origine et une destination."})
    recommendation = get_best_airline(origin, destination)
    return jsonify({"recommendation": recommendation})

if __name__ == '__main__':
    load_and_prepare_data()
    train_q_learning()
    app.run(debug=True)
