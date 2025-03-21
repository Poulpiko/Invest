import pandas as pd
from scipy.optimize import minimize
import numpy as np
import json
from alpha_vantage.timeseries import TimeSeries
import os
from dotenv import load_dotenv

load_dotenv()

# Clé API Alpha Vantage
api_key = os.getenv('API_KEY')

def get_stock_price(symbol):
    """Récupère le dernier prix d'une action à partir d'Alpha Vantage."""
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, _ = ts.get_daily(symbol=symbol, outputsize='compact')
    if not data.empty:
        return data['4. close'].iloc[0]
    else:
        print(f"Erreur : Impossible de récupérer le prix pour le symbole {symbol}.")
        return None

def update_portfolio_prices(portfolio_data):
    """Met à jour les prix des actions dans le portefeuille en utilisant Alpha Vantage."""
    for stock in portfolio_data['portfolio']:
        symbol = stock['CODE']
        price = get_stock_price(symbol)
        if price:
            stock['Prix unitaire'] = price

    return portfolio_data

def rebalance_portfolio(portfolio_data, method='SLSQP', objective='kl_divergence', bounds_ratio=0.1, transaction_cost=0.35, max_investment=None, min_transaction=0, allow_selling=True, max_iterations=10, tolerance=1e-4):
    """Rééquilibre un portefeuille d'actions avec gestion des frais de transaction et contraintes d'investissement."""

    # Vérification des données d'entrée
    try:
        current_portfolio = pd.DataFrame(portfolio_data['portfolio'])
    except (KeyError, TypeError):
        print("Erreur : Données JSON invalides.")
        return None

    # Calcul des valeurs si non fournies
    if 'VALO' not in current_portfolio.columns:
        if 'Prix unitaire' not in current_portfolio.columns or 'Nombre' not in current_portfolio.columns:
            print("Erreur : Colonnes 'Prix unitaire' et 'Nombre' requises.")
            return None
        current_portfolio['Valeur'] = current_portfolio['Nombre'] * current_portfolio['Prix unitaire']

    # Calcul des poids actuels
    current_portfolio['DISTRIB'] = current_portfolio['Valeur'] / current_portfolio['Valeur'].sum()
    portfolio = current_portfolio.copy()
    portfolio.fillna(0, inplace=True)

    # Vérification des poids cible
    if not np.isclose(portfolio['Poids cible'].sum(), 1, atol=0.01):
        raise ValueError("La somme des poids cible doit être égale à 1")

    # Fonction objectif pour l'optimisation
    def objective_function(weights):
        if objective == 'squared_error':
            return sum((weights - portfolio['Poids cible'])**2)
        elif objective == 'absolute_error':
            return sum(abs(weights - portfolio['Poids cible']))
        elif objective == 'kl_divergence':
            p = weights + 1e-10
            q = portfolio['Poids cible'].values + 1e-10
            p = p / np.sum(p)
            q = q / np.sum(q)
            return np.sum(p * np.log(p / q))
        else:
            raise ValueError("Fonction objectif non reconnue.")

    # Contraintes et bornes pour l'optimisation
    portfolio['Poids actuel'] = portfolio['DISTRIB']
    constraints = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})

    if allow_selling:
        bounds = [(max(0, weight - bounds_ratio), min(1, weight + bounds_ratio))
                  for weight in portfolio['Poids actuel']]
    else:
        # Si la vente n'est pas autorisée, les bornes inférieures sont les poids actuels
        bounds = [(weight, min(1, weight + bounds_ratio))
                  for weight in portfolio['Poids actuel']]

    # Optimisation initiale
    result = minimize(objective_function, portfolio['Poids actuel'],
                      method=method, bounds=bounds, constraints=constraints)
    portfolio['Poids optimal'] = result.x

    # Itérations pour affiner les résultats
    for _ in range(max_iterations):
        # Calcul des valeurs cibles
        total_value = portfolio['Valeur'].sum()
        if max_investment is not None:
            portfolio['Valeur cible'] = portfolio['Poids cible'] * (total_value + max_investment)
        else:
            portfolio['Valeur cible'] = portfolio['Poids cible'] * total_value

        # Calcul des transactions nécessaires
        portfolio['Valeur à acheter/vendre'] = portfolio['Valeur cible'] - portfolio['Valeur']

        # Si la vente n'est pas autorisée, mettre à zéro les valeurs négatives
        if not allow_selling:
            portfolio['Valeur à acheter/vendre'] = portfolio['Valeur à acheter/vendre'].clip(lower=0)

        portfolio['Valeur à acheter/vendre'] = np.where(
            abs(portfolio['Valeur à acheter/vendre']) < min_transaction,
            0,
            portfolio['Valeur à acheter/vendre']
        )

        # Gestion des opérations
        portfolio['Opération'] = portfolio['Valeur à acheter/vendre'].apply(
            lambda x: 'Acheter' if x > 0 else 'Vendre' if x < 0 else 'Conserver'
        )

        # Calcul des quantités d'actions
        portfolio['Nombre à acheter/vendre'] = np.where(
            portfolio['Opération'] == 'Acheter',
            np.floor(portfolio['Valeur à acheter/vendre'] / portfolio['Prix unitaire']),
            np.ceil(portfolio['Valeur à acheter/vendre'] / portfolio['Prix unitaire'])
        ).astype(int)

        # Calcul des valeurs de transaction
        portfolio['Valeur de transaction'] = (portfolio['Nombre à acheter/vendre'] *
                                              portfolio['Prix unitaire'])

        # Initialisation de la colonne 'Frais réels'
        portfolio['Frais réels'] = 0.0

        # Calcul des frais de transaction
        def calculate_transaction_costs(row):
            if row['Opération'] == 'Acheter':
                return row['Valeur de transaction'] * transaction_cost/100
            elif row['Opération'] == 'Vendre':
                return row['Valeur de transaction'] * transaction_cost/100
            return 0

        portfolio['Frais réels'] = portfolio.apply(calculate_transaction_costs, axis=1)

        # Ajustement final si max_investment est défini
        if max_investment is not None:
            total_cash_flow = (portfolio['Valeur de transaction'].sum() +
                               portfolio['Frais réels'].sum())

            if abs(total_cash_flow) > max_investment:
                # Ajustement proportionnel
                ratio = max_investment / abs(total_cash_flow)
                portfolio['Valeur de transaction'] *= ratio
                portfolio['Frais réels'] *= ratio

                # Recalculer le Nombre à acheter/vendre après l'ajustement
                portfolio['Nombre à acheter/vendre'] = np.where(
                    portfolio['Opération'] == 'Acheter',
                    np.floor(portfolio['Valeur de transaction'] / portfolio['Prix unitaire']),
                    np.ceil(portfolio['Valeur de transaction'] / portfolio['Prix unitaire'])
                ).astype(int)

                # Valeur finale
                portfolio['Valeur de transaction'] = portfolio['Nombre à acheter/vendre'] * portfolio['Prix unitaire']

        # Mise à jour des valeurs finales
        portfolio['Nouvelle Valeur'] = (portfolio['Valeur'] + portfolio['Valeur de transaction']).round(4)
        portfolio['Nouveau Poids'] = (portfolio['Nouvelle Valeur'] /
                                      portfolio['Nouvelle Valeur'].sum()).round(4)

        # Vérification de la convergence
        if np.allclose(portfolio['Nouveau Poids'], portfolio['Poids cible'], atol=tolerance):
            break

    # Calcul de l'écart moyen par position
    ecart_moyen = (portfolio['Nouveau Poids'] - portfolio['Poids cible']).abs().mean()

    # Formatage des résultats
    portfolio['Valeur de transaction'] = portfolio['Valeur de transaction'].round(4)
    portfolio['Valeur Actuelle'] = portfolio['Valeur'].round(4)
    portfolio['Poids actuel'] = portfolio['Poids actuel'].round(4)
    portfolio['Poids optimal'] = portfolio['Poids optimal'].round(4)

    # Sélection des colonnes finales
    final_display = portfolio[[
        'NOM', 'Opération', 'Prix unitaire', 'Nombre à acheter/vendre',
        'Valeur de transaction', 'Valeur Actuelle', 'Poids actuel',
        'Nouvelle Valeur', 'Nouveau Poids'
    ]]

    # Affichage des résultats
    print(final_display.to_string(index=False))
    print(f"\nRésumé de l'opération:")
    print(f"Investissement total: {portfolio['Valeur de transaction'].sum():.2f} €")
    print(f"Frais totaux: {portfolio['Frais réels'].sum():.2f} €")
    print(f"Écart moyen par position: {ecart_moyen:.2%}")

    # Demander à l'utilisateur s'il souhaite sauvegarder les modifications
    save_changes = input("Voulez-vous sauvegarder les modifications dans le fichier JSON ? (y/n): ").strip().lower()
    if save_changes == 'y':
        # Mettre à jour le fichier JSON
        for stock in portfolio_data['portfolio']:
            name = stock['NOM']
            row = portfolio[portfolio['NOM'] == name]
            if not row.empty:
                stock['Nombre'] = int(row['Nombre à acheter/vendre'].values[0] + stock['Nombre'])

        with open('portfolio.json', 'w') as f:
            json.dump(portfolio_data, f, indent=4, default=lambda x: int(x) if isinstance(x, np.integer) else x)
        print("Modifications sauvegardées avec succès.")
    else:
        print("Aucune modification n'a été sauvegardée.")

    return portfolio

# Chargement des données et appel de la fonction
try:
    with open('portfolio.json', 'r') as f:
        portfolio_data = json.load(f)
        portfolio_data = update_portfolio_prices(portfolio_data)
except FileNotFoundError:
    print("Fichier portfolio.json non trouvé. Veuillez créer le fichier avec la structure appropriée.")
    exit()
except json.JSONDecodeError:
    print("Erreur de décodage JSON dans portfolio.json. Vérifiez le format du fichier.")
    exit()

# Exemple d'utilisation
final_operations = rebalance_portfolio(
    portfolio_data,
    allow_selling=False,
    max_investment=float(input("Investissement (en euros) ?\t")),
    transaction_cost=0.35, # en pourcentage
    min_transaction=10
)
