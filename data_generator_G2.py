import numpy as np
import pandas as pd

def generate_data(n):
    # Car Types
    car_types = np.random.choice(['Haval', 'Foton', 'Great Wall'], n, p=[0.6, 0.3, 0.1])

    # Car Models
    haval_models = ['HAVAL Jolion', 'HAVAL H6']
    foton_models = ['Gratour V5', 'SAUVANA', 'TUNLAND']
    great_wall_models = ['WINGLE 5 DOUBLE CABINE', 'POER']

    car_models = []
    for car_type in car_types:
        if car_type == 'Haval':
            car_models.append(np.random.choice(haval_models))
        elif car_type == 'Foton':
            car_models.append(np.random.choice(foton_models))
        else:
            car_models.append(np.random.choice(great_wall_models))

    # Years
    years = np.random.randint(2010, 2022, n)

    # Entretien Frequencies
    entretien_frequencies = np.random.choice([5000, 10000, 15000, 20000, 25000], n)

    # Dates d'entretien
    start_date = pd.Timestamp('2019-01-01')
    end_date = pd.Timestamp('2022-12-31')
    dates = pd.date_range(start=start_date, end=end_date, periods=n).strftime('%Y-%m-%d')

    # Raison de la visite
    visite_reasons = np.random.choice(['Entretien régulier en usine', 'Une panne urgente'], n, p=[0.8, 0.2])

    # Sous garantie
    sous_garantie = np.random.choice(['Oui', 'Non'], n, p=[0.3, 0.7])

    # Problèmes résolus (PR), Satisfaction des frais (SF), Achèvement du service (TAS),
    # Facilité de prise de rendez-vous (FRDV), Coût des services (C), Satisfaction du client (S)
    avg_satisfaction = np.random.uniform(7, 10, n)
    PR = np.round(np.random.uniform(avg_satisfaction - 2, avg_satisfaction + 1)).astype(int)
    SF = np.round(np.random.uniform(avg_satisfaction - 2, avg_satisfaction + 1)).astype(int)
    TAS = np.round(np.random.uniform(avg_satisfaction - 2, avg_satisfaction + 1)).astype(int)
    FRDV = np.round(np.random.uniform(avg_satisfaction - 2, avg_satisfaction + 1)).astype(int)
    C = np.round(np.random.uniform(avg_satisfaction - 2, avg_satisfaction + 1)).astype(int)
    S = np.round(avg_satisfaction).astype(int)
    # Type de panne
    pannes = ['Problèmes électriques', 'Problèmes de suspension', 'Surchauffe', 'Fumée du moteur', 'Démarrage difficile']
    TP = np.random.choice(pannes, n)

    # Create the DataFrame
    df = pd.DataFrame({
        'TV': car_types,
        'MV': car_models,
        'AF': years,
        'FE': entretien_frequencies,
        'DE': dates,
        'RV': visite_reasons,
        'G': sous_garantie,
        'PR': PR,
        'SF': SF,
        'TAS': TAS,
        'FRDV': FRDV,
        'C': C,
        'S': S,
        'TP': TP
    })

    return df

# Generate 100 rows of the DataFrame
data_df = generate_data(16225)

# Save the DataFrame to a CSV file
data_df.to_csv('data.csv', index=False)