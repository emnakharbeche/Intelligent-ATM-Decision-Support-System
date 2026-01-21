import joblib
import numpy as np

# Charger les fichiers .pkl
ridge_model = joblib.load('ridge_model.pkl')
ridge_scaler = joblib.load('ridge_scaler.pkl')
ridge_columns = joblib.load('ridge_columns.pkl')

# Afficher les informations du modèle Ridge
print("Informations sur ridge_model :")
print(f"Type : {type(ridge_model)}")
print(f"Coefficients : {ridge_model.coef_}")
print(f"Intercept : {ridge_model.intercept_}")
print(f"Alpha : {ridge_model.alpha}")

# Afficher les informations du scaler
print("\nInformations sur ridge_scaler :")
print(f"Type : {type(ridge_scaler)}")
print(f"Mean : {ridge_scaler.mean_}")
print(f"Scale : {ridge_scaler.scale_}")
print(f"Feature names (si disponible) : {getattr(ridge_scaler, 'feature_names_in_', 'Non disponible')}")

# Afficher les colonnes
print("\nInformations sur ridge_columns :")
print(f"Type : {type(ridge_columns)}")
print(f"Nombre de colonnes : {len(ridge_columns)}")
print(f"Colonnes : {ridge_columns}")

# Vérifier une prédiction de test (exemple avec une agence fictive)
sample_data = np.zeros((1, len(ridge_columns)))
sample_data[0][ridge_columns.get_loc('agence_nom_AGENCE DOUZ')] = 1
sample_data[0][ridge_columns.get_loc('jour_semaine_1')] = 1
sample_data[0][ridge_columns.get_loc('mois_5')] = 1
sample_data[0][ridge_columns.get_loc('jour_6')] = 1
sample_data[0][ridge_columns.get_loc('montant_moyen_7j')] = 50000
sample_data_scaled = ridge_scaler.transform(sample_data)
prediction = ridge_model.predict(sample_data_scaled)
print(f"\nPrédiction de test (AGENCE DOUZ, 06/05/2025) : {prediction[0]:.2f} TND")