from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine, text
import bcrypt
import jwt
import datetime
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow 
from tensorflow.keras.models import load_model
app = Flask(__name__)
CORS(app)

# Configure PostgreSQL connection for pfe_atb (used for users table)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:rania123@localhost/pfe_atb'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'raniarania'  # Replace with a secure key
db = SQLAlchemy(app)

# Define the User model for pfe_atb database
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    def __repr__(self):
        return f'<User {self.username}>'

# Create the database tables for pfe_atb
with app.app_context():
    db.create_all()

# Configure a separate engine for atm_pfe database
atm_engine = create_engine('postgresql://postgres:rania123@localhost/atm_pfe')

# Load .pkl files for Ridge regression
ridge_model = joblib.load('ridge_model.pkl')
ridge_scaler = joblib.load('ridge_scaler.pkl')
ridge_columns = joblib.load('ridge_columns.pkl')

# Load .pkl file for SARIMA
sarima_model = joblib.load('sarima_model.pkl')

# Load .pkl files for Random Forest
rf_model = joblib.load('rf_model.pkl')
rf_scaler = joblib.load('rf_scaler.pkl')
rf_columns = joblib.load('rf_columns.pkl')
rf_historique = joblib.load('rf_historique.pkl')

# Signup endpoint
@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    if not username or not email or not password:
        return jsonify({'message': 'Missing fields'}), 400
    if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
        return jsonify({'message': 'User already exists'}), 400
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    new_user = User(username=username, email=email, password_hash=password_hash)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'message': 'User created successfully'}), 201

# Login endpoint
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'message': 'Missing fields'}), 400
    user = User.query.filter_by(username=username).first()
    if not user or not bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
        return jsonify({'message': 'Invalid credentials'}), 401
    token = jwt.encode({
        'user_id': user.id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    }, app.config['SECRET_KEY'], algorithm='HS256')
    return jsonify({'token': token}), 200

# Historical data endpoint
@app.route('/historical-data', methods=['GET'])
def historical_data():
    try:
        with atm_engine.connect() as conn:
            query = text("SELECT date_chargement, montant_chargés, agence_nom FROM fact_chargs_dechargs ORDER BY date_chargement")
            result = conn.execute(query)
            historical_data = [
                {
                    'date': row[0].strftime('%Y-%m-%d'),
                    'montant': float(row[1]),
                    'agence': row[2]
                }
                for row in result
            ]
        return jsonify(historical_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Agencies endpoint
@app.route('/agencies', methods=['GET'])
def get_agencies():
    try:
        with atm_engine.connect() as conn:
            query = text("SELECT DISTINCT agence_nom FROM fact_chargs_dechargs")
            result = conn.execute(query)
            agencies_list = [row[0] for row in result]
        return jsonify(agencies_list)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        model_name = data['model']
        date = data['date']
        agency = data.get('agency')
        print(f"Requête reçue: model={model_name}, date={date}, agency={agency}")

        if model_name == 'Régression Linéaire':
            if not agency:
                return jsonify({'error': 'Agency is required for Régression Linéaire'}), 400

            # Convert date
            date_obj = pd.to_datetime(date)
            jour_semaine = date_obj.dayofweek
            mois = date_obj.month
            jour = date_obj.day

            # Calculate montant_moyen_7j
            with atm_engine.connect() as conn:
                query = text("""
                    SELECT montant_chargés 
                    FROM fact_chargs_dechargs 
                    WHERE agence_nom = :agency AND date_chargement <= :date
                    ORDER BY date_chargement DESC 
                    LIMIT 7
                """)
                result = conn.execute(query, {'agency': agency, 'date': date}).fetchall()
                montant_moyen_7j = np.mean([row[0] for row in result]) if result else 0
                print(f"Régression Linéaire - Agence: {agency}, Date: {date}")
                print(f"Montant moyen 7j: {montant_moyen_7j}")
                print(f"Features: jour_semaine={jour_semaine}, mois={mois}, jour={jour}")

            # Prepare input data
            input_data = pd.DataFrame(columns=ridge_columns).fillna(0)
            input_data.loc[0] = 0 
            input_data['montant_moyen_7j'] = montant_moyen_7j
            col_agence = f'agence_nom_{agency.strip().upper()}'
            col_jour_semaine = f'jour_semaine_{jour_semaine}'
            col_mois = f'mois_{mois}'
            col_jour = f'jour_{jour}'
            one_hot_cols = []
            if col_agence in input_data.columns:
                input_data[col_agence] = 1
                one_hot_cols.append(col_agence)
            else:
                print(f"Erreur: {col_agence} non trouvé dans ridge_columns")
            if col_jour_semaine in input_data.columns:
                input_data[col_jour_semaine] = 1
                one_hot_cols.append(col_jour_semaine)
            if col_mois in input_data.columns:
                input_data[col_mois] = 1
                one_hot_cols.append(col_mois)
            if col_jour in input_data.columns:
                input_data[col_jour] = 1
                one_hot_cols.append(col_jour)
            print(f"One-hot columns activées: {one_hot_cols}")

            # Scale and predict
            input_data_scaled = ridge_scaler.transform(input_data)
            print(f"Input data scaled (first 5): {input_data_scaled[0][:5]}")
            prediction = ridge_model.predict(input_data_scaled)[0]
            print(f"Prédiction Flask (Régression Linéaire): {prediction:.2f} TND")
            response = jsonify({'prediction': float(prediction)})
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, proxy-revalidate'
            return response

        elif model_name == 'SARIMA':
            # Convert date
            date_obj = pd.to_datetime(date)

            # Load historical data
            with atm_engine.connect() as conn:
                query = text("""
                    SELECT date_chargement, SUM(montant_chargés) as montant_total
                    FROM fact_chargs_dechargs
                    GROUP BY date_chargement
                    ORDER BY date_chargement
                """)
                df = pd.read_sql_query(query, conn)
            df['date_chargement'] = pd.to_datetime(df['date_chargement'])
            df.set_index('date_chargement', inplace=True)
            df = df.asfreq('D').fillna(0)  # Ensure daily frequency, fill missing with 0

            # Calculate steps to forecast
            last_date = df.index[-1]
            steps = (date_obj - last_date).days
            if steps <= 0:
                return jsonify({'error': 'Date must be in the future'}), 400

            # Generate forecast
            forecast_result = sarima_model.get_forecast(steps=steps)
            forecast = forecast_result.predicted_mean
            prediction = forecast[date_obj]
            print(f"Prédiction Flask (SARIMA): {prediction:.2f} TND")
            response = jsonify({'prediction': float(prediction)})
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, proxy-revalidate'
            return response

        elif model_name == 'Random Forest':
            if not agency:
                return jsonify({'error': 'Agency is required for Random Forest'}), 400

            # Convert date
            date_obj = pd.to_datetime(date)
            jour_semaine = date_obj.dayofweek
            mois = date_obj.month

            # Create input data
            input_data = pd.DataFrame([[0] * len(rf_columns)], columns=rf_columns)
            # Temporal features
            input_data['jour_annee'] = date_obj.dayofyear
            input_data['est_weekend'] = 1 if date_obj.dayofweek in [5, 6] else 0
            input_data['est_fin_mois'] = 1 if date_obj.day >= 25 else 0
            # One-hot encoded features
            safe_agence_name = agency.replace(" ", "_").upper()
            col_agence = f'agence_nom_{safe_agence_name}'
            col_jour_semaine = f'jour_semaine_{jour_semaine}'
            col_mois = f'mois_{mois}'
            one_hot_cols = []
            if col_agence in input_data.columns:
                input_data[col_agence] = 1
                one_hot_cols.append(col_agence)
            else:
                print(f"Erreur: {col_agence} non trouvé dans rf_columns")
            if col_jour_semaine in input_data.columns:
                input_data[col_jour_semaine] = 1
                one_hot_cols.append(col_jour_semaine)
            if col_mois in input_data.columns:
                input_data[col_mois] = 1
                one_hot_cols.append(col_mois)
            # Historical features from rf_historique
            agency_data = rf_historique[rf_historique['agence_nom'] == agency].sort_values('date_chargement')
            if len(agency_data) > 0:
                montant_mean_7j = agency_data['montant_chargés'].tail(7).mean()
                agence_mean = agency_data['agence_mean'].iloc[-1]
                input_data['montant_mean_7j'] = montant_mean_7j if not np.isnan(montant_mean_7j) else rf_historique['montant_chargés'].median()
                input_data['agence_mean'] = agence_mean if not np.isnan(agence_mean) else rf_historique['montant_chargés'].median()
            else:
                input_data['montant_mean_7j'] = rf_historique['montant_chargés'].median()
                input_data['agence_mean'] = rf_historique['montant_chargés'].median()
            # Log features
            print(f"Random Forest - Agence: {agency}, Safe agence: {safe_agence_name}, Date: {date}")
            print(f"Montant mean 7j: {input_data['montant_mean_7j'].iloc[0]}")
            print(f"Agence mean: {input_data['agence_mean'].iloc[0]}")
            print(f"Features: jour_annee={input_data['jour_annee'].iloc[0]}, est_weekend={input_data['est_weekend'].iloc[0]}, est_fin_mois={input_data['est_fin_mois'].iloc[0]}")
            print(f"One-hot columns activées: {one_hot_cols}")
            # Scale numerical features
            numeric_cols = ['montant_mean_7j', 'jour_annee', 'agence_mean']
            input_data[numeric_cols] = rf_scaler.transform(input_data[numeric_cols])
            print(f"Input data scaled: {input_data[numeric_cols].iloc[0].values}")
            # Predict (log scale) and inverse transform
            prediction_log = rf_model.predict(input_data)[0]
            prediction = np.expm1(prediction_log)
            print(f"Prédiction brute (log scale): {prediction_log:.4f}")
            print(f"Prédiction Flask (Random Forest): {prediction:.2f} TND")
            response = jsonify({'prediction': float(prediction)})
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, proxy-revalidate'
            return response

        elif model_name == 'LSTM':
            if not agency:
                return jsonify({'error': 'Agency is required for LSTM'}), 400

            # Définir les fonctions nécessaires pour LSTM
            def create_features_temporelles(df):
                df['date_chargement'] = pd.to_datetime(df['date_chargement'])
                df['semaine_annee'] = df['date_chargement'].dt.isocalendar().week
                df['trimestre'] = df['date_chargement'].dt.quarter
                df['est_weekend'] = df['date_chargement'].dt.dayofweek.isin([5, 6]).astype(int)
                df['est_debut_mois'] = (df['date_chargement'].dt.day <= 7).astype(int)
                df['est_fin_mois'] = (df['date_chargement'].dt.day > 24).astype(int)
                return df

            def prepare_sequences(data, agence, sequence_length=14, feature_cols=None):
                agence_data = data[data['agence_nom'] == agence].sort_values('date_chargement')
                if agence_data.empty:
                    raise ValueError(f"Aucune donnée pour l'agence '{agence}'")
                if feature_cols is None:
                    feature_cols = ['montant_chargés', 'semaine_annee', 'trimestre', 'est_weekend', 'est_debut_mois', 'est_fin_mois']
                agence_data = agence_data.dropna(subset=feature_cols)
                features = agence_data[feature_cols].values
                if len(features) <= sequence_length:
                    raise ValueError(f"Pas assez de données ({len(features)}) pour {sequence_length} séquences")
                X, y = [], []
                for i in range(len(features) - sequence_length):
                    X.append(features[i:i + sequence_length])
                    y.append(features[i + sequence_length][0])
                return np.array(X), np.array(y)

            def normalize_data(X, y, scaler=None, y_scaler=None, fit_scaler=False):
                from sklearn.preprocessing import MinMaxScaler
                if scaler is None:
                    scaler = MinMaxScaler()
                if y_scaler is None:
                    y_scaler = MinMaxScaler()
                original_shape = X.shape
                X_reshaped = X.reshape(-1, original_shape[2])
                if fit_scaler:
                    X_scaled = scaler.fit_transform(X_reshaped)
                else:
                    X_scaled = scaler.transform(X_reshaped)
                X_scaled = X_scaled.reshape(original_shape)
                y_reshaped = y.reshape(-1, 1)
                if fit_scaler:
                    y_scaled = y_scaler.fit_transform(y_reshaped)
                else:
                    y_scaled = y_scaler.transform(y_reshaped)
                return X_scaled, y_scaled.flatten(), scaler, y_scaler

            def reshape_for_lstm(X, y=None):
                if X.shape[2] != 6:
                    raise ValueError(f"Nombre de features inattendu. Reçu: {X.shape[2]}, Attendu: 6")
                if X.ndim == 2:
                    X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
                elif X.ndim == 3:
                    X_reshaped = X
                else:
                    raise ValueError(f"Shape de X non supporté: {X.shape}")
                return (X_reshaped, y) if y is not None else X_reshaped

            def create_lstm_model(input_shape, lstm_units=64, dropout_rate=0.2):
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
                from tensorflow.keras.optimizers import Adam
                model = Sequential()
                model.add(Input(shape=input_shape))
                model.add(LSTM(lstm_units, activation='tanh', return_sequences=True))
                model.add(Dropout(dropout_rate))
                model.add(LSTM(lstm_units // 2, activation='tanh'))
                model.add(Dropout(dropout_rate))
                model.add(Dense(32, activation='relu'))
                model.add(Dense(1))
                model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
                return model

            def train_model(model, X, y, epochs=50, batch_size=32):
                from tensorflow.keras.callbacks import EarlyStopping
                callbacks = [
                    EarlyStopping(monitor='loss', patience=15, restore_best_weights=True, verbose=1)
                ]
                history = model.fit(
                    X, y,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=False,
                    callbacks=callbacks,
                    verbose=1
                )
                return history

            # Charger les données
            with atm_engine.connect() as conn:
                df = pd.read_sql("SELECT * FROM fact_chargs_dechargs", conn)
            df = create_features_temporelles(df)

            # Vérifier si l'agence est valide (assez de données)
            sequence_length = 14
            df_agence = df[df['agence_nom'] == agency]
            if len(df_agence) < sequence_length + 1:
                return jsonify({'error': f'Pas assez de données pour {agency}'}), 400

            # Features
            feature_cols = ['montant_chargés', 'semaine_annee', 'trimestre', 'est_weekend', 'est_debut_mois', 'est_fin_mois']

            # Préparer les séquences pour l’entraînement
            X, y = prepare_sequences(df, agency, sequence_length=sequence_length, feature_cols=feature_cols)
            X_scaled, y_scaled, scaler, y_scaler = normalize_data(X, y, fit_scaler=True)
            X_reshaped, y_reshaped = reshape_for_lstm(X_scaled, y_scaled)

            # Entraîner le modèle
            model = create_lstm_model(input_shape=(X_reshaped.shape[1], X_reshaped.shape[2]))
            train_model(model, X_reshaped, y_reshaped)

            # Préparer la séquence pour la prédiction
            date_obj = pd.to_datetime(date)
            df_seq = df_agence[df_agence['date_chargement'] < date_obj].tail(sequence_length)
            if len(df_seq) < sequence_length:
                return jsonify({'error': f'Pas assez de données récentes pour {agency}'}), 400

            seq_features = df_seq[feature_cols].values
            seq_features_scaled = scaler.transform(seq_features)
            X_pred = np.array([seq_features_scaled])

            # Prédire
            y_pred_scaled = model.predict(X_pred, verbose=0)
            y_pred = y_scaler.inverse_transform(y_pred_scaled)[0][0]

            print(f"Prédiction Flask (LSTM): {y_pred:.2f} TND")
            return jsonify({'prediction': float(y_pred)})

        else:
            return jsonify({'error': 'Unsupported model'}), 400

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

