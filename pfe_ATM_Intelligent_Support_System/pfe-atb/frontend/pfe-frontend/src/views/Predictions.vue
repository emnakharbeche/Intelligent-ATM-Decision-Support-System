<template>
  <div class="container mt-5">
    <h2>Prédictions</h2>

    <!-- Étape 1 : Affichage des carreaux pour les modèles -->
    <div v-if="!selectedModel">
      <div class="row">
        <div class="col-md-3 mb-4" v-for="model in models" :key="model.name">
          <div class="card model-card" @click="selectModel(model.name)">
            <div class="card-body text-center">
              <h5 class="card-title">{{ model.name }}</h5>
              <p class="card-text">{{ model.description }}</p>
            </div>
          </div>
        </div>
      </div>

      <!-- Visualisations côte à côte -->
      <div class="row mt-4">
        <div class="col-md-6">
          <h4>Évolution des Montants Chargés (par Mois)</h4>
          <div class="chart-container">
            <canvas id="lineChart"></canvas>
          </div>
        </div>
        <div class="col-md-6">
          <h4>Montants Chargés par Agence</h4>
          <div class="chart-container">
            <canvas id="barChart"></canvas>
          </div>
        </div>
      </div>
    </div>

    <!-- Étape 2 : Formulaire après sélection d’un modèle -->
    <div v-if="selectedModel && !predictionResult">
      <div class="card p-4">
        <h4>Prédiction avec {{ selectedModel }}</h4>
        <div class="row">
          <div class="col-md-6">
            <div class="form-group">
              <label for="date">Date de Chargement :</label>
              <input type="date" id="date" v-model="formData.date" class="form-control" :min="minDate" required />
            </div>
          </div>
          <div class="col-md-6">
            <div class="form-group">
              <label for="agency">Nom d’Agence :</label>
              <select
                id="agency"
                v-model="formData.agency"
                class="form-control"
                :disabled="selectedModel === 'SARIMA'"
                required
              >
                <option value="" disabled>Sélectionner une agence</option>
                <option v-for="agency in agencies" :key="agency" :value="agency">
                  {{ agency }}
                </option>
              </select>
            </div>
          </div>
        </div>
        <button class="btn btn-primary mt-3" @click="submitPrediction" :disabled="isLoading">
          <span v-if="isLoading" class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
          {{ isLoading ? 'Chargement...' : 'Prédire' }}
        </button>
        <button class="btn btn-secondary mt-3 ms-2" @click="resetForm" :disabled="isLoading">Retour</button>
      </div>
    </div>

    <!-- Étape 3 : Résultat de la prédiction -->
    <div v-if="predictionResult">
      <div class="card p-4">
        <h4>
          Résultat de la Prédiction avec {{ selectedModel }}
          <span v-if="selectedModel !== 'SARIMA'"> pour {{ formData.agency }}</span>
        </h4>
        <p><strong>Valeur Prédite :</strong> {{ predictionResult.toFixed(2) }} TND</p>
        <!-- Affichage des métriques -->
        <div class="metrics-section">
          <h5>Métriques de Performance du Modèle</h5>
          <p><strong>MAE :</strong> {{ modelMetrics.mae_normalized.toFixed(2) }}</p>
          <p><strong>MSE :</strong> {{ modelMetrics.mse_normalized.toFixed(2) }}</p>
          <p><strong>RMSE :</strong> {{ modelMetrics.rmse_normalized.toFixed(2) }}</p>
          <p><strong>MAPE :</strong> {{ modelMetrics.mape.toFixed(2) }}%</p>
        </div>
        <h5>Comparaison avec les Valeurs Réelles</h5>
        <div class="chart-container">
          <canvas id="comparisonChart"></canvas>
        </div>
        <button class="btn btn-primary mt-3" @click="resetPrediction">Nouvelle Prédiction</button>
      </div>
    </div>
  </div>
</template>

<script>
import Chart from 'chart.js/auto';
import 'chartjs-adapter-date-fns';

export default {
  name: 'Predictions',
  data() {
    return {
      models: [
        { name: 'LSTM', description: 'Réseau neuronal pour séries temporelles' },
        { name: 'Random Forest', description: 'Forêt aléatoire pour prédictions robustes' },
        { name: 'SARIMA', description: 'Modèle de série temporelle saisonnière' },
        { name: 'Régression Linéaire', description: 'Modèle linéaire simple' },
      ],
    /*ydeclari fihom ferghin hata tjihom ml app.py ml route */
      agencies: [],  
      selectedModel: null,
      formData: {
        date: '',
        agency: '',
      },
      historicalData: [],
      predictionResult: null,
      modelMetrics: {
        mae_normalized: 0,
        mse_normalized: 0,
        rmse_normalized: 0,
        mape: 0
      },
      minDate: '',
      lineChart: null,
      barChart: null,
      comparisonChart: null,
      isLoading: false,
    };
  },
  mounted() {
    this.fetchHistoricalData();
    this.fetchAgencies();
    const today = new Date().toISOString().split('T')[0];
    this.minDate = today;
  },
  methods: {
    async fetchHistoricalData() {
      try {
        const response = await fetch('http://localhost:5000/historical-data');
        const data = await response.json();
        if (!Array.isArray(data) || data.length === 0) {
          console.warn('Aucune donnée historique récupérée ou format invalide:', data);
          return;
        }
        this.historicalData = data;
        this.renderCharts();
      } catch (error) {
        console.error('Erreur lors de la récupération des données historiques:', error);
      }
    },
    async fetchAgencies() {
      try {
        const response = await fetch('http://localhost:5000/agencies');
        this.agencies = await response.json();
        console.log('Agences récupérées:', this.agencies);
      } catch (error) {
        console.error('Erreur lors de la récupération des agences:', error);
      }
    },
    renderCharts() {
      const lineCanvas = document.getElementById('lineChart');
      const barCanvas = document.getElementById('barChart');

      if (!lineCanvas || !barCanvas) {
        console.warn('Les éléments canvas pour lineChart ou barChart ne sont pas disponibles dans le DOM.');
        return;
      }

      const monthlyData = {};
      this.historicalData.forEach(data => {
        const month = data.date.slice(0, 7);
        if (!monthlyData[month]) {
          monthlyData[month] = 0;
        }
        monthlyData[month] += data.montant || 0;
      });

      const lineCtx = lineCanvas.getContext('2d');
      if (this.lineChart) this.lineChart.destroy();
      this.lineChart = new Chart(lineCtx, {
        type: 'line',
        data: {
          labels: Object.keys(monthlyData),
          datasets: [{
            label: 'Montant Chargé (Réel)',
            data: Object.values(monthlyData),
            borderColor: '#800020',
            fill: false,
          }],
        },
        options: {
          responsive: true,
          scales: {
            x: { title: { display: true, text: 'Mois' } },
            y: { title: { display: true, text: 'Montant Chargé (TND)' } },
          },
        },
      });

      const agenciesData = {};
      this.historicalData.forEach(data => {
        if (data.agence) {
          if (!agenciesData[data.agence]) {
            agenciesData[data.agence] = 0;
          }
          agenciesData[data.agence] += data.montant || 0;
        }
      });

      const barCtx = barCanvas.getContext('2d');
      if (this.barChart) this.barChart.destroy();
      this.barChart = new Chart(barCtx, {
        type: 'bar',
        data: {
          labels: Object.keys(agenciesData),
          datasets: [{
            label: 'Montant Chargé (TND)',
            data: Object.values(agenciesData),
            backgroundColor: '#800020',
          }],
        },
        options: {
          responsive: true,
          scales: {
            x: { title: { display: true, text: 'Agence' } },
            y: { title: { display: true, text: 'Montant Chargé (TND)' } },
          },
        },
      });
    },
    selectModel(model) {
      this.selectedModel = model;
    },
    resetForm() {
      this.selectedModel = null;
      this.formData = { date: '', agency: '' };
    },
    async submitPrediction() {
      if (!this.formData.date) {
        alert('Veuillez sélectionner une date.');
        return;
      }
      if (this.selectedModel !== 'SARIMA' && (!this.formData.agency || this.formData.agency.trim() === '')) {
        alert('Veuillez sélectionner une agence.');
        return;
      }
      try {
        this.isLoading = true;
        console.log('Envoi de la requête de prédiction:', {
          model: this.selectedModel,
          date: this.formData.date,
          agency: this.selectedModel === 'SARIMA' ? null : this.formData.agency,
        });
        const response = await fetch('http://localhost:5000/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: this.selectedModel,
            date: this.formData.date,
            agency: this.selectedModel === 'SARIMA' ? null : this.formData.agency,
          }),
        });
        const result = await response.json();
        if (response.ok) {
          this.predictionResult = result.prediction;
          this.modelMetrics = result.metrics || {
            mae_normalized: 0,
            mse_normalized: 0,
            rmse_normalized: 0,
            mape: 0
          };
          console.log('Prédiction réussie pour', this.selectedModel, ':', this.predictionResult);
          console.log('Métriques récupérées:', this.modelMetrics);
          this.$nextTick(() => {
            this.renderComparisonChart();
          });
        } else {
          console.error('Erreur de prédiction:', result.error);
          alert('Échec de la prédiction: ' + result.error);
        }
      } catch (error) {
        console.error('Erreur lors de la requête:', error);
        alert('Erreur lors de la prédiction: ' + error.message);
      } finally {
        this.isLoading = false;
      }
    },
    renderComparisonChart() {
      try {
        const canvas = document.getElementById('comparisonChart');
        if (!canvas) {
          console.error('Canvas comparisonChart introuvable dans le DOM');
          return;
        }
        const ctx = canvas.getContext('2d');
        if (this.comparisonChart) this.comparisonChart.destroy();

        const selectedDate = new Date(this.formData.date);
        const selectedMonth = selectedDate.getMonth() + 1;
        const selectedYear = selectedDate.getFullYear();

        let filteredData = this.historicalData;
        if (this.selectedModel !== 'SARIMA' && this.formData.agency) {
          filteredData = filteredData.filter(data => data.agence === this.formData.agency);
        }

        filteredData = filteredData.filter(data => {
          const dataDate = new Date(data.date);
          const dataMonth = dataDate.getMonth() + 1;
          const dataYear = dataDate.getFullYear();
          return dataMonth === selectedMonth && dataYear <= selectedYear;
        });

        if (this.selectedModel === 'SARIMA') {
          const aggregatedData = {};
          filteredData.forEach(data => {
            const date = data.date;
            if (!aggregatedData[date]) {
              aggregatedData[date] = 0;
            }
            aggregatedData[date] += data.montant || 0;
          });

          const labels = Object.keys(aggregatedData).sort();
          const values = labels.map(date => aggregatedData[date]);

          labels.push(this.formData.date);
          values.push(this.predictionResult);

          this.comparisonChart = new Chart(ctx, {
            type: 'bar',
            data: {
              labels: labels,
              datasets: [
                {
                  label: 'Montant Chargé (TND)',
                  data: values,
                  backgroundColor: labels.map((label, index) =>
                    index === labels.length - 1 ? '#1A2A44' : '#800020'
                  ),
                  borderColor: labels.map((label, index) =>
                    index === labels.length - 1 ? '#1A2A44' : '#800020'
                  ),
                  borderWidth: 1,
                },
              ],
            },
            options: {
              responsive: true,
              scales: {
                x: {
                  title: { display: true, text: 'Date' },
                  ticks: { maxTicksLimit: 10 },
                },
                y: {
                  title: { display: true, text: 'Montant Chargé (TND)' },
                  beginAtZero: true,
                },
              },
              plugins: {
                legend: {
                  labels: {
                    generateLabels: chart => [
                      { text: 'Montant Chargé (TND)', fillStyle: '#800020' },
                      { text: 'Montant Chargé Prédit', fillStyle: '#1A2A44' },
                    ],
                  },
                },
                tooltip: { mode: 'index', intersect: false },
              },
            },
          });
        } else {
          const aggregatedData = {};
          filteredData.forEach(data => {
            const date = data.date;
            if (!aggregatedData[date]) {
              aggregatedData[date] = 0;
            }
            aggregatedData[date] += data.montant || 0;
          });

          const labels = Object.keys(aggregatedData).sort();
          const values = labels.map(date => aggregatedData[date]);

          labels.push(this.formData.date);
          values.push(this.predictionResult);

          this.comparisonChart = new Chart(ctx, {
            type: 'bar',
            data: {
              labels: labels,
              datasets: [
                {
                  label: 'Montant Chargé (TND)',
                  data: values,
                  backgroundColor: labels.map((label, index) =>
                    index === labels.length - 1 ? '#1A2A44' : '#800020'
                  ),
                  borderColor: labels.map((label, index) =>
                    index === labels.length - 1 ? '#1A2A44' : '#800020'
                  ),
                  borderWidth: 1,
                },
              ],
            },
            options: {
              responsive: true,
              scales: {
                x: {
                  title: { display: true, text: 'Date' },
                  ticks: { maxTicksLimit: 10 },
                },
                y: {
                  title: { display: true, text: 'Montant Chargé (TND)' },
                  beginAtZero: true,
                },
              },
              plugins: {
                legend: {
                  labels: {
                    generateLabels: chart => [
                      { text: 'Montant Chargé (TND)', fillStyle: '#800020' },
                      { text: 'Montant Chargé Prédit', fillStyle: '#1A2A44' },
                    ],
                  },
                },
                tooltip: { mode: 'index', intersect: false },
              },
            },
          });
        }
      } catch (error) {
        console.error('Erreur lors du rendu du graphique de comparaison:', error);
      }
    },
    resetPrediction() {
      this.predictionResult = null;
      this.modelMetrics = {
        mae_normalized: 0,
        mse_normalized: 0,
        rmse_normalized: 0,
        mape: 0
      };
      this.formData = { date: '', agency: '' };
      this.selectedModel = null;

      if (this.comparisonChart) {
        this.comparisonChart.destroy();
        this.comparisonChart = null;
      }
      if (this.lineChart) {
        this.lineChart.destroy();
        this.lineChart = null;
      }
      if (this.barChart) {
        this.barChart.destroy();
        this.barChart = null;
      }

      this.$nextTick(() => {
        this.$nextTick(() => {
          const lineCanvas = document.getElementById('lineChart');
          const barCanvas = document.getElementById('barChart');
          if (lineCanvas && barCanvas) {
            this.renderCharts();
          } else {
            console.warn('Les éléments canvas ne sont pas encore disponibles dans le DOM après resetPrediction.');
          }
        });
      });
    },
  },
  beforeDestroy() {
    if (this.lineChart) this.lineChart.destroy();
    if (this.barChart) this.barChart.destroy();
    if (this.comparisonChart) this.comparisonChart.destroy();
  },
};
</script>

<style scoped>
.model-card {
  cursor: pointer;
  transition: transform 0.2s;
  background-color: #ffffff;
  border: 1px solid #800020;
}

.model-card:hover {
  transform: scale(1.05);
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

.card {
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.form-group {
  margin-bottom: 15px;
}

.form-control, select {
  border-color: #800020;
}

.form-control:focus, select:focus {
  border-color: #a00028;
  box-shadow: 0 0 5px rgba(160, 0, 40, 0.3);
}

.btn-primary {
  background-color: #800020;
  border-color: #800020;
}

.btn-primary:hover {
  background-color: #a00028;
  border-color: #a00028;
}

.btn-secondary {
  background-color: #6c757d;
  border-color: #6c757d;
}

.btn-secondary:hover {
  background-color: #5a6268;
  border-color: #5a6268;
}

.chart-container {
  position: relative;
  height: 400px;
  margin-bottom: 20px;
}

.spinner-border {
  margin-right: 5px;
}

.metrics-section {
  margin: 15px; /* Augmente la marge supérieure pour plus d'espacement et centre le bloc */
  padding: 15px; /* Padding plus généreux pour un aspect aéré */
  background-color: #f8f9fa;
  border-radius: 5px;
  max-width: 70%; /* Limite la largeur pour un meilleur centrage */
  border: 1px solid #800020;
}

.metrics-section h5 {
  font-size: 1.2rem; /* Taille légèrement plus grande pour le titre */
  margin-bottom: 15px; /* Plus d'espace sous le titre */
  color: #333; /* Couleur plus sombre pour le contraste */
  font-weight: 600; /* Légère augmentation du poids pour plus d'élégance */
  
}

.metrics-section p {
  margin-bottom: 8px; /* Plus d'espace entre les lignes de métriques */
  font-size: 1rem; /* Taille légèrement augmentée pour les métriques */
  color: #444; /* Couleur légèrement plus foncée pour la lisibilité */
  line-height: 1.3; /* Espacement des lignes pour une meilleure lisibilité */
  font-family: 'Open Sans', sans-serif
}

/* Style pour le titre "Comparaison avec les Valeurs Réelles" */
h5 {
  font-size: 1.2rem; /* Même taille que le titre des métriques pour la cohérence */
  color: #333; /* Même couleur sombre pour le contraste */
  font-weight: 600; /* Même poids pour la cohérence */
  margin-top: 2px; /* Augmente l'espace au-dessus pour séparer du bloc gris */
  margin-bottom: 20px; /* Espace sous le titre pour la lisibilité */
}
</style>