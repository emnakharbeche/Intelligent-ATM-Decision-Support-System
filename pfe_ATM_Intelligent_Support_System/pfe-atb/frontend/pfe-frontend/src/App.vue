<template>
  <div>
    <nav v-if="isLoggedIn" class="navbar navbar-expand-lg navbar-dark custom-navbar">
      <div class="container-fluid">
        <a class="navbar-brand" href="#">
          <img src="@/assets/atb-logo.png" alt="ATB Logo" class="logo" />
        </a>
        <button
          class="navbar-toggler"
          type="button"
          data-bs-toggle="collapse"
          data-bs-target="#navbarNav"
          aria-controls="navbarNav"
          aria-expanded="false"
          aria-label="Toggle navigation"
        >
          <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbarNav">
          <ul class="navbar-nav">
            <li class="nav-item">
              <router-link class="nav-link" to="/general">Vue Générale</router-link>
            </li>
            <li class="nav-item">
              <router-link class="nav-link" to="/charges-decharges">Charges/Décharges</router-link>
            </li>
            <li class="nav-item">
              <router-link class="nav-link" to="/transactions">Transactions</router-link>
            </li>
            <li class="nav-item">
              <router-link class="nav-link" to="/disponibilite-atms">Disponibilité ATMs</router-link>
            </li>
            <li class="nav-item">
              <router-link class="nav-link" to="/statut-atms">Statut ATMs</router-link>
            </li>
            <li class="nav-item">
              <router-link class="nav-link" to="/predictions">Prédictions</router-link>
            </li>
            <li class="nav-item ms-auto">
              <button class="btn btn-outline-light" @click="logout">Déconnexion</button>
            </li>
          </ul>
        </div>
      </div>
    </nav>
    <router-view />
  </div>
</template>

<script>
export default {
  name: 'App',
  data() {
    return {
      isLoggedIn: !!localStorage.getItem('token'), // État initial basé sur le token
    };
  },
  methods: {
    logout() {
      localStorage.removeItem('token');
      this.isLoggedIn = false; // Mettre à jour l'état réactif
      this.$router.push('/login');
    },
    updateLoginStatus() {
      this.isLoggedIn = !!localStorage.getItem('token'); // Mettre à jour l'état
    },
  },
  watch: {
    '$route'() {
      this.updateLoginStatus(); // Mettre à jour l'état à chaque changement de route
    },
  },
  created() {
    this.updateLoginStatus(); // Vérifier l'état au démarrage
  },
};
</script>

<style scoped>
.custom-navbar {
  background-color: #800020; /* Rouge bordeaux */
}

.logo {
  height: 40px; /* Ajustez la taille selon votre logo */
  width: auto;
}

.nav-link {
  font-family: 'Alice', serif; /* Utilisation de Roboto */
  color: #ffffff !important; /* Blanc pour les liens */
}

.nav-link:hover {
  color: #f0e68c !important; /* Couleur au survol (jaune pâle pour contraste) */
}

.btn-outline-light {
  border-color: #ffffff;
  color: #ffffff;
  font-family: 'Alice', serif; /* Utilisation de Roboto */
}

.btn-outline-light:hover {
  background-color: #ffffff;
  color: #800020;
}
</style>

<style>
/* Styles globaux pour le thème */
body {
  background-color: #f5f5f5; /* Fond clair pour le reste de l’application */
  font-family: 'Alice', serif; /* Police globale */
}

.container {
  background-color: #ffffff; /* Fond blanc pour les conteneurs */
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  max-width: 1200px; /* Limiter la largeur */
  margin: 0 auto; /* Centrer le conteneur */
}

h2 {
  color: #800020; /* Titres en rouge bordeaux */
  font-family: 'Alice', serif; /* Police pour les titres */
}

.btn-primary {
  background-color: #800020;
  border-color: #800020;
  font-family: 'Alice', serif; /* Police pour les boutons */
}

.btn-primary:hover {
  background-color: #a00028; /* Légère variation au survol */
  border-color: #a00028;
}

.btn-danger {
  background-color: #800020;
  border-color: #800020;
  font-family: 'Alice', serif; /* Police pour les boutons danger */
}

.btn-danger:hover {
  background-color: #a00028;
  border-color: #a00028;
}
</style>