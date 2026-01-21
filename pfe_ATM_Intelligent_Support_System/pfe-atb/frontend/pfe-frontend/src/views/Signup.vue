<template>
  <div class="login-container">
    <div class="content-wrapper">
      <div class="form-section">
        <h2>Inscription</h2>
        <form @submit.prevent="handleSignup">
          <div class="mb-3">
            <label for="username" class="form-label">Nom d'utilisateur*</label>
            <input v-model="username" type="text" class="form-control" id="username" required />
          </div>
          <div class="mb-3">
            <label for="email" class="form-label">Email*</label>
            <input v-model="email" type="email" class="form-control" id="email" required />
          </div>
          <div class="mb-3">
            <label for="password" class="form-label">Mot de passe*</label>
            <input v-model="password" type="password" class="form-control" id="password" required />
          </div>
          <button type="submit" class="btn btn-primary">Inscription</button>
          <p v-if="error" class="text-danger mt-2">{{ error }}</p>
          <p class="mt-2">
            Déjà un compte ? <router-link to="/login">Connectez-vous</router-link>
          </p>
        </form>
      </div>
      <div class="image-section">
        <img src="@/assets/atb-home.jpg" alt="ATB Home" class="home-image" />
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  name: 'SignupView',
  data() {
    return {
      username: '',
      email: '',
      password: '',
      error: '',
    };
  },
  methods: {
    async handleSignup() {
      try {
        await axios.post('http://localhost:5000/signup', {
          username: this.username,
          email: this.email,
          password: this.password,
        });
        this.$router.push('/login');
      } catch (err) {
        this.error = err.response?.data?.message || "Échec de l'inscription";
      }
    },
  },
};
</script>

<style scoped>
/* Identique à login.vue */
.login-container {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background-color: #f5f5f5;
}

.content-wrapper {
  display: flex;
  flex: 1;
}

.form-section {
  flex: 1;
  padding: 40px;
  background-color: #ffffff;
  display: flex;
  flex-direction: column;
  justify-content: center;
  border-radius: 8px 0 0 8px;
}

h2 {
  color: #800020;
  font-family: 'Alice', serif;
  margin-bottom: 20px;
  text-align: center;     /* Centrage du texte */
  font-weight: bold;      /* Texte en gras */
  font-size: 45px;
}


.form-label {
  font-family: 'Alice', serif;
  color: #333;
}

.form-control {
  font-family: 'Alice', serif;
}

.btn-primary {
  background-color: #800020;
  border-color: #800020;
  font-family: 'Alice', serif;
  width: 100%;
  padding: 10px;
}

.btn-primary:hover {
  background-color: #a00028;
  border-color: #a00028;
}

.text-danger {
  font-family: 'Alice', serif;
}

.image-section {
  flex: 1;
  display: flex;
  justify-content: flex-end;
  align-items: center;
  overflow: hidden;
  position: relative;
  border-radius: 0 0 0 0; /* arrondi à gauche uniquement */
  box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.1);
}

.home-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transform: scale(1.05);
  border-radius: 0 0 0 0; /* arrondi à gauche uniquement */
}

@media (max-width: 768px) {
  .content-wrapper {
    flex-direction: column;
  }

  .image-section {
    height: 300px;
    border-radius: 0 0 8px 8px;
  }

  .form-section {
    border-radius: 8px 8px 0 0;
  }

  .home-image {
    max-height: 300px;
    border-radius: 0 0 8px 8px;
  }
}
</style>
