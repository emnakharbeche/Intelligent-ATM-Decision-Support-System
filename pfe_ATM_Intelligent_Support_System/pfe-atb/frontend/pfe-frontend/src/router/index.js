import { createRouter, createWebHistory } from 'vue-router';

const routes = [
  { path: '/', redirect: '/login' },
  { path: '/login', name: 'Login', component: () => import('../views/Login.vue') },
  { path: '/signup', name: 'Signup', component: () => import('../views/Signup.vue') },
  { path: '/general', name: 'General', component: () => import('../views/GeneralView.vue'), meta: { requiresAuth: true } },
  { path: '/charges-decharges', name: 'ChargesDecharges', component: () => import('../views/ChargesDecharges.vue'), meta: { requiresAuth: true } },
  { path: '/transactions', name: 'Transactions', component: () => import('../views/Transactions.vue'), meta: { requiresAuth: true } },
  { path: '/disponibilite-atms', name: 'DisponibiliteATMs', component: () => import('../views/DisponibiliteATMs.vue'), meta: { requiresAuth: true } },
  { path: '/statut-atms', name: 'StatutATMs', component: () => import('../views/StatutATMs.vue'), meta: { requiresAuth: true } },
  { path: '/predictions', name: 'Predictions', component: () => import('../views/Predictions.vue'), meta: { requiresAuth: true } },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

router.beforeEach((to, from, next) => {
  const token = localStorage.getItem('token');
  if (to.meta.requiresAuth && !token) {
    next('/login');
  } else {
    next();
  }
});

export default router;