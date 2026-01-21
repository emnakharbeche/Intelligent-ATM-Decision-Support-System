<template>
  <div class="container mt-5">
    <div class="d-flex justify-content-between align-items-center mb-3">
      <h2 class="title">Tableaux de bord de la disponibilité des ATMs</h2>
      <button class="btn btn-danger" @click="refreshDashboard">Rafraîchir</button>
    </div>
    <div class="iframe-container">
      <iframe
        :key="refreshKey"
        :src="embedUrl"
        frameborder="0"
        allowFullScreen="true"
      ></iframe>
    </div>
  </div>
</template>

<script>
export default {
  name: 'DisponibiliteATMs',
  data() {
    return {
      embedUrl: 'https://app.powerbi.com/reportEmbed?reportId=1290e56c-7fe7-4c91-a66f-506bd667ef83&autoAuth=true&ctid=dbd6664d-4eb9-46eb-99d8-5c43ba153c61',
      refreshKey: 0,
    };
  },
  methods: {
    refreshDashboard() {
      this.refreshKey += 1;
    },
  },
};
</script>

<style scoped>
.container {
  max-width: 1200px;
  margin: 0 auto;
}

.title {
  font-family: 'Alice', serif;
  color: #800020;
  margin: 0;
}

.iframe-container {
  position: relative;
  padding-bottom: 56.25%;
  height: 0;
  overflow: hidden;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  margin: 0 auto;
  max-width: 1000px;
}

.iframe-container iframe {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  border: none;
}

.btn-danger {
  background-color: #800020;
  border-color: #800020;
}

.btn-danger:hover {
  background-color: #a00028;
  border-color: #a00028;
}
</style>