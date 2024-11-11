<template>
  <div id="app">
    <h1>Code Archeologist</h1>
    <AnalysisDashboard />
  </div>
</template>

<script>
import AnalysisDashboard from './components/AnalysisDashboard.vue';

export default {
  components: {
    AnalysisDashboard
  }
};
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

h1{
  text-align: center;
}
</style>