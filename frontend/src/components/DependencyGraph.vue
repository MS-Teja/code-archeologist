<template>
  <div class="dependency-graph">
    <h2>Dependency Graph</h2>
    <div v-if="hasData" id="dependencyGraph" class="cy-container"></div>
    <div v-else-if="noData">No dependency data available.</div>
    <div v-else-if="errorMessage">{{ errorMessage }}</div>
    <div v-else>Loading dependency graph...</div>
  </div>
</template>

<script>
import { ref, onMounted, nextTick } from 'vue';
import axios from 'axios';
import cytoscape from 'cytoscape';
import { store } from '../store/analysis';

export default {
  name: 'DependencyGraph',
  data() {
    return {
      store,
    };
  },
  setup() {
    const hasData = ref(false);
    const noData = ref(false);
    const errorMessage = ref('');
    const dependencies = ref({});

    const fetchData = async () => {
      try {
        const response = await axios.get('/api/dependency-graph', {
          withCredentials: true,
          params: { analysisId: store.analysisId }
        });
        dependencies.value = response.data.data || {};

        if (dependencies.value && Object.keys(dependencies.value).length > 0) {
          hasData.value = true;
          noData.value = false;
          errorMessage.value = '';

          await nextTick();
          renderGraph(buildGraphElements(dependencies.value));
        } else {
          hasData.value = false;
          noData.value = true;
        }
      } catch (error) {
        console.error('Error fetching dependency graph data:', error);
        hasData.value = false;
        errorMessage.value = 'Failed to load dependency data.';
      }
    };

    const buildGraphElements = (deps) => {
      const elements = [];
      const rootNodeId = 'root';
      elements.push({ data: { id: rootNodeId, label: 'Project' } });

      for (const [dep, version] of Object.entries(deps)) {
        const depId = dep;
        elements.push({ data: { id: depId, label: `${dep}@${version}` } });
        elements.push({ data: { source: rootNodeId, target: depId } });
      }

      return elements;
    };

    const renderGraph = (elements) => {
      cytoscape({
        container: document.getElementById('dependencyGraph'),
        elements: elements,
        style: [
          {
            selector: 'node',
            style: {
              label: 'data(label)',
              'text-valign': 'center',
              'text-halign': 'center',
              'background-color': '#0074D9',
              color: '#fff',
              width: 100, // Increase node size
              height: 100,
              'text-wrap': 'wrap',
              'text-max-width': 90, // Allow text wrapping
              'font-size': 10,
              padding: '10px'
            },
          },
          {
            selector: 'edge',
            style: {
              'curve-style': 'bezier',
              'target-arrow-shape': 'triangle',
              width: 2,
              'line-color': '#ccc',
              'target-arrow-color': '#ccc',
            },
          },
        ],
        layout: {
          name: 'breadthfirst',
        },
      });
    };

    onMounted(() => {
      fetchData();
    });

    return {
      hasData,
      noData,
      errorMessage,
    };
  },
};
</script>

<style scoped>
.dependency-graph {
  max-width: 800px;
  margin: 0 auto;
}

h2{
  text-align: center;
}

.dependency-graph .cy-container {
  width: 100%;
  height: 600px;
}

.error-message {
  color: red;
  text-align: center;
  margin-top: 20px;
}

.no-data-message {
  color: #555;
  text-align: center;
  margin-top: 20px;
}

.loading-message {
  text-align: center;
  margin-top: 20px;
}
</style>