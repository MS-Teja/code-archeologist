<template>
  <div class="file-change-frequency container">
    <h2>File Change Frequency</h2>
    <div v-if="error" class="error">{{ error }}</div>
    <div v-else-if="isLoading" class="loading">Loading file change frequency...</div>
    <div v-else-if="!chartData" class="no-data">
      No file change data available.
    </div>
    <div v-else class="chart-container container">
      <Bar :data="chartData" :options="chartOptions" />
      <br />
      <i>Hover over the graph for more details</i>
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue';
import axios from 'axios';
import { Bar } from 'vue-chartjs';
import { store } from '../store/analysis';
import {
  Chart as ChartJS,
  Title,
  Tooltip,
  Legend,
  BarElement,
  CategoryScale,
  LinearScale,
} from 'chart.js';

ChartJS.register(Title, Tooltip, Legend, BarElement, CategoryScale, LinearScale);

export default {
  name: 'FileChangeFrequency',
  data() {
    return {
      store,
    };
  },
  components: {
    Bar,
  },
  setup() {
    const fileChanges = ref({});
    const chartData = ref(null);
    const chartOptions = ref({
      responsive: true,
      plugins: {
        legend: {
          position: 'top',
        },
        title: {
          display: true,
          text: 'File Change Frequency',
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          ticks: {
            precision: 0,
          },
        },
      },
    });
    const isLoading = ref(true);
    const error = ref(null);

    const fetchData = async () => {
      try {
        const response = await axios.get('/api/file-change-frequency', {
          withCredentials: true,
          params: { analysisId: store.analysisId }
        });
        fileChanges.value = response.data.data || {};
        prepareChartData();
      } catch (err) {
        console.error('Error fetching file change frequency data:', err);
        error.value = 'Failed to load file change frequency data.';
      } finally {
        isLoading.value = false;
      }
    };

    const prepareChartData = () => {
      // Add null check
      if (!fileChanges.value) {
        chartData.value = null;
        return;
      }
      const labels = Object.keys(fileChanges.value);
      const data = Object.values(fileChanges.value);
      if (labels.length === 0) {
        chartData.value = null;
        return;
      }
      chartData.value = {
        labels: labels,
        datasets: [
          {
            label: 'Number of Changes',
            data: data,
            backgroundColor: 'rgba(75, 192, 192, 0.6)',
          },
        ],
      };
    };

    onMounted(() => {
      fetchData();
    });

    return {
      chartData,
      chartOptions,
      isLoading,
      error,
    };
  },
};
</script>

<style scoped>
.container {
  display: flex;
  flex-direction: column;
  align-items: center;
}
.chart-container {
  width: 100%;
  height: 600px;
  max-width: 1200px;
  margin: 0 auto;
}
i {
  font-style: italic;
  font-weight: 100;
  font-size: smaller;
  padding: 20px;
}
.file-change-frequency {
  max-width: 800px;
  margin: 0 auto;
}
.no-data,
.error,
.loading {
  text-align: center;
  color: #666;
  font-style: italic;
  margin-top: 20px;
}
.error {
  color: #e53935;
}
.loading {
  color: #2196F3;
}
</style>