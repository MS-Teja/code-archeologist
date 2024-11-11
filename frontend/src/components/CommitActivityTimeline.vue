<template>
  <div class="commit-activity-timeline">
    <h2>Commit Activity Timeline</h2>
    <div v-if="error" class="error">{{ error }}</div>
    <div v-else-if="isLoading" class="loading">Loading commit activity data...</div>
    <div v-else-if="commitActivity.length === 0" class="no-data">
      No commit activity data available.
    </div>
    <div v-else>
      <LineChart v-if="chartData" :data="chartData" :options="chartOptions" />
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue';
import axios from 'axios';
import { Line } from 'vue-chartjs';
import {
  Chart as ChartJS,
  Title,
  Tooltip,
  Legend,
  LineElement,
  PointElement,
  CategoryScale,
  LinearScale,
} from 'chart.js';

ChartJS.register(Title, Tooltip, Legend, LineElement, PointElement, CategoryScale, LinearScale);

export default {
  name: 'CommitActivityTimeline',
  components: {
    LineChart: Line
  },
  props: {
    analysisId: {
      type: Number,
      required: true
    }
  },
  setup(props) {
    const commitActivity = ref([]);
    const chartData = ref(null);
    const chartOptions = ref({
      responsive: true,
      plugins: {
        legend: {
          position: 'top',
        },
        title: {
          display: true,
          text: 'Commit Activity Timeline',
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

    const prepareChartData = () => {
      if (!commitActivity.value.length) return null;
      const data = commitActivity.value.map(week => ({
        date: new Date(week.week * 1000),
        count: week.total
      }));
      return {
        labels: data.map(d => d.date.toLocaleDateString()),
        datasets: [{
          label: 'Commits',
          data: data.map(d => d.count),
          borderColor: '#2196F3',
          tension: 0.1
        }]
      };
    };

    const fetchData = async () => {
      try {
        const response = await axios.get('/api/commit-activity-timeline', {
          withCredentials: true,
          params: { analysisId: props.analysisId }
        });
        console.log('Commit activity response:', response.data);

        if (response.data.status === 'success') {
          commitActivity.value = response.data.data || [];
          console.log('Assigned commit activity:', commitActivity.value);
        } else {
          throw new Error(response.data.error);
        }
        chartData.value = prepareChartData();
      } catch (err) {
        console.error('Error fetching commit activity data:', err);
        error.value = err.response?.data?.error || 'Failed to load commit activity data.';
      } finally {
        isLoading.value = false;
      }
    };

    onMounted(() => {
      fetchData();
    });

    return {
      chartData,
      chartOptions,
      isLoading,
      error,
      commitActivity,
    };
  },
};
</script>

<style scoped>
h2 {
  text-align: center;
}
.commit-activity-timeline {
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
</style>