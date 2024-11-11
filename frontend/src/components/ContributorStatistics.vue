<template>
  <div class="contributor-statistics">
    <h2>Contributor Statistics</h2>
    <div v-if="error" class="error">{{ error }}</div>
    <div v-else-if="isLoading" class="loading">Loading contributor statistics...</div>
    <div v-else-if="contributors.length === 0" class="no-data">
      No contributor data available.
    </div>
    <div v-else>
      <table>
        <thead>
          <tr>
            <th>Contributor</th>
            <th>Commits</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="contributor in contributors" :key="contributor.login">
            <td>{{ contributor.login }}</td>
            <td>{{ contributor.contributions }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue';
import axios from 'axios';
import { store } from '../store/analysis';

export default {
  name: 'ContributorStatistics',
  props: {
    analysisId: {
      type: Number,
      required: true
    }
  },
  setup(props) {
    const contributors = ref([]);
    const isLoading = ref(true);
    const error = ref(null);

    const fetchData = async () => {
      try {
        const response = await axios.get('/api/contributor-statistics', {
          withCredentials: true,
          params: { analysisId: props.analysisId }
        });
        // Update to handle the new response structure
        if (response.data.status === 'success') {
          contributors.value = response.data.data || [];
        } else {
          throw new Error(response.data.error);
        }
      } catch (err) {
        console.error('Error fetching contributor statistics:', err);
        error.value = err.response?.data?.error || 'Failed to load contributor statistics.';
      } finally {
        isLoading.value = false;
      }
    };

    onMounted(() => {
      fetchData();
    });

    return {
      contributors,
      isLoading,
      error,
    };
  },
};
</script>

<style scoped>
h2 {
  text-align: center;
}

.contributor-statistics table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 20px;
}

.contributor-statistics th,
.contributor-statistics td {
  border: 1px solid #ddd;
  padding: 8px;
  text-align: left;
}

.contributor-statistics th {
  background-color: #f2f2f2;
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