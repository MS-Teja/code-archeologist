<template>
  <div class="issue-integration">
    <h2>Issue Tracking</h2>
    <div v-if="error" class="error">{{ error }}</div>
    <div v-else-if="isLoading" class="loading">Loading issues...</div>
    <div v-else-if="issues.length === 0" class="no-issues">
      No issues available.
    </div>
    <div v-else>
      <ul>
        <li v-for="issue in issues" :key="issue.id">
          <a :href="issue.html_url" target="_blank">{{ issue.title }}</a>
          <span :class="['status', issue.state]">{{ issue.state }}</span>
        </li>
      </ul>
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue';
import axios from 'axios';
import { store } from '../store/analysis';

export default {
  name: 'IssueIntegration',
  data() {
    return {
      store,
    };
  },
  setup() {
    const issues = ref([]);
    const isLoading = ref(true);
    const error = ref(null);

    const fetchData = async () => {
      try {
        const response = await axios.get('/api/linked-issues', {
          withCredentials: true,
          params: { analysisId: store.analysisId } // Add params
        });
        issues.value = response.data.data || [];
      } catch (err) {
        error.value = 'Failed to load issues.';
      } finally {
        isLoading.value = false;
      }
    };

    onMounted(() => {
      fetchData();
    });

    return {
      issues,
      isLoading,
      error,
    };
  },
};
</script>

<style scoped>
.issue-integration ul {
  list-style: none;
  padding: 0;
}

.issue-integration li {
  margin: 10px 0;
}

.issue-integration a {
  text-decoration: none;
  color: #1e88e5;
}

.no-issues,
.error,
.loading {
  text-align: center;
  color: #666;
  font-style: italic;
  margin-top: 20px;
}

.status {
  margin-left: 10px;
  padding: 2px 6px;
  border-radius: 4px;
  color: #fff;
  font-size: 0.8em;
}

.status.open {
  background-color: #43a047;
}

.status.closed {
  background-color: #e53935;
}

.error {
  color: #e53935;
}

.loading {
  color: #2196F3;
}
</style>