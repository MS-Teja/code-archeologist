<template>
  <div class="commit-summary">
    <div class="container">
      <h2>Repository Summary</h2>
      <button @click="generateSummary" :disabled="loading">
        Generate Summary
      </button>
    </div>
    <div v-if="loading" class="loading">Generating summary...</div>
    <div v-else-if="error" class="error">{{ error }}</div>
    <div v-else-if="summary" class="summary" v-html="formattedSummary"></div>
  </div>
</template>

<script>
import { marked } from 'marked';
import{ store } from '../store/analysis';
import axios from 'axios';

export default {
  name: 'CommitSummary',
  data() {
    return {
        summary: null,
        loading: false,
        error: null,
        store
    }
  },
  computed: {
    formattedSummary() {
      return this.summary ? marked(this.summary) : '';
    }
  },
  methods: {
    async generateSummary() {
        this.loading = true;
        try {
            const response = await axios.post('/api/summarize', {}, {
            withCredentials: true,
            headers: { 'X-Analysis-ID': store.analysisId }
            });
            this.summary = response.data.summary;
        } catch (error) {
            this.error = error.response?.data?.message || 'Failed to generate summary';
        } finally {
            this.loading = false;
        }
    }
  }
}
</script>

<style scoped>
.container {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.ai-features {
  margin-top: 20px;
  padding: 20px;
  background: #f8f9fa;
  border-radius: 8px;
}

.search-box, .question-box {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
}

.commit-card {
  background: white;
  padding: 15px;
  margin: 10px 0;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.answer, .summary {
  background: white;
  padding: 20px;
  border-radius: 8px;
  margin-top: 20px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

button {
  padding: 12px 24px;
  background-color: #2196F3;
  color: #fff;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 16px;
  transition: background-color 0.3s, transform 0.2s, box-shadow 0.3s;
}

button:hover {
  background-color: #1976D2;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

button:active {
  transform: scale(0.98);
}

button:disabled {
  background-color: #90CAF9;
  cursor: not-allowed;
  opacity: 0.7;
}
</style>