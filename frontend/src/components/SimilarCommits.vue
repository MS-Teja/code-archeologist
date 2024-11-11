<template>
  <div class="similar-commits">
    <div class="container">
      <h2>Search Similar Commits</h2>
      <div class="search-box">
        <input
          v-model="searchQuery"
          placeholder="Search commits..."
          @keyup.enter="searchCommits"
        />
        <button
          @click="searchCommits"
          :disabled="!searchQuery.trim() || loading"
        >
          {{ loading ? 'Searching...' : 'Search' }}
        </button>
      </div>
    </div>

    <div v-if="error" class="error">
      {{ error }}
    </div>

    <div v-if="loading" class="loading">
      Searching for similar commits...
    </div>

    <div v-else-if="commits.length" class="results">
      <div v-for="commit in commits" :key="commit.commit_hash" class="commit-card">
        <div class="commit-hash">{{ commit.commit_hash.substring(0, 7) }}</div>
        <div class="commit-message">{{ commit.commit_message }}</div>
        <div class="similarity">
          Similarity: {{ (commit.similarity * 100).toFixed(1) }}%
        </div>
      </div>
    </div>

    <div v-else-if="hasSearched" class="no-results">
      No similar commits found
    </div>
  </div>
</template>

<script>
import { store } from '../store/analysis';
import axios from 'axios';

export default {
  name: 'SimilarCommits',
  data() {
    return {
      searchQuery: '',
      commits: [],
      loading: false,
      error: null,
      hasSearched: false,
      store
    }
  },
  methods: {
    async searchCommits() {
      if (!this.searchQuery.trim()) return;

      this.loading = true;
      this.error = null;
      this.commits = [];
      this.hasSearched = true;

      try {
        const response = await axios.get(`/api/search-commits?query=${encodeURIComponent(this.searchQuery)}`, {
          withCredentials: true,
          headers: {
            'X-Analysis-ID': store.analysisId
          }
        });

        if (response.data.status === 'success') {
          this.commits = response.data.results;
        } else {
          throw new Error(response.data.message);
        }
      } catch (error) {
        console.error('Search error:', error);
        this.error = error.response?.data?.message || 'Failed to search commits';
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

.similar-commits {
  padding: 20px;
  max-width: 800px;
  margin: 0 auto;
}

input {
  padding: 8px 12px;
  border: 1px solid #ccc;
  border-radius: 6px;
  width: 300px;
  transition: border-color 0.3s, box-shadow 0.3s;
  font-size: 16px;
}

input:focus {
  border-color: #2196F3;
  box-shadow: 0 0 8px rgba(33, 150, 243, 0.5);
  outline: none;
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

.search-box {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
}

.search-box input {
  flex: 1;
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 16px;
}

.search-box button {
  padding: 8px 16px;
  background: #2196F3;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.search-box button:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.commit-card {
  background: white;
  padding: 15px;
  margin: 10px 0;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.commit-hash {
  font-family: monospace;
  color: #666;
}

.commit-message {
  margin: 8px 0;
  font-size: 16px;
}

.similarity {
  color: #2196F3;
  font-size: 14px;
}

.error {
  color: #f44336;
  padding: 10px;
  margin: 10px 0;
  border-radius: 4px;
  background: #ffebee;
}

.loading {
  text-align: center;
  padding: 20px;
  color: #666;
}

.no-results {
  text-align: center;
  padding: 20px;
  color: #666;
  font-style: italic;
}
</style>