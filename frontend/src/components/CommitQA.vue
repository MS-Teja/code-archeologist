<template>
  <div class="commit-qa">
    <div class="container">
      <h2>Ask About Commits</h2>
      <div class="question-box">
        <input v-model="question" placeholder="Ask a question about the commits..." />
        <button @click="askQuestion">Ask</button>
      </div>
    </div>
    <div v-if="loading" class="loading">Thinking...</div>
    <div v-else-if="error" class="error">{{ error }}</div>
    <div v-else-if="answer" class="answer" v-html="formattedAnswer"></div>
  </div>
</template>

<script>
import { marked } from 'marked';
import { store } from '../store/analysis';
import axios from 'axios';

export default {
  name: 'CommitQA',
  data() {
    return {
        question: '',
        answer: null,
        loading: false,
        error: null,
        store
    }
  },
  computed: {
    formattedAnswer() {
      return this.answer ? marked(this.answer) : '';
    }
  },
  methods: {
    async askQuestion() {
        this.loading = true;
        try {
            const response = await axios.post('/api/question-answering', {
            question: this.question
            }, {
            withCredentials: true,
            headers: { 'X-Analysis-ID': store.analysisId }
            });
            this.answer = response.data.answer;
        } catch (error) {
            this.error = error.response?.data?.message || 'Failed to get answer';
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
</style>