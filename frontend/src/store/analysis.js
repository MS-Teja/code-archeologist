import { reactive } from 'vue';
import axios from 'axios';

export const store = reactive({
  analysisId: localStorage.getItem('analysisId') || null,
  sessionId: null,
  isAnalyzing: false,

  async init() {
    try {
      const response = await axios.get('/api/session');
      this.analysisId = response.data.analysisId;
    } catch (error) {
      console.error('Initialization error:', error);
    }
  },

  setAnalysisId(id) {
    if (!id) return;
    this.analysisId = Number(id);
    localStorage.setItem('analysisId', id);
    console.log('Analysis ID set:', id);
  },

  setSessionId(id) {
    if (!id) return;
    this.sessionId = id;
    localStorage.setItem('sessionId', id);
    console.log('Session ID set:', id);
  },

  clear() {
    localStorage.removeItem('analysisId');
    localStorage.removeItem('sessionId');
    this.analysisId = null;
    this.sessionId = null;
  }
});