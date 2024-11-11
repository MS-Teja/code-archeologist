<template>
  <!-- Repository Input Section -->
  <div class="repo-input">
    <input v-model="repoUrl" placeholder="Enter GitHub Repository URL" />
    <button @click="analyzeRepo" :disabled="isAnalyzing">Analyze</button>
  </div>

  <!-- Step 2: Specify number of commits -->
  <div v-if="totalCommits !== null" class="commit-input">
    <p>Total Commits: {{ totalCommits }}</p>
    <input v-model.number="commitCount" type="number" :max="totalCommits" placeholder="Enter number of commits to analyze" />
    <button @click="processCommits" :disabled="isProcessing">Process Commits</button>
  </div>

  <!-- Status Messages -->
  <div v-if="isAnalyzing" class="status analyzing">
    Analyzing repository...(Might take a while for large repositories)
  </div>
  <div v-if="isProcessing" class="status processing">
    Processing commits...(Many commits might take a while)
  </div>
  <div v-if="statusMessage" class="status">
    {{ statusMessage }}
  </div>

  <div class="dashboard">
    <div v-if="isLoading" class="loading">
      Loading analysis data...
    </div>
    <div v-else-if="error" class="error">
      {{ error }}
    </div>
    <div v-else>
      <div class="top-section">
        <div class="graph-panel">
          <h2>Code Evolution Analysis</h2>
          <div id="code-evolution-graph" class="graph-container" ref="graphContainer"></div>
          <div class="legend">
            <div v-for="(color, type) in commitTypes" :key="type" class="legend-item">
              <span class="circle" :style="{ backgroundColor: color }"></span>
              {{ type }}
            </div>
          </div>
        </div>
      </div>

      <!-- Tabs for Additional Components -->
      <div class="tabs">
        <div class="tab-headers">
          <!-- AI Feature Tabs -->
          <button
            :class="{ active: activeTab === 'ai-similar', 'ai-tab': true }"
            @click="activeTab = 'ai-similar'"
          >
            Similar Commits
          </button>
          <button
            :class="{ active: activeTab === 'ai-qa', 'ai-tab': true }"
            @click="activeTab = 'ai-qa'"
          >
            Ask Questions
          </button>
          <button
            :class="{ active: activeTab === 'ai-summary', 'ai-tab': true }"
            @click="activeTab = 'ai-summary'"
          >
            Repository Summary
          </button>

          <!-- Analysis Tabs -->
          <button
            :class="{ active: activeTab === 'change-frequency' }"
            @click="activeTab = 'change-frequency'"
          >
            File Change Frequency
          </button>
          <button
            :class="{ active: activeTab === 'activity-timeline' }"
            @click="activeTab = 'activity-timeline'"
          >
            Commit Activity Timeline
          </button>
          <button
            :class="{ active: activeTab === 'contributors' }"
            @click="activeTab = 'contributors'"
          >
            Contributor Statistics
          </button>
          <button
            :class="{ active: activeTab === 'heatmap' }"
            @click="activeTab = 'heatmap'"
          >
            Codebase Heatmap
          </button>
          <button
            :class="{ active: activeTab === 'dependencies' }"
            @click="activeTab = 'dependencies'"
          >
            Dependency Graph
          </button>
          <button
            :class="{ active: activeTab === 'issues' }"
            @click="activeTab = 'issues'"
          >
            Issue Tracking
          </button>
        </div>

        <div class="tab-content">
          <div v-if="activeTab === 'ai-similar'">
            <div class="analysis-section">
              <SimilarCommits />
            </div>
          </div>
          <div v-else-if="activeTab === 'ai-qa'">
            <div class="analysis-section">
              <CommitQA />
            </div>
          </div>
          <div v-else-if="activeTab === 'ai-summary'">
            <div class="analysis-section">
              <CommitSummary />
            </div>
          </div>
          <div v-else-if="activeTab === 'change-frequency'">
            <div class="analysis-section">
              <FileChangeFrequency :analysisId="Number(analysisId)"/>
            </div>
          </div>
          <div v-else-if="activeTab === 'activity-timeline'">
            <div class="analysis-section">
              <CommitActivityTimeline :analysisId="Number(analysisId)"/>
            </div>
          </div>
          <div v-else-if="activeTab === 'contributors'">
            <div class="analysis-section">
              <ContributorStatistics :analysisId="Number(analysisId)"/>
            </div>
          </div>
          <div v-else-if="activeTab === 'heatmap'">
            <div class="analysis-section">
              <CodebaseHeatmap :analysisId="Number(analysisId)"/>
            </div>
          </div>
          <div v-else-if="activeTab === 'dependencies'">
            <div class="analysis-section">
              <DependencyGraph :analysisId="Number(analysisId)"/>
            </div>
          </div>
          <div v-else-if="activeTab === 'issues'">
            <div class="analysis-section">
              <IssueIntegration :analysisId="Number(analysisId)"/>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios';
import cytoscape from 'cytoscape';
import dagre from 'cytoscape-dagre';
import qtip from 'cytoscape-qtip';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import hljs from 'highlight.js';
import 'qtip2/dist/jquery.qtip.min.css';
import 'highlight.js/styles/default.css';
import $ from 'jquery';
import { store } from '../store/analysis.js';

// Import child components
import FileChangeFrequency from './FileChangeFrequency.vue';
import CommitActivityTimeline from './CommitActivityTimeline.vue';
import ContributorStatistics from './ContributorStatistics.vue';
import CodebaseHeatmap from './CodebaseHeatmap.vue';
import DependencyGraph from './DependencyGraph.vue';
import IssueIntegration from './IssueIntegration.vue';
import SimilarCommits from './SimilarCommits.vue';
import CommitQA from './CommitQA.vue';
import CommitSummary from './CommitSummary.vue';

window.$ = window.jQuery = $;
axios.defaults.withCredentials = true;

// Configure marked to use highlight.js for syntax highlighting
marked.setOptions({
  highlight: function (code, language) {
    return hljs.highlightAuto(code).value;
  },
});

// Initialize Cytoscape plugins
cytoscape.use(dagre);
cytoscape.use(qtip);

export default {
  components: {
    FileChangeFrequency,
    CommitActivityTimeline,
    ContributorStatistics,
    CodebaseHeatmap,
    DependencyGraph,
    IssueIntegration,
    SimilarCommits,
    CommitQA,
    CommitSummary,
  },
  data() {
    return {
      codeEvolution: [],
      commitTypes: {
        Feature: '#4caf50',
        Bugfix: '#f44336',
        Refactor: '#2196f3',
        Chore: '#9e9e9e',
        Other: '#ff9800'
      },
      isLoading: true,
      error: null,
      graph: null,
      resizeObserver: null,
      analysisStatus: null,
      analysisError: null,
      repoUrl: '', // Repository URL input
      isAnalyzing: false, // Flag to disable analyze button during analysis
      activeTab: 'ai-similar', // Default active tab
      totalCommits: null,
      commitCount: null,
      analysisId: null,
      isProcessing: false,
      statusMessage: '',
      store
    };
  },
  async created() {
    await store.init();
    if (store.analysisId) {
      try {
        await this.fetchAnalysisData();
      } catch (error) {
        console.error('Error fetching analysis data:', error);
        if (error.response?.status === 404) {
          store.setAnalysisId(null);
        }
      }
    }
  },
  mounted() {
    this.initializeDashboard();
  },
  beforeUnmount() {
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
    }
    if (this.graph) {
      this.graph.destroy();
    }
  },
  methods: {
    isValidGitHubUrl(url) {
      const regex = /^https:\/\/github\.com\/[\w.-]+\/[\w.-]+(?:\.git)?$/;
      return regex.test(url);
    },
    async fetchTotalCommits() {
      this.isLoading = true;
      this.error = null;
      try {
        const response = await axios.post('/api/get-total-commits', {
          repoUrl: this.repoUrl,
        });
        this.totalCommits = response.data.totalCommits;
      } catch (error) {
        this.error = 'Failed to fetch total commits.';
        console.error('Error fetching total commits:', error);
      } finally {
        this.isLoading = false;
      }
    },
    async analyzeRepo() {
      this.isAnalyzing = true;
      this.error = null;
      this.statusMessage = '';

      store.clear(); // Clear any stale data

      try {
        const response = await axios.post('/api/analyze', {
          repoUrl: this.repoUrl
        }, {
          withCredentials: true
        });

        if (response.data.status === 'success') {
          store.setAnalysisId(response.data.analysisId);
          this.analysisId = response.data.analysisId;
          this.totalCommits = response.data.totalCommits;
          this.statusMessage = 'Analysis initialized successfully';
          await this.fetchAnalysisData(); // Fetch initial data
        }

      } catch (error) {
        console.error('Analysis error:', error);
        this.statusMessage = 'Error: ' + (error.response?.data?.message || error.message);
        store.clear();
      } finally {
        this.isAnalyzing = false;
      }
    },

    async processCommits() {
      this.isProcessing = true;
      this.statusMessage = '';
      try {
        const response = await axios.post('/api/process-commits', {
          analysisId: this.analysisId,
          commitCount: this.commitCount,
        });
        this.statusMessage = response.data.message || 'Commits processed successfully';
        // Fetch analysis data after processing commits
        await this.fetchAnalysisData();
      } catch (error) {
        console.error('Error in processCommits:', error);
        this.statusMessage = 'Error processing commits: ' + (error.response?.data?.message || error.message);
      } finally {
        this.isProcessing = false;
      }
    },
    async initializeDashboard() {
      try {
        await this.fetchAnalysisData();
        if (this.error) {
          this.isLoading = false;
          return;
        }
        this.isLoading = false;
        await this.$nextTick();
        const graphContainer = document.getElementById('code-evolution-graph');
        if (graphContainer instanceof Element) {
          console.log('graphContainer is available. Initializing graph...');
          this.resizeObserver = new ResizeObserver(() => {
            if (this.graph) {
              this.graph.resize();
              this.graph.fit();
            }
          });
          this.resizeObserver.observe(graphContainer);
          this.renderGraph(this.prepareCytoscapeData(this.codeEvolution), graphContainer);
        } else {
          console.error('graphContainer is not a valid DOM Element');
          this.error = 'Graph container is not available.';
        }
      } catch (error) {
        console.error('Initialization error:', error);
        this.error = 'An unexpected error occurred during initialization.';
        this.isLoading = false;
      }
    },
    async fetchAnalysisData() {
      if (!store.analysisId) {
        this.error = 'No analysis available';
        return;
      }

      try {
        const response = await axios.get(`/api/analysis-data`, {
          withCredentials: true,
          params: { analysisId: store.analysisId }
        });

        if (response.data.status === 'success') {
          const { data } = response.data;

          this.codeEvolution = data.codeEvolution || [];

          // Set other data
          this.fileChanges = data.file_changes || {};
          this.commitActivity = data.commit_activity || [];
          this.contributors = data.contributors || [];
          this.dependencies = data.dependencies || {};
          this.issues = data.issues || [];

            console.log('Assigned commitActivity:', this.commitActivity);
            console.log('Assigned contributors:', this.contributors);

          console.log('Fetched analysis data:', {
            commits: this.codeEvolution.length,
            fileChanges: Object.keys(this.fileChanges).length,
            commitActivity: this.commitActivity.length,
            contributors: this.contributors.length
          });
        }

      } catch (error) {
        console.error('Data fetch error:', error);
        this.error = 'Failed to fetch analysis data';
        if (error.response?.status === 404) {
          store.clear();
        }
      }
    },

    truncateLabel(label) {
      const maxLength = 20; // Adjust the max length as needed
      return label.length > maxLength ? label.slice(0, maxLength) + '...' : label;
    },

    getCommitType(message) {
      const types = {
        feat: 'Feature',
        fix: 'Bugfix',
        refactor: 'Refactor',
        chore: 'Chore',
        docs: 'Chore',
        style: 'Refactor',
        perf: 'Feature',
        test: 'Chore',
        ci: 'Chore',
      };

      const prefixMatch = message.match(/^(\w+)(\(.+\))?:/);
      const prefix = prefixMatch ? prefixMatch[1].toLowerCase().trim() : '';

      return types[prefix] || 'Other';
    },

    formatCode(embedding) {
      if (typeof embedding === 'string') {
        const html = marked(embedding);
        return DOMPurify.sanitize(html);
      }
      return '';
    },

    prepareCytoscapeData(data) {
      if (!Array.isArray(data) || data.length === 0) {
        console.warn('No valid commit data for graph');
        return [];
      }

      const elements = [];
      const nodeMap = new Map();
      const commitMap = new Map();

      // Map commits by SHA
      data.forEach(commit => commitMap.set(commit.sha, commit));

      // Create nodes and edges
      data.forEach(commit => {
        const sha = commit.sha;
        const message = commit.message;
        const author = commit.author.name;
        const date = commit.author.date;
        const commitType = this.getCommitType(message.toLowerCase());

        // Add node if not exists
        if (!nodeMap.has(sha)) {
          nodeMap.set(sha, true);
          elements.push({
            group: 'nodes',
            data: {
              id: sha,
              label: this.truncateLabel(message),
              fullLabel: message,
              author,
              date,
              type: commitType,
            }
          });
        }

        // Add edges for parent commits
        commit.parents.forEach(parent => {
          const parentSha = parent.sha;
          elements.push({
            group: 'edges',
            data: {
              id: `${parentSha}-${sha}`,
              source: parentSha,
              target: sha,
            }
          });
        });
      });

      return elements;
    },

    renderGraph(data, container) {
      if (!container) {
        console.error('No container element provided');
        return;
      }

      if (!data || !Array.isArray(data) || data.length === 0) {
        console.warn('No valid data to render graph');
        container.innerHTML = '<div class="no-data">No commit history available</div>';
        return;
      }

      const commitTypes = { ...this.commitTypes };

      try {
        if (this.graph) {
          this.graph.destroy();
        }

        this.graph = cytoscape({
          container: container,
          elements: data,
          style: [
            {
              selector: 'node',
              style: {
                'background-color': (ele) => commitTypes[ele.data('type')],
                'label': 'data(label)',
                'font-size': 10,
                'text-valign': 'center',
                'text-halign': 'center',
                'width': 60,
                'height': 60,
                'text-wrap': 'wrap',
                'text-max-width': 50,
                'min-zoomed-font-size': 8,
              },
            },
            {
              selector: 'edge',
              style: {
                'width': 2,
                'line-color': '#ccc',
                'target-arrow-color': '#ccc',
                'target-arrow-shape': 'triangle',
                'curve-style': 'bezier',
              },
            },
          ],
          layout: {
            name: 'dagre',
            rankDir: 'LR',
            nodeSep: 50,
            edgeSep: 10,
            rankSep: 100,
          },
        });

        // Add interactivity for nodes (click event)
        this.graph.on('tap', 'node', (evt) => {
          const node = evt.target;
          alert(`Commit: ${node.data('fullLabel')}\nAuthor: ${node.data('author')}\nDate: ${node.data('date')}`);
        });

        // Add qTip tooltips to nodes on hover
        this.graph.nodes().forEach((node) => {
          const qtipContent = `<strong>${node.data('fullLabel')}</strong><br/>Author: ${node.data('author')}<br/>Date: ${node.data('date')}`;

          node.qtip({
            content: {
              text: qtipContent,
            },
            show: {
              event: 'mouseover',
            },
            hide: {
              event: 'mouseout',
            },
            position: {
              my: 'top center',
              at: 'bottom center',
            },
            style: {
              classes: 'qtip-bootstrap',
              tip: {
                width: 16,
                height: 8,
              },
            },
          });
        });

        // Add viewport padding
        this.graph.fit(undefined, 50);
        this.graph.center();
      } catch (error) {
        console.error('Graph rendering error:', error);
        container.innerHTML = '<div class="error">Failed to render commit graph</div>';
      }
    }
  },
};
</script>

<style scoped>
.dashboard {
  gap: 20px;
  padding: 20px;
  background-color: #ffffff;
}

.repo-input, .commit-input {
  display: flex;
  gap: 12px;
  margin-bottom: 25px;
  align-items: center;
  justify-content: center;
}

.repo-input {
  padding: 20px;
  border-radius: 10px;
}

.commit-input {
  flex-direction: column;
  padding: 20px;
  border-radius: 10px;
}

.repo-input input, .commit-input input {
  padding: 12px 16px;
  border: 1px solid #ccc;
  border-radius: 6px;
  width: 300px;
  transition: border-color 0.3s, box-shadow 0.3s;
  font-size: 16px;
}

.repo-input input:focus, .commit-input input:focus {
  border-color: #2196F3;
  box-shadow: 0 0 8px rgba(33, 150, 243, 0.5);
  outline: none;
}

.repo-input button, .commit-input button {
  padding: 12px 24px;
  background-color: #2196F3;
  color: #fff;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 16px;
  transition: background-color 0.3s, transform 0.2s, box-shadow 0.3s;
}

.repo-input button:hover, .commit-input button:hover {
  background-color: #1976D2;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

.repo-input button:active, .commit-input button:active {
  transform: scale(0.98);
}

.repo-input button:disabled, .commit-input button:disabled {
  background-color: #90CAF9;
  cursor: not-allowed;
  opacity: 0.7;
}

.commit-input input {
  margin-right: 10px;
}

.commit-input p {
  font-weight: bold;
  margin-bottom: 10px;
}

.status {
  text-align: center;
  padding: 10px;
  font-weight: bold;
  margin-top: 10px;
}

.status.analyzing {
  color: #2196F3; /* Blue */
}

.status.processing {
  color: #4caf50; /* Green */
}

.status.error {
  color: #f44336; /* Red */
}

.status.success {
  color: #4caf50; /* Green */
}

.top-section {
  display: flex;
  flex-direction: column;
  margin-bottom: 30px;
}

.graph-panel,
.suggestions-panel {
  background: #fff;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.graph-container {
  width: 100%;
  height: 400px;
  border: 1px solid #eee;
  border-radius: 8px;
  margin: 20px 0;
  overflow: hidden; /* Prevent overflow */
  position: relative; /* For proper sizing */
}

.suggestions-container {
  max-height: 600px;
  overflow-y: auto;
}

.suggestion-card {
  background: #f8f9fa;
  border: 3px solid #323232;
  border-radius: 6px;
  padding: 15px;
  margin-bottom: 15px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.code-section {
  margin: 10px 0;
}

.controls select {
  padding: 8px;
  border-radius: 4px;
  border: 1px solid #ddd;
  margin-bottom: 15px;
}

/* Tabs Styling */
.tabs {
  border-top: 1px solid #ddd;
}

.tab-headers {
  display: flex;
  border-bottom: 1px solid #ddd;
  margin-bottom: 10px;
}

.tab-headers button {
  background: none;
  border: none;
  padding: 10px 20px;
  cursor: pointer;
  outline: none;
  transition: background-color 0.3s;
}

.tab-headers button:hover {
  background-color: #f1f1f1;
}

.tab-headers button.active {
  border-bottom: 2px solid #2196F3;
  font-weight: bold;
}

.tab-content {
  padding: 10px 0;
}

.legend {
  display: flex;
  gap: 15px;
  margin-top: 15px;
  flex-wrap: wrap;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.circle {
  width: 12px;
  height: 12px;
  border-radius: 50%;
}

.code-block pre {
  background: #1e1e1e;
  color: #d4d4d4;
  padding: 15px;
  border-radius: 4px;
  overflow-x: auto;
  font-family: 'Consolas', monospace;
}

.code-block code {
  color: inherit;
  background: none;
  font-family: 'Consolas', monospace;
}

/* Styles for syntax highlighting */
.hljs-keyword {
  color: #569cd6;
}
.hljs-comment {
  color: #6a9955;
}
/* Add more styles as needed */

.analysis-section {
  margin: 10px 0;
  line-height: 1.5;
}

.legend-item.Feature .circle {
  background-color: #4caf50;
}
.legend-item.Bugfix .circle {
  background-color: #f44336;
}
.legend-item.Refactor .circle {
  background-color: #2196f3;
}
.legend-item.Chore .circle {
  background-color: #9e9e9e;
}
.legend-item.Other .circle {
  background-color: #ff9800;
}

.loading {
  font-size: 1.2em;
  text-align: center;
  margin-top: 50px;
  background-color: #fff;
}

.error {
  color: red;
  font-size: 1.2em;
  text-align: center;
  margin-top: 50px;
}

.no-data, .error {
  text-align: center;
  padding: 20px;
  color: #666;
  font-style: italic;
}

.error {
  color: #f44336;
}

.legend-item .circle {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  display: inline-block;
  margin-right: 8px;
  border: 2px solid #fff;
}

.qtip-bootstrap {
  z-index: 10000 !important;
}

/* Tab styling */
.tabs {
  width: 100%;
  margin: 20px 0;
  background: #ffffff;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
}

.tab-headers {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 16px;
  background: #f8f9fa;
  border-radius: 12px 12px 0 0;
  border-bottom: 2px solid #e9ecef;
}

.tab-headers button {
  padding: 12px 24px;
  background: transparent;
  border: none;
  border-radius: 8px;
  color: #495057;
  font-weight: 500;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
  white-space: nowrap;
}

.tab-headers button:hover {
  background: rgba(33, 150, 243, 0.1);
  color: #2196F3;
}

.tab-headers button.active {
  background: #2196F3;
  color: white;
  box-shadow: 0 2px 8px rgba(2, 2, 2, 0.4);
}

.tab-headers button:before {
  content: '';
  position: absolute;
  bottom: -2px;
  left: 0;
  width: 100%;
  height: 2px;
  background: #2196F3;
  transform: scaleX(0);
  transition: transform 0.3s ease;
}

.tab-headers button.active:before {
  transform: scaleX(1);
}

.ai-tab {
  background: linear-gradient(45deg, #2196F3, #21CBF3) !important;
  color: white !important;
  border: none;
  position: relative;
  overflow: hidden;
}

.ai-tab::before {
  content: '🤖';
  position: absolute;
  top: 4px;
  right: 4px;
  font-size: 10px;
}

.ai-tab:hover {
  background: linear-gradient(45deg, #1976D2, #00BCD4) !important;
  transform: translateY(-1px);
}

.ai-tab.active {
  background: linear-gradient(45deg, #1565C0, #00ACC1) !important;
  box-shadow: 0 4px 12px rgba(33, 150, 243, 0.3);
}

@media (max-width: 768px) {
  .tab-headers {
    flex-direction: column;
  }

  .ai-tab::before {
    right: 8px;
  }
}

/* Responsive design */
@media (max-width: 768px) {
  .tab-headers {
    flex-direction: column;
    gap: 4px;
  }

  .tab-headers button {
    width: 100%;
    text-align: left;
  }
}

/* Tab content transition */
.tab-content {
  padding: 24px;
  background: #ffffff;
  border-radius: 0 0 12px 12px;
  min-height: 200px;
}

.similar-commits {
  margin-top: 20px;
}
</style>