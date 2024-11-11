# Code Archeologist

**Code Archeologist** analyzes your Git repository history to identify patterns in code evolution. It creates a "genetic tree" of your code's ancestry, provides refactoring suggestions, generates a codebase heatmap, commit activity timeline, contributor statistics, dependency graph, file change frequency, and integrates with issue tracking.

## Features

- **Genetic Tree:** Visualize the ancestry of your codebase.
- **Refactoring Suggestions:** Receive actionable recommendations to improve your code.
- **Codebase Heatmap:** Identify hotspots and areas with high activity.
- **Commit Activity Timeline:** Track commit patterns over time.
- **Contributor Statistics:** Analyze contributions from different team members.
- **Dependency Graph:** Visualize project dependencies.
- **File Change Frequency:** Monitor how often files are modified.
- **Issue Integration:** Link code changes with issue tracking systems.
- **Semantic Search in Commits:** Find similar commits based on semantic meaning using vector embeddings.
- **Question Answering:** Ask questions about your codebase and receive AI-generated answers.
- **Summarization:** Get concise summaries of commit messages.

## Features with AI Integration

- **Vector Embeddings:** Store and analyze commit messages and file changes using vector embeddings.
- **Semantic Similarity Search:** Find similar commits based on semantic meaning using vector similarity.
- **AI-Driven Insights:** Enhance analysis with AI-generated recommendations and patterns.
- **Question Answering:** Interact with your codebase using natural language queries.
- **Summarization:** Generate summaries of large sets of commit messages.

## Technologies Used

- **Frontend:**
  - Vue.js
  - Axios
  - Cytoscape.js
  - Chart.js
  - Highlight.js
  - QTip2

- **Backend:**
  - Node.js
  - Express.js
  - PostgreSQL
  - pgvector
  - pgvectorscale
  - pgai
  - Octokit
  - OpenAI SDK
  - dotenv
  - express-session

## Installation

### Prerequisites

- **Node.js** (v14 or later)
- **npm** or **yarn**
- **PostgreSQL** database with extensions: `pgvector`, `pgvectorscale`, `pgai`
- **ollama** and it should have **nomic-embed-text** model

### Backend Setup

1. **Clone the Repository**

    ```bash
    git clone https://github.com/MS-Teja/code-archeologist.git
    cd code-archeologist/backend
    ```

2. **Install Dependencies**
    ```bash
    npm install
    ```

3. **Configure Environment Variables**
    Create a .env file in the backend directory and add the following:
    ```
        GITHUB_TOKEN=your_github_token
        OPENAI_API_KEY=your_openai_api_key
        SESSION_SECRET=your_session_secret
        DB_USER=your_db_user
        DB_PASSWORD=your_db_password
        DB_HOST=your_db_host
        DB_PORT=your_db_port
        DB_NAME=your_db_name
        PORT=3000
    ```

4. **Enable PostgreSQL Extensions**
    Connect to your PostgreSQL database and enable the necessary extensions:
    ```
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS vectorscale;
    ```

5. **Run Database Migrations**
    5. **Run Database Migrations**
        Apply any necessary migrations to include vector columns as shown in [Step 2](#install-dependencies).

6. **Start the Backend Server**
    ```
    npm start
    ```

### Frontend Setup
1. **Navigate to Frontend Directory**
    ```
    npm start
    ```

2. **Install Dependencies**
    ```
    npm install
    ```

3. **Start the Frontend Development Server**
    ```
    npm run dev
    ```

## Usage

1. **Access the Application**
    - Open your browser and navigate to `http://localhost:3000` (or the port specified in your `.env` file).

2. **Analyze a Repository**
    - Enter the GitHub repository URL you wish to analyze.
    - Specify the number of commits to include in the analysis.
    - Click on the Analyze button to start the process.
    - Once analysis is complete, explore the various visualizations and insights provided.

3. **Find Similar Commits**
    - Navigate to the Similar Commits section in the dashboard.
    - Enter a commit message or select an existing one to find semantically similar commits based on vector embeddings.

4. **Ask Questions**
    - Use the Question Answering feature to ask questions about your codebase.
    - Receive AI-generated answers based on your commit history.

5. **View Summaries**
    - Access the Summarization section to get concise summaries of your commit messages.

## Contributing

We welcome contributions from everyone! Whether you're fixing bugs, adding new features, or improving documentation, your help is appreciated.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.