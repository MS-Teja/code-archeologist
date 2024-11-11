import express from 'express';
import dotenv from 'dotenv';
import { promisify } from 'util';
import { exec } from 'child_process';
import { Octokit } from '@octokit/rest';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import session from 'express-session';
import pg from 'pg';
import fs from 'fs/promises';
import OpenAI from 'openai';
import winston from 'winston';
import cors from 'cors';

// Initialize environment variables
dotenv.config();

const execAsync = promisify(exec);

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
app.use(express.json());

// CORS configuration
app.use(cors({
  origin: 'http://localhost:5173', // Replace with your frontend URL
  credentials: true, // Allow credentials (cookies)
}));

// Set up session middleware
app.use(
  session({
    secret: process.env.SESSION_SECRET, // Use environment variable
    resave: false,
    saveUninitialized: true,
    cookie: { secure: false }, // Set to true in production with HTTPS
  })
);

// Database configuration
const pool = new pg.Pool({
  connectionString: `postgres://${process.env.DB_USER}:${process.env.DB_PASSWORD}@${process.env.DB_HOST}:${process.env.DB_PORT}/${process.env.DB_NAME}?sslmode=require`,
  ssl: {
    rejectUnauthorized: false,
  },
});

// Test and initialize database
pool
  .connect()
  .then(() => {
    console.log('Successfully connected to database');
    return initializeDatabase();
  })
  .catch((err) => {
    console.error('Database connection error:', err);
    process.exit(1); // Exit if we can't connect to the database
  });

// Initialize database schema
async function initializeDatabase() {
  try {
    await pool.query(`
      -- Ensure the pgvector and pgvectorscale extensions are installed
      CREATE EXTENSION IF NOT EXISTS vector;
      CREATE EXTENSION IF NOT EXISTS vectorscale;

      CREATE TABLE IF NOT EXISTS code_analysis (
        id BIGSERIAL PRIMARY KEY,
        session_id TEXT NOT NULL,
        repo_url TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        commits JSONB DEFAULT '{}',
        file_changes JSONB DEFAULT '{}',
        contributors JSONB DEFAULT '{}',
        commit_activity JSONB DEFAULT '{}',
        dependencies JSONB DEFAULT '{}',
        issues JSONB DEFAULT '{}',
        UNIQUE(session_id, repo_url)
      );

      -- Create the commit_embeddings table with the correct vector dimensions
      CREATE TABLE IF NOT EXISTS commit_embeddings (
        id BIGSERIAL PRIMARY KEY,
        code_analysis_id BIGINT REFERENCES code_analysis(id),
        commit_hash TEXT NOT NULL,
        commit_message TEXT NOT NULL,
        embedding VECTOR(768), -- Adjusted to match LLM embedding size
        created_at TIMESTAMPTZ DEFAULT NOW(),
        UNIQUE(code_analysis_id, commit_hash)
      );

      -- Create vector similarity search index using diskann
      DO $$
      BEGIN
        IF NOT EXISTS (
          SELECT 1 FROM pg_indexes WHERE indexname = 'commit_embeddings_idx'
        ) THEN
          CREATE INDEX commit_embeddings_idx
          ON commit_embeddings USING diskann (embedding);
        END IF;
      END$$;
    `);

    console.log('Database schema initialized successfully');
  } catch (error) {
    console.error('Error initializing database:', error);
    throw error;
  }
}

// Initialize Octokit for GitHub API access
const octokit = new Octokit({
  auth: process.env.GITHUB_TOKEN,
});

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Helper function to extract owner and repo from URL
function extractRepoInfo(repoUrl) {
  const match = repoUrl.match(/github\.com\/([\w-]+)\/([\w-]+)/);
  if (match) {
    const owner = match[1];
    const repo = match[2];
    return { owner, repo };
  } else {
    throw new Error('Invalid GitHub repository URL');
  }
}

// Fetch total commit count
async function getTotalCommitCount(owner, repo) {
  let totalCommits = 0;
  let page = 1;
  const perPage = 100;

  try {
    while (true) {
      const response = await octokit.repos.listCommits({
        owner,
        repo,
        per_page: perPage,
        page,
      });

      totalCommits += response.data.length;

      if (response.data.length < perPage) {
        break;
      }
      page++;
    }
    return totalCommits;
  } catch (error) {
    console.error('Error fetching total commit count:', error.message);
    throw error;
  }
}

// Fetch contributors
async function fetchContributors(owner, repo) {
  let contributors = [];
  let page = 1;
  const perPage = 100;

  try {
    while (true) {
      const response = await octokit.repos.listContributors({
        owner,
        repo,
        per_page: perPage,
        page,
      });

      contributors = contributors.concat(response.data);

      if (response.data.length < perPage) {
        break;
      }
      page++;
    }
    return contributors;
  } catch (error) {
    console.error('Error fetching contributors:', error.message);
    throw error;
  }
}

// Fetch commit activity with retry mechanism
async function fetchCommitActivity(owner, repo, retries = 3) {
  try {
    const response = await octokit.repos.getCommitActivityStats({
      owner,
      repo,
    });

    // If stats are being generated (202 status)
    if (response.status === 202 && retries > 0) {
      console.log('Commit stats being generated, retrying in 1 second...');
      await new Promise((resolve) => setTimeout(resolve, 1000));
      return fetchCommitActivity(owner, repo, retries - 1);
    }

    return response.data; // Array of weekly commit activity
  } catch (error) {
    console.error('Error fetching commit activity:', error);
    return [];
  }
}

// Fetch file changes
async function fetchFileChanges(owner, repo) {
  let page = 1;
  const perPage = 100;
  const fileChangeCounts = {};

  try {
    while (true) {
      const response = await octokit.repos.listCommits({
        owner,
        repo,
        per_page: perPage,
        page,
      });

      if (response.data.length === 0) {
        break;
      }

      for (const commit of response.data) {
        const commitDetails = await octokit.repos.getCommit({
          owner,
          repo,
          ref: commit.sha,
        });

        for (const file of commitDetails.data.files) {
          fileChangeCounts[file.filename] = (fileChangeCounts[file.filename] || 0) + 1;
        }
      }

      if (response.data.length < perPage) {
        break;
      }
      page++;
    }

    return fileChangeCounts;
  } catch (error) {
    console.error('Error fetching file changes:', error);
    return {};
  }
}

// Fetch issues
async function fetchIssues(owner, repo) {
  let page = 1;
  const perPage = 100;
  const issues = [];

  try {
    while (true) {
      const response = await octokit.issues.listForRepo({
        owner,
        repo,
        state: 'all',
        per_page: perPage,
        page,
      });

      if (response.data.length === 0) {
        break;
      }

      issues.push(...response.data);

      if (response.data.length < perPage) {
        break;
      }
      page++;
    }

    // Process issues to include only necessary fields
    const processedIssues = issues.map((issue) => ({
      id: issue.id,
      number: issue.number,
      title: issue.title,
      state: issue.state,
      created_at: issue.created_at,
      updated_at: issue.updated_at,
      closed_at: issue.closed_at,
      url: issue.html_url,
    }));

    return processedIssues;
  } catch (error) {
    console.error('Error fetching issues:', error);
    return [];
  }
}

// Fetch dependencies
async function fetchDependencies(owner, repo) {
  try {
    const response = await octokit.repos.getContent({
      owner,
      repo,
      path: 'package.json',
    });

    const content = Buffer.from(response.data.content, 'base64').toString('utf-8');
    const packageJson = JSON.parse(content);

    // Combine dependencies and devDependencies
    const dependencies = {
      ...packageJson.dependencies,
      ...packageJson.devDependencies,
    };

    return dependencies || {};
  } catch (error) {
    if (error.status === 404) {
      console.warn(`package.json not found in repository: ${owner}/${repo}. Dependencies will be empty.`);
      return {};
    }
    console.error('Error fetching dependencies:', error);
    return {};
  }
}

// Fetch commits from the repository
async function fetchCommits(owner, repo) {
  let page = 1;
  const perPage = 100;
  const commits = [];

  try {
    while (true) {
      const response = await octokit.repos.listCommits({
        owner,
        repo,
        per_page: perPage,
        page,
      });

      if (response.data.length === 0) {
        break;
      }

      commits.push(...response.data);

      if (response.data.length < perPage) {
        break;
      }
      page++;
    }

    return commits;
  } catch (error) {
    console.error('Error fetching commits:', error);
    return [];
  }
}

// Function to generate embeddings using OpenAI API
async function generateEmbedding(text) {
  try {
    const response = await openai.embeddings.create({
      model: 'text-embedding-ada-002',
      input: text,
    });
    return response.data[0].embedding;
  } catch (error) {
    console.error('Error generating embedding:', error.response?.data || error.message);
    return null;
  }
}

// Update generateEmbeddingWithOllama function
async function generateEmbeddingWithOllama(text) {
  try {
    console.log('Sending request to Ollama API...');

    const response = await fetch('http://localhost:11434/api/embeddings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: process.env.OLLAMA_MODEL || 'nomic-embed-text',
        prompt: text,
        options: { temperature: 0 }
      })
    });

    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    console.log('Raw response from Ollama API:', JSON.stringify(data, null, 2));

    // Extract embedding from response
    let embedding;
    if (data.embeddings && Array.isArray(data.embeddings)) {
      embedding = data.embeddings;
    } else if (data.embedding && Array.isArray(data.embedding)) {
      embedding = data.embedding;
    } else {
      throw new Error('Invalid embedding format in response');
    }

    if (!embedding.length) {
      throw new Error('Empty embedding array received');
    }

    console.log(`Successfully generated embedding with ${embedding.length} dimensions`);
    return embedding;

  } catch (error) {
    console.error('Error in generateEmbeddingWithOllama:', error.message);
    throw error;
  }
}

// Update the existing generateEmbeddingWithRetry function
async function generateEmbeddingWithRetry(text, retries = 3, delay = 1000) {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const embedding = await generateEmbeddingWithOllama(text);
      if (embedding) return embedding;
    } catch (error) {
      if (attempt < retries) {
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
  }
  return null;
}

// Function to generate completions using OpenAI API
async function generateCompletion(prompt) {
  try {
    const response = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages: [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: prompt },
      ]
    });
    const completion = response.choices[0].message.content.trim();
    return completion;
  } catch (error) {
    console.error('Error generating completion:', error.response?.data || error.message);
    return 'Sorry, I could not generate a response.';
  }
}

// Update insertCommitEmbeddings function
async function insertCommitEmbeddings(analysisId, commits) {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    const insertQuery = `
      INSERT INTO commit_embeddings (commit_hash, commit_message, embedding, code_analysis_id)
      VALUES ($1, $2, $3::vector, $4)
      ON CONFLICT (code_analysis_id, commit_hash) DO NOTHING;
    `;

    for (const commit of commits) {
      const embedding = await generateEmbeddingWithRetry(commit.commit.message);
      if (embedding) {
        const formattedEmbedding = `[${embedding.join(',')}]`;
        await client.query(insertQuery, [
          commit.sha,
          commit.commit.message,
          formattedEmbedding,
          analysisId
        ]);
      }
    }

    await client.query('COMMIT');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Error inserting commit embeddings:', error);
    throw error;
  } finally {
    client.release();
  }
}

// Middleware to attach repo information based on sessionId
async function attachRepoInfo(req, res, next) {
  const sessionId = req.sessionID;

  if (!sessionId) {
    return res.status(400).json({ error: 'Missing session ID.' });
  }

  try {
    const { rows } = await pool.query(
      'SELECT repo_url FROM code_analysis WHERE session_id = $1 AND status = $2 ORDER BY created_at DESC LIMIT 1',
      [sessionId, 'complete']
    );

    if (rows.length === 0) {
      return res.status(404).json({ error: 'No analysis found for the provided session ID.' });
    }

    const repoUrl = rows[0].repo_url;
    const { owner, repo } = extractRepoInfo(repoUrl);

    // Attach to request object
    req.owner = owner;
    req.repo = repo;

    next();
  } catch (error) {
    console.error('Error attaching repo info:', error);
    res.status(500).json({ error: 'Failed to retrieve repository information.' });
  }
}

// Add this middleware for session validation
const validateSession = async (req, res, next) => {
  const sessionId = req.sessionID || req.cookies['connect.sid'];
  if (!sessionId) {
    return res.status(401).json({ error: 'No session ID provided' });
  }

  try {
    const { rows } = await pool.query(
      'SELECT id FROM code_analysis WHERE session_id = $1 ORDER BY created_at DESC LIMIT 1',
      [sessionId]
    );

    if (rows.length === 0) {
      return res.status(404).json({ error: 'No analysis found for session' });
    }

    req.analysisId = rows[0].id;
    next();
  } catch (error) {
    console.error('Session validation error:', error);
    res.status(500).json({ error: 'Session validation failed' });
  }
};

// Main analysis endpoint
app.get('/api/analysis-data', async (req, res) => {
  const { analysisId } = req.query;
  
  try {
    const { rows } = await pool.query(
      'SELECT * FROM code_analysis WHERE id = $1',
      [analysisId]
    );

    if (rows.length === 0) {
      return res.status(404).json({
        status: 'error',
        message: 'Analysis not found'
      });
    }

    // Format data consistently
    const analysis = rows[0];
    
    // Get all commit data
    const commitData = analysis.commits?.commits || [];

    res.json({
      status: 'success',
      data: {
        id: analysis.id,
        repo_url: analysis.repo_url,
        status: analysis.status,
        created_at: analysis.created_at,
        codeEvolution: commitData, // Return full commit objects
        file_changes: analysis.file_changes || {},
        commit_activity: analysis.commit_activity || [],
        contributors: analysis.contributors || [],
        dependencies: analysis.dependencies || {},
        issues: analysis.issues || []
      }
    });

  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({
      status: 'error', 
      message: error.message
    });
  }
});

app.post('/api/process-commits', async (req, res) => {
  const { analysisId, commitCount } = req.body;

  try {
    // Get repository info
    const { rows } = await pool.query(
      'SELECT repo_url FROM code_analysis WHERE id = $1',
      [analysisId]
    );

    if (rows.length === 0) {
      return res.status(404).json({
        status: 'error',
        message: 'Analysis not found'
      });
    }

    const { owner, repo } = extractRepoInfo(rows[0].repo_url);
    
    // Fetch commits
    const commits = await fetchCommits(owner, repo, commitCount);
    
    // Store full commit data in code_analysis
    const processedCommits = commits.map(commit => ({
      sha: commit.sha,
      message: commit.commit.message,
      author: {
        name: commit.commit.author.name,
        email: commit.commit.author.email,
        date: commit.commit.author.date
      },
      parents: commit.parents.map(p => ({ sha: p.sha }))
    }));

    // Update code_analysis with full commit data
    await pool.query(
      `UPDATE code_analysis 
       SET commits = $1,
           status = 'completed'
       WHERE id = $2`,
      [JSON.stringify({
        totalCommits: commitCount,
        commits: processedCommits
      }), analysisId]
    );

    // Process embeddings in parallel
    await Promise.all(commits.map(async (commit) => {
      try {
        const embedding = await generateEmbeddingWithRetry(commit.commit.message);
        if (embedding) {
          await pool.query(
            `INSERT INTO commit_embeddings 
             (code_analysis_id, commit_hash, commit_message, embedding)
             VALUES ($1, $2, $3, $4)
             ON CONFLICT (code_analysis_id, commit_hash) 
             DO UPDATE SET embedding = EXCLUDED.embedding`,
            [analysisId, commit.sha, commit.commit.message, `[${embedding.join(',')}]`]
          );
        }
      } catch (error) {
        console.error(`Error processing embedding for commit ${commit.sha}:`, error);
      }
    }));

    res.json({
      status: 'success',
      message: 'Commits processed successfully',
      processedCommits: processedCommits.length
    });

  } catch (error) {
    console.error('Error processing commits:', error);
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
});

// Main analysis endpoint
app.post('/api/analyze', async (req, res) => {
  const { repoUrl } = req.body;
  const sessionId = req.sessionID;

  console.log(`Received /api/analyze request with repoUrl: ${repoUrl} and sessionId: ${sessionId}`);

  if (!repoUrl) {
    return res.status(400).json({ error: 'repoUrl is required.' });
  }

  try {
    const { owner, repo } = extractRepoInfo(repoUrl);

    // Fetch repository data
    const totalCommits = await getTotalCommitCount(owner, repo);
    const contributors = await fetchContributors(owner, repo);
    const commitActivity = await fetchCommitActivity(owner, repo);
    const fileChanges = await fetchFileChanges(owner, repo);
    const issues = await fetchIssues(owner, repo);
    const dependencies = await fetchDependencies(owner, repo);

    // Create or update analysis record
    const analysis = await pool.query(
      `
      INSERT INTO code_analysis
        (session_id, repo_url, status, commits, contributors, commit_activity, file_changes, dependencies, issues)
      VALUES
        ($1, $2, $3, $4, $5, $6, $7, $8, $9)
      ON CONFLICT (session_id, repo_url)
      DO UPDATE SET
        status = 'completed',
        commits = $4,
        contributors = $5,
        commit_activity = $6,
        file_changes = $7,
        dependencies = $8,
        issues = $9
      RETURNING id
      `,
      [
        sessionId,
        repoUrl,
        'completed',
        JSON.stringify({ totalCommits }),
        JSON.stringify(contributors),
        JSON.stringify(commitActivity),
        JSON.stringify(fileChanges),
        JSON.stringify(dependencies),
        JSON.stringify(issues),
      ]
    );

    const analysisId = analysis.rows[0].id;
    console.log(`Generated analysisId: ${analysisId} for sessionId: ${sessionId}`);

    // Store analysisId in session
    req.session.analysisId = analysisId;

    // Explicitly save the session before sending the response
    req.session.save(err => {
      if (err) {
        console.error('Session save error:', err);
        return res.status(500).json({
          status: 'error',
          message: 'Failed to save session.',
        });
      }

      // Send response after session is saved
      res.json({
        status: 'success',
        message: 'Analysis completed successfully',
        analysisId: analysisId,
        totalCommits,
      });
    });
  } catch (error) {
    console.error('Error initializing analysis:', error);
    res.status(500).json({
      status: 'error',
      message: 'Failed to initialize analysis.',
    });
  }
});

// Endpoint to retrieve analysis by ID
app.get('/api/analysis/:analysisId', async (req, res) => {
  const { analysisId } = req.params;

  try {
    const { rows } = await pool.query('SELECT * FROM code_analysis WHERE id = $1', [analysisId]);

    if (rows.length === 0) {
      return res.status(404).json({ status: 'error', message: 'Analysis not found' });
    }

    res.json(rows[0]);
  } catch (error) {
    console.error('Error retrieving analysis:', error);
    res.status(500).json({ status: 'error', message: error.message });
  }
});

// Test session endpoint
app.get('/api/session', async (req, res) => {
  const sessionId = req.sessionID || req.cookies['connect.sid'];
  const analysisId = req.session.analysisId || req.headers['x-analysis-id'];

  try {
    if (analysisId) {
      // Verify analysis exists
      const { rows } = await pool.query(
        'SELECT id FROM code_analysis WHERE id = $1',
        [analysisId]
      );
      
      if (rows.length > 0) {
        return res.json({
          sessionId,
          analysisId: rows[0].id
        });
      }
    }

    res.json({ 
      sessionId,
      analysisId: null 
    });
  } catch (error) {
    console.error('Session error:', error);
    res.status(500).json({ error: 'Session error' });
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'healthy' });
});

// File Change Frequency Endpoint
app.get('/api/file-change-frequency', validateSession, async (req, res) => {
  try {
    const { rows } = await pool.query(
      'SELECT file_changes FROM code_analysis WHERE id = $1',
      [req.analysisId]
    );

    if (!rows[0] || !rows[0].file_changes) {
      return res.status(404).json({ error: 'No file change data found' });
    }

    const fileChanges = rows[0].file_changes;
    res.json({ status: 'success', data: fileChanges });
  } catch (error) {
    console.error('Error fetching file changes:', error);
    res.status(500).json({ error: 'Failed to fetch file changes' });
  }
});

// Commit Activity Timeline Endpoint
app.get('/api/commit-activity-timeline', validateSession, async (req, res) => {
  try {
    const { rows } = await pool.query(
      'SELECT commit_activity FROM code_analysis WHERE id = $1',
      [req.analysisId]
    );
    if (!rows[0]?.commit_activity) {
      return res.status(404).json({ error: 'No commit activity data found' });
    }
    res.json({
      status: 'success',
      data: rows[0].commit_activity
    });
  } catch (error) {
    console.error('Error fetching commit activity:', error);
    res.status(500).json({ error: 'Failed to fetch commit activity' });
  }
});

// Contributor Statistics
app.get('/api/contributor-statistics', validateSession, async (req, res) => {
  try {
    const { rows } = await pool.query(
      'SELECT contributors FROM code_analysis WHERE id = $1',
      [req.analysisId]
    );
    if (!rows[0]?.contributors) {
      return res.status(404).json({ error: 'No contributor data found' });
    }
    res.json({ status: 'success', data: rows[0].contributors });
  } catch (error) {
    console.error('Error fetching contributors:', error);
    res.status(500).json({ error: 'Failed to fetch contributors' });
  }
});

// Codebase Heatmap
app.get('/api/codebase-heatmap', validateSession, async (req, res) => {
  try {
    const { rows } = await pool.query(
      'SELECT file_changes FROM code_analysis WHERE id = $1',
      [req.analysisId]
    );
    if (!rows[0]?.file_changes) {
      return res.status(404).json({ error: 'No file change data found' });
    }
    res.json({ status: 'success', data: rows[0].file_changes });
  } catch (error) {
    console.error('Error generating heatmap:', error);
    res.status(500).json({ error: 'Failed to generate heatmap' });
  }
});

// Dependency Graph
app.get('/api/dependency-graph', validateSession, async (req, res) => {
  try {
    const { rows } = await pool.query(
      'SELECT dependencies FROM code_analysis WHERE id = $1',
      [req.analysisId]
    );
    if (!rows[0]?.dependencies) {
      return res.status(404).json({ error: 'No dependency data found' });
    }
    res.json({ status: 'success', data: rows[0].dependencies });
  } catch (error) {
    console.error('Error fetching dependencies:', error);
    res.status(500).json({ error: 'Failed to fetch dependencies' });
  }
});

// Linked Issues
app.get('/api/linked-issues', validateSession, async (req, res) => {
  try {
    const { rows } = await pool.query(
      'SELECT issues FROM code_analysis WHERE id = $1',
      [req.analysisId]
    );
    if (!rows[0]?.issues) {
      return res.status(404).json({ error: 'No issues data found' });
    }
    res.json({ status: 'success', data: rows[0].issues });
  } catch (error) {
    console.error('Error fetching issues:', error);
    res.status(500).json({ error: 'Failed to fetch issues' });
  }
});

// Semantic Search in Commits
app.get('/api/search-commits', validateSession, async (req, res) => {
  const { query } = req.query;
  if (!query?.trim()) {
    return res.status(400).json({
      status: 'error',
      message: 'Search query is required'
    });
  }

  try {
    // Generate embedding for search query
    const queryEmbedding = await generateEmbeddingWithRetry(query);
    if (!queryEmbedding) {
      return res.status(500).json({
        status: 'error',
        message: 'Failed to generate embedding for query'
      });
    }

    console.log('Generated embedding dimensions:', queryEmbedding.length);

    // Modified query with explicit casting and better similarity threshold
    const searchResults = await pool.query(
      `SELECT
        commit_hash,
        commit_message,
        1 - (embedding <=> $1::vector) as similarity
       FROM commit_embeddings
       WHERE code_analysis_id = $2
       AND 1 - (embedding <=> $1::vector) > 0.5
       ORDER BY similarity DESC
       LIMIT 5`,
      [
        `[${queryEmbedding.join(',')}]`,
        req.analysisId
      ]
    );

    console.log(`Found ${searchResults.rows.length} results`);

    if (searchResults.rows.length === 0) {
      return res.json({
        status: 'success',
        message: 'No matching commits found',
        results: []
      });
    }

    res.json({
      status: 'success',
      results: searchResults.rows.map(row => ({
        commit_hash: row.commit_hash,
        commit_message: row.commit_message,
        similarity: parseFloat(row.similarity.toFixed(4))
      }))
    });

  } catch (error) {
    console.error('Search error:', error);
    console.error('Error details:', {
      query,
      analysisId: req.analysisId,
      error: error.message,
      stack: error.stack
    });

    res.status(500).json({
      status: 'error',
      message: 'Failed to perform semantic search',
      details: error.message
    });
  }
});

// Question Answering
app.post('/api/question-answering', validateSession, async (req, res) => {
  const { question } = req.body;
  if (!question?.trim()) {
    return res.status(400).json({
      status: 'error',
      message: 'Valid question is required'
    });
  }

  try {
    const { rows } = await pool.query(
      `SELECT commit_message
       FROM commit_embeddings
       WHERE code_analysis_id = $1
       ORDER BY created_at DESC
       LIMIT 100`,
      [req.analysisId]
    );

    if (rows.length === 0) {
      return res.status(404).json({
        status: 'error',
        message: 'No commit data available'
      });
    }

    const prompt = `
      You are an assistant analyzing commit messages.

      Commit History:
      ${rows.map(r => r.commit_message).join('\n')}

      Question: ${question}

      Answer:`;

    const answer = await generateCompletion(prompt);
    res.json({ status: 'success', answer });

  } catch (error) {
    console.error('Question answering error:', error);
    res.status(500).json({
      status: 'error',
      message: 'Failed to process question'
    });
  }
});

// Summarization
app.post('/api/summarize', validateSession, async (req, res) => {
  try {
    const { rows } = await pool.query(
      `SELECT commit_message
       FROM commit_embeddings
       WHERE code_analysis_id = $1
       ORDER BY created_at DESC
       LIMIT 500`,
      [req.analysisId]
    );

    if (rows.length === 0) {
      return res.status(404).json({
        status: 'error',
        message: 'No commit messages to summarize'
      });
    }

    const prompt = `
      Provide a concise summary of these commit messages:
      ${rows.map(r => r.commit_message).join('\n')}

      Summary:`;

    const summary = await generateCompletion(prompt);
    res.json({ status: 'success', summary });

  } catch (error) {
    console.error('Summarization error:', error);
    res.status(500).json({
      status: 'error',
      message: 'Failed to generate summary'
    });
  }
});

// Start the server
const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
