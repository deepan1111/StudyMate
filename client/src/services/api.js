
import axios from "axios";

const API_BASE_URL = "http://localhost:8000";

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, 
});

// Request interceptor for debugging
api.interceptors.request.use((config) => {
  console.log("🔹 API Request:", {
    method: config.method?.toUpperCase(),
    url: config.baseURL + config.url,
    headers: config.headers,
    data: config.data,
  });
  return config;
});

// Response interceptor for debugging
api.interceptors.response.use(
  (response) => {
    console.log("✅ API Response:", {
      status: response.status,
      url: response.config.url,
      data: response.data,
    });
    return response;
  },
  (error) => {
    console.error("❌ API Error:", {
      status: error.response?.status,
      url: error.config?.url,
      message: error.message,
      data: error.response?.data,
    });
    return Promise.reject(error);
  }
);

// ---------------- PDF Service ----------------
export const pdfService = {
  // Upload single PDF with enhanced retry logic
  uploadSingle: async (file, chunkSize = 1000, chunkOverlap = 200, retries = 2) => {
    for (let attempt = 1; attempt <= retries; attempt++) {
      try {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("chunk_size", chunkSize.toString());
        formData.append("chunk_overlap", chunkOverlap.toString());

        console.log(`Upload attempt ${attempt}/${retries} for ${file.name}`);
        
        const response = await api.post("/upload/single", formData, {
          headers: { "Content-Type": "multipart/form-data" },
          timeout: 600000, // 10 minutes
          onUploadProgress: (progressEvent) => {
            const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            if (progress % 10 === 0) { // Log every 10%
              console.log(`Upload progress: ${progress}%`);
            }
          }
        });
        
        return response.data;
      } catch (error) {
        console.error(`Upload attempt ${attempt} failed:`, error);
        
        if (attempt === retries) {
          throw error;
        }
        
        // Wait before retry
        console.log(`Waiting 3 seconds before retry...`);
        await new Promise(resolve => setTimeout(resolve, 3000));
      }
    }
  },

  // Upload multiple PDFs using single upload endpoint
  uploadMultiple: async (files, chunkSize = 1000, chunkOverlap = 200) => {
    const results = [];
    const errors = [];

    for (const file of files) {
      try {
        const result = await pdfService.uploadSingle(file, chunkSize, chunkOverlap);
        results.push({
          filename: file.name,
          success: true,
          ...result
        });
      } catch (error) {
        errors.push({
          filename: file.name,
          success: false,
          error: error.response?.data?.detail || error.message
        });
      }
    }

    return {
      success: errors.length === 0,
      results: results.length > 0 ? 
        Object.fromEntries(results.map(r => [r.filename, r])) : {},
      errors: errors,
      total_files: files.length,
      successful_files: results.length,
      failed_files: errors.length
    };
  },

  // Process raw text with proper error handling
  processText: async (text, question = null) => {
    try {
      const payload = question ? { text, question } : { text };
      const response = await api.post("/process/text", payload, {
        headers: { "Content-Type": "application/json" }
      });
      return response.data;
    } catch (error) {
      console.error("❌ Error processing text:", error.response?.data || error.message);
      
      return {
        success: false,
        message: "Text processing failed",
        error: error.response?.data?.detail || error.message,
        fallback: true
      };
    }
  },

  // ✅ ADD: Missing health check method
  healthCheck: async () => {
    const response = await api.get("/health");
    return response.data;
  },

  // Get supported formats
  getSupportedFormats: async () => {
    return {
      formats: ['.pdf'],
      max_size_mb: 100,
      description: 'PDF documents only',
      note: 'This is a frontend mock - backend only supports PDF uploads'
    };
  },
};

// ---------------- StudyMate Q&A Service with Semantic Search ----------------
export const studyMateService = {
  // Ask question with semantic search options
  askQuestion: async (question, sessionId = null, options = {}) => {
    const payload = {
      question,
      session_id: sessionId,
      document_ids: options.documentIds || null,
      max_context_chunks: options.maxChunks || 5,
      temperature: options.temperature || 0.3,
      use_semantic_search: options.useSemanticSearch !== false, // Default to true
    };

    console.log('DEBUG: Sending ask request with semantic search:', payload);
    
    const response = await api.post("/ask", payload);
    return response.data;
  },

  // Get conversation history using sessions endpoint
  getConversationHistory: async (sessionId) => {
    try {
      const sessions = await api.get("/sessions");
      const session = sessions.data.sessions?.find(s => s.id === sessionId);
      
      return {
        success: true,
        messages: session?.messages || [],
        session_id: sessionId,
        created_at: session?.created_at || null
      };
    } catch (error) {
      console.error("Failed to get conversation history:", error);
      return { 
        success: false,
        messages: [], 
        session_id: sessionId,
        error: error.message
      };
    }
  },

  // Create new session
  createSession: async () => {
    const response = await api.post("/sessions/create");
    return response.data;
  },

  // Get all sessions
  getSessions: async () => {
    const response = await api.get("/sessions");
    return response.data;
  },

  // Delete session
  deleteSession: async (sessionId) => {
    const response = await api.delete(`/sessions/${sessionId}`);
    return response.data;
  },

  // Get document library with search stats
  getDocuments: async () => {
    const response = await api.get("/documents");
    return response.data;
  },

  // Delete document (also removes from search index)
  deleteDocument: async (documentId) => {
    const response = await api.delete(`/documents/${documentId}`);
    return response.data;
  },

  // Enhanced search with semantic search options
  searchDocuments: async (query, options = {}) => {
    const payload = {
      query,
      document_ids: options.documentIds || null,
      limit: options.limit || 10,
      include_key_terms: options.includeKeyTerms !== false,
      use_semantic_search: options.useSemanticSearch !== false,
      search_method: options.searchMethod || "hybrid" // "semantic", "keyword", "hybrid"
    };

    const response = await api.post("/search", payload);
    return response.data;
  },

  // Find similar chunks using semantic search
  findSimilarChunks: async (chunkId, options = {}) => {
    const payload = {
      chunk_id: chunkId,
      limit: options.limit || 5
    };

    const response = await api.post("/search/similar", payload);
    return response.data;
  },

  // Get system statistics including search engine stats
  getStats: async () => {
    try {
      const response = await api.get("/stats");
      return response.data;
    } catch (error) {
      console.error("Failed to get stats:", error);
      return {
        documents: 0,
        active_sessions: 0,
        total_chunks: 0,
        total_words: 0,
        search_engine: {},
        error: error.message
      };
    }
  },

  // Get detailed search engine statistics
  getSearchStats: async () => {
    try {
      const response = await api.get("/search/stats");
      return response.data;
    } catch (error) {
      console.error("Failed to get search stats:", error);
      return {
        error: error.message,
        search_engine_stats: {}
      };
    }
  },

  // Rebuild search index
  reindexDocuments: async () => {
    try {
      const response = await api.post("/search/reindex");
      return response.data;
    } catch (error) {
      console.error("Failed to reindex documents:", error);
      return {
        success: false,
        error: error.message
      };
    }
  },

  // Get LLM health status
  getLLMHealth: async () => {
    try {
      const response = await api.get("/llm/health");
      return response.data;
    } catch (error) {
      return {
        status: "unhealthy",
        error: error.message
      };
    }
  },

  // Estimate LLM cost with semantic search preview
  estimateCost: async (question, contextChunks = 5) => {
    try {
      const response = await api.post("/llm/cost-estimate", {
        question,
        context_chunks: contextChunks
      });
      return response.data;
    } catch (error) {
      return {
        error: error.message,
        estimated_costs: { huggingface: 0 }
      };
    }
  }
};

// ---------------- Semantic Search Service ----------------
export const semanticSearchService = {
  // Perform semantic search
  search: async (query, options = {}) => {
    return await studyMateService.searchDocuments(query, {
      ...options,
      useSemanticSearch: true,
      searchMethod: "semantic"
    });
  },

  // Perform hybrid search (semantic + keyword)
  hybridSearch: async (query, options = {}) => {
    return await studyMateService.searchDocuments(query, {
      ...options,
      useSemanticSearch: true,
      searchMethod: "hybrid"
    });
  },

  // Perform keyword-only search
  keywordSearch: async (query, options = {}) => {
    return await studyMateService.searchDocuments(query, {
      ...options,
      useSemanticSearch: false,
      searchMethod: "keyword"
    });
  },

  // Compare search methods
  compareSearchMethods: async (query, options = {}) => {
    try {
      const [semanticResults, hybridResults, keywordResults] = await Promise.all([
        semanticSearchService.search(query, options),
        semanticSearchService.hybridSearch(query, options),
        semanticSearchService.keywordSearch(query, options)
      ]);

      return {
        query,
        comparison: {
          semantic: {
            method: "semantic",
            results: semanticResults.results,
            total_results: semanticResults.total_results,
            search_time: semanticResults.search_time
          },
          hybrid: {
            method: "hybrid",
            results: hybridResults.results,
            total_results: hybridResults.total_results,
            search_time: hybridResults.search_time
          },
          keyword: {
            method: "keyword", 
            results: keywordResults.results,
            total_results: keywordResults.total_results,
            search_time: keywordResults.search_time
          }
        }
      };
    } catch (error) {
      console.error("Error comparing search methods:", error);
      return {
        error: error.message,
        query
      };
    }
  },

  // Get search statistics
  getStats: async () => {
    return await studyMateService.getSearchStats();
  },

  // Reindex all documents
  reindex: async () => {
    return await studyMateService.reindexDocuments();
  },

  // Find similar content to a given chunk
  findSimilar: async (chunkId, options = {}) => {
    return await studyMateService.findSimilarChunks(chunkId, options);
  }
};

// ---------------- Debug Service ----------------
export const debugService = {
  // Test all endpoints including semantic search
  runConnectionTests: async () => {
    const results = {};
    
    try {
      results.health = await pdfService.healthCheck();
    } catch (error) {
      results.health = { error: error.message };
    }

    try {
      results.session = await studyMateService.createSession();
    } catch (error) {
      results.session = { error: error.message };
    }

    try {
      results.documents = await studyMateService.getDocuments();
    } catch (error) {
      results.documents = { error: error.message };
    }

    try {
      results.stats = await studyMateService.getStats();
    } catch (error) {
      results.stats = { error: error.message };
    }

    try {
      results.searchStats = await semanticSearchService.getStats();
    } catch (error) {
      results.searchStats = { error: error.message };
    }

    return results;
  },

  // Get debug info about a specific document
  getDocumentDebug: async (documentId) => {
    try {
      const response = await api.get(`/debug/document/${documentId}`);
      return response.data;
    } catch (error) {
      return { error: error.message };
    }
  },

  // Test semantic search functionality
  testSemanticSearch: async (testQuery = "artificial intelligence") => {
    try {
      const comparison = await semanticSearchService.compareSearchMethods(testQuery, {
        limit: 3
      });
      return {
        success: true,
        test_query: testQuery,
        comparison_results: comparison
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
        test_query: testQuery
      };
    }
  },

  // Performance test for different search methods
  performanceTest: async (queries = ["machine learning", "neural networks", "data science"]) => {
    const results = [];
    
    for (const query of queries) {
      try {
        const startTime = performance.now();
        const comparison = await semanticSearchService.compareSearchMethods(query, {
          limit: 5
        });
        const endTime = performance.now();
        
        results.push({
          query,
          total_time: endTime - startTime,
          comparison,
          success: true
        });
      } catch (error) {
        results.push({
          query,
          error: error.message,
          success: false
        });
      }
    }
    
    return {
      test_queries: queries,
      results,
      summary: {
        total_tests: queries.length,
        successful_tests: results.filter(r => r.success).length,
        failed_tests: results.filter(r => !r.success).length
      }
    };
  }
};

// ---------------- Configuration Service ----------------
export const configService = {
  // Get default search configuration
  getSearchConfig: () => ({
    defaultSearchMethod: "hybrid",
    useSemanticSearch: true,
    maxContextChunks: 5,
    searchLimit: 10,
    scoreThreshold: 0.1,
    semanticWeight: 0.7,
    keywordWeight: 0.3
  }),

  // Get embedding model information
  getEmbeddingInfo: async () => {
    try {
      const stats = await semanticSearchService.getStats();
      return {
        model_name: stats.search_engine_stats?.model_name || "all-MiniLM-L6-v2",
        embedding_dimension: stats.search_engine_stats?.embedding_dimension || 384,
        total_chunks: stats.search_engine_stats?.total_chunks || 0,
        index_size_mb: stats.search_engine_stats?.index_size_mb || 0
      };
    } catch (error) {
      return {
        error: error.message,
        model_name: "unknown",
        embedding_dimension: 0
      };
    }
  },

  // Validate search query
  validateSearchQuery: (query) => {
    if (!query || typeof query !== 'string') {
      return { valid: false, error: "Query must be a non-empty string" };
    }
    
    if (query.trim().length < 2) {
      return { valid: false, error: "Query must be at least 2 characters long" };
    }
    
    if (query.length > 1000) {
      return { valid: false, error: "Query must be less than 1000 characters" };
    }
    
    return { valid: true };
  },

  // Optimize search parameters based on query type
  optimizeSearchParams: (query, documentCount = 1) => {
    const baseConfig = configService.getSearchConfig();
    
    // Adjust based on query length
    if (query.length > 100) {
      // Longer queries benefit more from semantic search
      return {
        ...baseConfig,
        searchMethod: "semantic",
        semanticWeight: 0.9,
        keywordWeight: 0.1
      };
    } else if (query.length < 20) {
      // Short queries may benefit from hybrid approach
      return {
        ...baseConfig,
        searchMethod: "hybrid",
        semanticWeight: 0.6,
        keywordWeight: 0.4
      };
    }
    
    // Adjust based on document count
    if (documentCount > 10) {
      return {
        ...baseConfig,
        maxContextChunks: 8,
        searchLimit: 15
      };
    }
    
    return baseConfig;
  }
};

// Default export axios instance
export default api;