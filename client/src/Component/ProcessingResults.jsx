
import React, { useState, useEffect, useMemo, useCallback } from "react";
import {
  FileText,
  BarChart3,
  Eye,
  EyeOff,
  Download,
  Search,
  ChevronDown,
  ChevronUp,
  Copy,
  Check,
  ListCollapse,
  ListRestart,
  Zap,
  RefreshCw,
  Target,
  Info,
  FileDown
} from "lucide-react";
import {
  formatProcessingStats,
  truncateText,
  formatFileSize,
} from "../utile/formatters";
import { semanticSearchService, debugService } from "../services/api";

const ProcessingResults = ({ results, onDocumentReady }) => {
  const [expandedChunks, setExpandedChunks] = useState(new Set());
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedDocument, setSelectedDocument] = useState("all");
  const [selectedPage, setSelectedPage] = useState("");
  const [copiedChunk, setCopiedChunk] = useState(null);
  const [searchMethod, setSearchMethod] = useState("hybrid");
  const [searchResults, setSearchResults] = useState(null);
  const [searching, setSearching] = useState(false);
  const [searchStats, setSearchStats] = useState(null);
  const [showSearchDebug, setShowSearchDebug] = useState(false);
  const [exportFormat, setExportFormat] = useState("json");

  if (!results) return null;

  const toggleChunk = (chunkId) => {
    setExpandedChunks((prev) => {
      const newSet = new Set(prev);
      newSet.has(chunkId) ? newSet.delete(chunkId) : newSet.add(chunkId);
      return newSet;
    });
  };

  const expandAllChunks = (chunks) => {
    setExpandedChunks(new Set(chunks.map((c) => c.chunk_id)));
  };

  const collapseAllChunks = () => {
    setExpandedChunks(new Set());
  };

  const copyToClipboard = async (text, chunkId) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedChunk(chunkId);
      setTimeout(() => setCopiedChunk(null), 2000);
    } catch (error) {
      console.error("Failed to copy:", error);
    }
  };

  // Enhanced download function with PDF support
  const downloadResults = async () => {
    const timestamp = new Date().toISOString().split('T')[0];
    
    if (exportFormat === "json") {
      const dataStr = JSON.stringify(results, null, 2);
      const dataUri = "data:application/json;charset=utf-8," + encodeURIComponent(dataStr);
      const exportFileDefaultName = `pdf_processing_results_${timestamp}.json`;

      const linkElement = document.createElement("a");
      linkElement.setAttribute("href", dataUri);
      linkElement.setAttribute("download", exportFileDefaultName);
      linkElement.click();
    } else if (exportFormat === "pdf") {
      await downloadAsPDF();
    }
  };

  // PDF export function
  const downloadAsPDF = async () => {
    try {
      // Dynamically import jsPDF to avoid bundle size issues
      const { jsPDF } = await import('jspdf');
      
      const doc = new jsPDF();
      const pageWidth = doc.internal.pageSize.getWidth();
      const pageHeight = doc.internal.pageSize.getHeight();
      const margin = 20;
      const maxWidth = pageWidth - (margin * 2);
      let yPosition = margin;

      // Helper function to add new page if needed
      const checkPageBreak = (requiredHeight = 20) => {
        if (yPosition + requiredHeight > pageHeight - margin) {
          doc.addPage();
          yPosition = margin;
          return true;
        }
        return false;
      };

      // Helper function to split text into lines
      const splitTextToLines = (text, maxWidth, fontSize = 10) => {
        doc.setFontSize(fontSize);
        return doc.splitTextToSize(text, maxWidth);
      };

      // Header
      doc.setFontSize(20);
      doc.setFont(undefined, 'bold');
      doc.text('PDF Processing Results', margin, yPosition);
      yPosition += 15;

      // Timestamp
      doc.setFontSize(10);
      doc.setFont(undefined, 'normal');
      doc.text(`Generated on: ${new Date().toLocaleString()}`, margin, yPosition);
      yPosition += 20;

      // Process each document
      documents.forEach(([filename, docData]) => {
        // Document header
        checkPageBreak(30);
        doc.setFontSize(16);
        doc.setFont(undefined, 'bold');
        doc.text(filename, margin, yPosition);
        yPosition += 10;

        // Document metadata
        doc.setFontSize(10);
        doc.setFont(undefined, 'normal');
        const metadata = [
          `Pages: ${docData.metadata?.page_count || 0}`,
          `File Size: ${formatFileSize(docData.metadata?.file_size || 0)}`,
          `Chunks: ${docData.chunks_count || docData.chunks?.length || 0}`
        ].join(' • ');
        doc.text(metadata, margin, yPosition);
        yPosition += 15;

        if (docData.success === false) {
          // Error handling
          doc.setTextColor(200, 0, 0);
          doc.text(`Error: ${docData.error || "Processing failed"}`, margin, yPosition);
          doc.setTextColor(0, 0, 0);
          yPosition += 15;
          return;
        }

        // Process chunks
        const filteredChunks = getFilteredChunks(docData.chunks || []);
        
        if (filteredChunks.length === 0) {
          doc.setTextColor(100, 100, 100);
          doc.text('No chunks found matching the current filters.', margin, yPosition);
          doc.setTextColor(0, 0, 0);
          yPosition += 15;
          return;
        }

        filteredChunks.forEach((chunk, index) => {
          checkPageBreak(40);

          // Chunk header
          doc.setFontSize(12);
          doc.setFont(undefined, 'bold');
          const pageDisplay = chunk.page_numbers && chunk.page_numbers.length > 0
            ? `Pages ${chunk.page_numbers.join(', ')} • `
            : chunk.page_number 
              ? `Page ${chunk.page_number} • `
              : '';
          doc.text(`${pageDisplay}Chunk #${index + 1}`, margin, yPosition);
          yPosition += 8;

          // Search score if available
          const searchScore = getSearchScore(chunk.chunk_id);
          if (searchScore) {
            doc.setFontSize(9);
            doc.setFont(undefined, 'normal');
            doc.setTextColor(0, 100, 200);
            doc.text(`Relevance Score: ${(searchScore.score * 100).toFixed(1)}%`, margin, yPosition);
            doc.setTextColor(0, 0, 0);
            yPosition += 6;
          }

          // Key terms
          if (chunk.key_terms && chunk.key_terms.length > 0) {
            doc.setFontSize(9);
            doc.setTextColor(100, 0, 150);
            doc.text(`Key Terms: ${chunk.key_terms.slice(0, 5).join(', ')}`, margin, yPosition);
            doc.setTextColor(0, 0, 0);
            yPosition += 6;
          }

          // Chunk text
          if (chunk.text) {
            doc.setFontSize(10);
            doc.setFont(undefined, 'normal');
            const textLines = splitTextToLines(chunk.text, maxWidth, 10);
            
            textLines.forEach(line => {
              checkPageBreak(6);
              doc.text(line, margin, yPosition);
              yPosition += 6;
            });
          }

          // Metadata
          if (chunk.word_count || chunk.char_count) {
            yPosition += 3;
            doc.setFontSize(8);
            doc.setTextColor(100, 100, 100);
            const metadata = [];
            if (chunk.word_count) metadata.push(`Words: ${chunk.word_count}`);
            if (chunk.char_count) metadata.push(`Characters: ${chunk.char_count}`);
            doc.text(metadata.join(' • '), margin, yPosition);
            doc.setTextColor(0, 0, 0);
            yPosition += 8;
          }

          yPosition += 5; // Space between chunks
        });

        yPosition += 10; // Space between documents
      });

      // Footer with search info if applicable
      if (searchResults && searchResults.results) {
        checkPageBreak(30);
        doc.setFontSize(10);
        doc.setFont(undefined, 'bold');
        doc.text('Search Summary', margin, yPosition);
        yPosition += 8;

        doc.setFont(undefined, 'normal');
        doc.setFontSize(9);
        const searchInfo = [
          `Query: "${searchTerm}"`,
          `Method: ${searchResults.search_method}`,
          `Results: ${searchResults.total_results}`,
          searchResults.search_time ? `Time: ${(searchResults.search_time * 1000).toFixed(0)}ms` : ''
        ].filter(Boolean);

        searchInfo.forEach(info => {
          checkPageBreak(6);
          doc.text(info, margin, yPosition);
          yPosition += 6;
        });
      }

      // Save the PDF
      const timestamp = new Date().toISOString().split('T')[0];
      const filename = searchTerm 
        ? `pdf_processing_results_search_${searchTerm.replace(/[^a-z0-9]/gi, '_')}_${timestamp}.pdf`
        : `pdf_processing_results_${timestamp}.pdf`;
      
      doc.save(filename);

    } catch (error) {
      console.error("Failed to generate PDF:", error);
      alert("Failed to generate PDF. Please try again or use JSON export.");
    }
  };

  // Enhanced semantic search function
  const performSemanticSearch = async () => {
    if (!searchTerm.trim()) {
      setSearchResults(null);
      return;
    }

    setSearching(true);
    try {
      let searchResponse;
      
      // Get document IDs for filtering
      const documentIds = selectedDocument === "all" ? null : [selectedDocument];
      
      switch (searchMethod) {
        case "semantic":
          searchResponse = await semanticSearchService.search(searchTerm, {
            documentIds,
            limit: 20
          });
          break;
        case "keyword":
          searchResponse = await semanticSearchService.keywordSearch(searchTerm, {
            documentIds,
            limit: 20
          });
          break;
        case "hybrid":
        default:
          searchResponse = await semanticSearchService.hybridSearch(searchTerm, {
            documentIds,
            limit: 20
          });
          break;
      }

      setSearchResults(searchResponse);
      
      // Get search stats if debug mode is enabled
      if (showSearchDebug) {
        const stats = await semanticSearchService.getStats();
        setSearchStats(stats);
      }

    } catch (error) {
      console.error("Search failed:", error);
      setSearchResults({ results: [], error: error.message });
    } finally {
      setSearching(false);
    }
  };

  // Debounced search effect
  useEffect(() => {
    if (!searchTerm.trim()) {
      setSearchResults(null);
      return;
    }

    const timeoutId = setTimeout(() => {
      performSemanticSearch();
    }, 500);

    return () => clearTimeout(timeoutId);
  }, [searchTerm, searchMethod, selectedDocument]);

  // Handle single/multiple documents properly - memoized to prevent re-renders
  const documents = useMemo(() => {
    if (!results) return [];
    
    // Only log once when results actually change
    console.log('ProcessingResults received:', results);
    
    if (results.results && typeof results.results === 'object') {
      return Object.entries(results.results);
    } else if (results.success && results.filename) {
      return [[results.filename, results]];
    } else if (results.filename) {
      return [[results.filename, results]];
    } else {
      console.warn('Invalid results format:', results);
      return [];
    }
  }, [results]);

  const filteredDocuments = documents.filter(
    ([filename]) => selectedDocument === "all" || filename === selectedDocument
  );

  // Enhanced chunk filtering with search results - memoized for performance
  const getFilteredChunks = useCallback((chunks) => {
    if (!chunks || !Array.isArray(chunks)) {
      console.warn('Invalid chunks format:', chunks);
      return [];
    }

    let filtered = chunks;
    
    // If we have search results, filter by them
    if (searchResults && searchResults.results && searchResults.results.length > 0) {
      const searchChunkIds = new Set(searchResults.results.map(r => r.chunk_id));
      filtered = filtered.filter(chunk => {
        const chunkId = chunk.chunk_id || (chunk.key && chunk.key.includes('chunk'));
        return searchChunkIds.has(chunkId);
      });
      
      // Sort by search relevance
      filtered.sort((a, b) => {
        const aResult = searchResults.results.find(r => r.chunk_id === a.chunk_id);
        const bResult = searchResults.results.find(r => r.chunk_id === b.chunk_id);
        return (bResult?.score || 0) - (aResult?.score || 0);
      });
    } else if (searchTerm && !searchResults) {
      // Fallback to basic text filtering if semantic search is not available
      const searchLower = searchTerm.toLowerCase();
      filtered = filtered.filter((chunk) =>
        (chunk.text && chunk.text.toLowerCase().includes(searchLower)) ||
        (chunk.chunk_id && chunk.chunk_id.toLowerCase().includes(searchLower)) ||
        (chunk.key_terms && chunk.key_terms.some(term => 
          term.toLowerCase().includes(searchLower)
        ))
      );
    }
    
    // Filter by page number
    if (selectedPage) {
      const pageNum = Number(selectedPage);
      filtered = filtered.filter((chunk) => {
        if (chunk.page_numbers && Array.isArray(chunk.page_numbers)) {
          return chunk.page_numbers.includes(pageNum);
        } else if (chunk.page_number) {
          return chunk.page_number === pageNum;
        }
        return false;
      });
    }
    
    return filtered;
  }, [searchResults, searchTerm, selectedPage]);

  // Get search result score for a chunk - memoized
  const getSearchScore = useCallback((chunkId) => {
    if (!searchResults || !searchResults.results) return null;
    const result = searchResults.results.find(r => r.chunk_id === chunkId);
    return result ? {
      score: result.score,
      embedding_score: result.embedding_score,
      method: searchResults.search_method
    } : null;
  }, [searchResults]);

  useEffect(() => {
    if (documents.length > 0 && onDocumentReady) {
      const processedDocs = documents
        .filter(([, docData]) => docData.success !== false)
        .map(([filename, docData]) => ({
          filename,
          chunks: docData.chunks?.length || docData.chunks_count || 0,
          metadata: docData.metadata,
        }));
      onDocumentReady(processedDocs);
    }
  }, [results, onDocumentReady, documents]);

  return (
    <div className="card p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-800 flex items-center space-x-2">
          <BarChart3 className="w-6 h-6 text-primary-500" />
          <span>Processing Results</span>
          {searchResults && (
            <span className="text-sm bg-blue-100 text-blue-800 px-2 py-1 rounded">
              {searchResults.results?.length || 0} matches
            </span>
          )}
        </h2>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowSearchDebug(!showSearchDebug)}
            className="btn-light text-sm flex items-center space-x-1"
          >
            <Info className="w-4 h-4" />
            <span>Debug</span>
          </button>
          
          {/* Export Format Selector */}
          <select
            value={exportFormat}
            onChange={(e) => setExportFormat(e.target.value)}
            className="text-sm border border-gray-300 rounded px-2 py-1"
          >
            <option value="json">JSON</option>
            <option value="pdf">PDF</option>
          </select>
          
          <button
            onClick={downloadResults}
            className="btn-secondary text-sm flex items-center space-x-1"
          >
            {exportFormat === "pdf" ? (
              <FileDown className="w-4 h-4" />
            ) : (
              <Download className="w-4 h-4" />
            )}
            <span>Export {exportFormat.toUpperCase()}</span>
          </button>
        </div>
      </div>

      {/* Enhanced Search Controls */}
      <div className="mb-6 space-y-4">
        {/* Document and Page Selectors */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Document
            </label>
            <select
              value={selectedDocument}
              onChange={(e) => setSelectedDocument(e.target.value)}
              className="input-field"
            >
              <option value="all">All Documents ({documents.length})</option>
              {documents.map(([filename]) => (
                <option key={filename} value={filename}>
                  {filename}
                </option>
              ))}
            </select>
          </div>

          {filteredDocuments.length === 1 &&
            filteredDocuments[0][1].chunks?.some((c) => c.page_numbers?.length > 0 || c.page_number) && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Select Page
                </label>
                <select
                  value={selectedPage}
                  onChange={(e) => setSelectedPage(e.target.value)}
                  className="input-field"
                >
                  <option value="">All Pages</option>
                  {[
                    ...new Set(
                      filteredDocuments[0][1].chunks.flatMap((c) => 
                        c.page_numbers || (c.page_number ? [c.page_number] : [])
                      )
                    ),
                  ]
                    .sort((a, b) => a - b)
                    .map((page) => (
                      <option key={page} value={page}>
                        Page {page}
                      </option>
                    ))}
                </select>
              </div>
            )}
        </div>

        {/* Enhanced Search */}
        <div className="space-y-3">
          <div className="flex items-center space-x-2">
            <label className="block text-sm font-medium text-gray-700">
              Semantic Search
            </label>
            <select
              value={searchMethod}
              onChange={(e) => setSearchMethod(e.target.value)}
              className="text-sm border border-gray-300 rounded px-2 py-1"
            >
              <option value="hybrid">Hybrid (Recommended)</option>
              <option value="semantic">Semantic Only</option>
              <option value="keyword">Keyword Only</option>
            </select>
          </div>
          
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            {searching && (
              <RefreshCw className="absolute right-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-primary-500 animate-spin" />
            )}
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search chunks with AI-powered semantic understanding..."
              className="input-field pl-10 pr-10"
            />
          </div>

          {searchResults?.search_method && (
            <div className="flex items-center space-x-4 text-xs text-gray-600">
              <span className="flex items-center space-x-1">
                <Target className="w-3 h-3" />
                <span>Method: {searchResults.search_method}</span>
              </span>
              {searchResults.search_time && (
                <span className="flex items-center space-x-1">
                  <Zap className="w-3 h-3" />
                  <span>Time: {(searchResults.search_time * 1000).toFixed(0)}ms</span>
                </span>
              )}
            </div>
          )}
        </div>

        {/* Search Debug Info */}
        {showSearchDebug && searchStats && (
          <div className="bg-gray-50 p-3 rounded text-xs">
            <h4 className="font-medium mb-2">Search Engine Statistics</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              <div>
                <span className="text-gray-600">Total Chunks:</span>
                <span className="ml-1 font-mono">{searchStats.search_engine_stats?.total_chunks || 0}</span>
              </div>
              <div>
                <span className="text-gray-600">Model:</span>
                <span className="ml-1 font-mono">{searchStats.search_engine_stats?.model_name || "N/A"}</span>
              </div>
              <div>
                <span className="text-gray-600">Dimensions:</span>
                <span className="ml-1 font-mono">{searchStats.search_engine_stats?.embedding_dimension || 0}</span>
              </div>
              <div>
                <span className="text-gray-600">Index Size:</span>
                <span className="ml-1 font-mono">{(searchStats.search_engine_stats?.index_size_mb || 0).toFixed(1)}MB</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Results for each document */}
      <div className="space-y-6">
        {filteredDocuments.map(([filename, docData]) => {
          if (docData.success === false) {
            return (
              <div
                key={filename}
                className="bg-red-50 border border-red-200 rounded-lg p-4"
              >
                <div className="flex items-center space-x-2 text-red-800">
                  <FileText className="w-5 h-5" />
                  <span className="font-medium">{filename}</span>
                </div>
                <p className="text-red-600 mt-2">
                  {docData.error || "Processing failed"}
                </p>
              </div>
            );
          }

          const formattedStats = formatProcessingStats(docData.processing_stats);
          const filteredChunks = getFilteredChunks(docData.chunks || []);

          return (
            <div key={filename} className="border border-gray-200 rounded-lg">
              {/* Header */}
              <div className="bg-gray-50 p-4 border-b border-gray-200">
                <div className="flex justify-between items-start">
                  <div className="flex items-center space-x-3">
                    <FileText className="w-6 h-6 text-primary-500" />
                    <div>
                      <h3 className="font-semibold">{filename}</h3>
                      <div className="text-sm text-gray-600 space-y-1">
                        <div>
                          {docData.metadata?.page_count || 0} pages •{" "}
                          {formatFileSize(docData.metadata?.file_size || 0)}
                          {docData.chunks_count && (
                            <span> • {docData.chunks_count} chunks</span>
                          )}
                        </div>
                        {searchResults && (
                          <div className="flex items-center space-x-2 text-xs">
                            <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded">
                              {filteredChunks.length} matches found
                            </span>
                            <span className="text-gray-500">
                              Search method: {searchResults.search_method}
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                  <div className="flex space-x-2">
                    <button
                      onClick={() => expandAllChunks(filteredChunks)}
                      className="btn-light flex items-center space-x-1"
                    >
                      <ListCollapse className="w-4 h-4" />
                      <span>Expand All</span>
                    </button>
                    <button
                      onClick={collapseAllChunks}
                      className="btn-light flex items-center space-x-1"
                    >
                      <ListRestart className="w-4 h-4" />
                      <span>Collapse</span>
                    </button>
                  </div>
                </div>
              </div>

              {/* Enhanced Chunks Display */}
              <div className="max-h-[450px] overflow-y-auto divide-y">
                {filteredChunks.length > 0 ? (
                  filteredChunks.map((chunk, index) => {
                    const isExpanded = expandedChunks.has(chunk.chunk_id);
                    const displayText = isExpanded
                      ? chunk.text
                      : truncateText(chunk.text || '', 200);

                    const pageDisplay = chunk.page_numbers && chunk.page_numbers.length > 0
                      ? `Pages ${chunk.page_numbers.join(', ')} • `
                      : chunk.page_number 
                        ? `Page ${chunk.page_number} • `
                        : '';

                    const searchScore = getSearchScore(chunk.chunk_id);

                    return (
                      <div
                        key={chunk.chunk_id || index}
                        className={`p-4 transition ${
                          searchScore ? 'bg-blue-50 border-l-4 border-l-blue-400' : 'hover:bg-gray-50'
                        }`}
                      >
                        <div className="flex justify-between items-start mb-2">
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="text-xs bg-primary-100 text-primary-800 px-2 py-1 rounded">
                              {pageDisplay}Chunk #{index + 1}
                            </span>
                            
                            {/* Search Score Display */}
                            {searchScore && (
                              <div className="flex items-center space-x-2">
                                <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                                  Relevance: {(searchScore.score * 100).toFixed(1)}%
                                </span>
                                {searchScore.embedding_score > 0 && (
                                  <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded">
                                    Semantic: {(searchScore.embedding_score * 100).toFixed(1)}%
                                  </span>
                                )}
                              </div>
                            )}
                            
                            {/* Key Terms */}
                            {chunk.key_terms && chunk.key_terms.length > 0 && (
                              <div className="flex flex-wrap gap-1">
                                {chunk.key_terms.slice(0, 3).map((term, termIndex) => (
                                  <span
                                    key={termIndex}
                                    className="text-xs bg-purple-100 text-purple-800 px-2 py-1 rounded-full"
                                  >
                                    {term}
                                  </span>
                                ))}
                                {chunk.key_terms.length > 3 && (
                                  <span className="text-xs text-gray-500">
                                    +{chunk.key_terms.length - 3} more
                                  </span>
                                )}
                              </div>
                            )}
                          </div>
                          
                          <div className="flex space-x-1">
                            <button
                              onClick={() => copyToClipboard(chunk.text || '', chunk.chunk_id)}
                              className="p-1 text-gray-400 hover:text-gray-600"
                              title="Copy chunk text"
                            >
                              {copiedChunk === chunk.chunk_id ? (
                                <Check className="w-4 h-4 text-green-500" />
                              ) : (
                                <Copy className="w-4 h-4" />
                              )}
                            </button>
                            <button
                              onClick={() => toggleChunk(chunk.chunk_id)}
                              className="p-1 text-gray-400 hover:text-gray-600"
                              title={isExpanded ? "Collapse" : "Expand"}
                            >
                              {isExpanded ? (
                                <EyeOff className="w-4 h-4" />
                              ) : (
                                <Eye className="w-4 h-4" />
                              )}
                            </button>
                          </div>
                        </div>

                        {/* Enhanced Text Display with Search Highlighting */}
                        <div className="mt-2 text-sm text-gray-700">
                          {searchTerm && !searchResults ? (
                            // Fallback highlighting for non-semantic search
                            <span
                              dangerouslySetInnerHTML={{
                                __html: displayText.replace(
                                  new RegExp(`(${searchTerm})`, "gi"),
                                  '<mark class="bg-yellow-200 px-1 rounded">$1</mark>'
                                ),
                              }}
                            />
                          ) : searchTerm && searchResults ? (
                            // Enhanced highlighting for semantic search
                            <span
                              dangerouslySetInnerHTML={{
                                __html: displayText.replace(
                                  new RegExp(`(${searchTerm})`, "gi"),
                                  '<mark class="bg-blue-200 px-1 rounded font-medium">$1</mark>'
                                ),
                              }}
                            />
                          ) : (
                            displayText
                          )}
                        </div>

                        {/* Show More/Less Button */}
                        {chunk.text && chunk.text.length > 200 && (
                          <button
                            onClick={() => toggleChunk(chunk.chunk_id)}
                            className="mt-2 flex items-center space-x-1 text-primary-600 text-sm hover:text-primary-700"
                          >
                            {isExpanded ? (
                              <>
                                <ChevronUp className="w-4 h-4" />
                                <span>Show Less</span>
                              </>
                            ) : (
                              <>
                                <ChevronDown className="w-4 h-4" />
                                <span>Show More</span>
                              </>
                            )}
                          </button>
                        )}

                        {/* Enhanced Metadata Display */}
                        <div className="mt-3 flex flex-wrap items-center gap-4 text-xs text-gray-500">
                          {(chunk.word_count || chunk.char_count) && (
                            <div className="flex space-x-3">
                              {chunk.word_count && <span>Words: {chunk.word_count}</span>}
                              {chunk.char_count && <span>Characters: {chunk.char_count}</span>}
                            </div>
                          )}
                          
                          {searchScore && showSearchDebug && (
                            <div className="flex space-x-3 text-xs bg-gray-100 px-2 py-1 rounded">
                              <span>Method: {searchScore.method}</span>
                              <span>Score: {searchScore.score.toFixed(3)}</span>
                              {searchScore.embedding_score > 0 && (
                                <span>Embedding: {searchScore.embedding_score.toFixed(3)}</span>
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })
                ) : (
                  <div className="p-6 text-center text-gray-500">
                    {searchTerm ? (
                      <div className="space-y-2">
                        <p>No chunks match your search for "{searchTerm}"</p>
                        {searchResults?.error && (
                          <p className="text-red-500 text-sm">
                            Search error: {searchResults.error}
                          </p>
                        )}
                        <p className="text-sm">
                          Try adjusting your search terms or using a different search method.
                        </p>
                      </div>
                    ) : docData.chunks?.length === 0 ? (
                      'No chunks were generated from this document.'
                    ) : (
                      'No chunks match your filter criteria.'
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Search Results Summary */}
      {searchResults && searchResults.results && (
        <div className="mt-4 p-4 bg-blue-50 rounded-lg">
          <h4 className="font-medium text-blue-900 mb-2">Search Results Summary</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-blue-700">Total Results:</span>
              <span className="ml-1 font-mono">{searchResults.total_results}</span>
            </div>
            <div>
              <span className="text-blue-700">Search Method:</span>
              <span className="ml-1 font-mono">{searchResults.search_method}</span>
            </div>
            {searchResults.search_time && (
              <div>
                <span className="text-blue-700">Search Time:</span>
                <span className="ml-1 font-mono">{(searchResults.search_time * 1000).toFixed(0)}ms</span>
              </div>
            )}
            <div>
              <span className="text-blue-700">Query:</span>
              <span className="ml-1 font-mono">"{searchTerm}"</span>
            </div>
          </div>
        </div>
      )}

      {/* Development Debug Info */}
      {process.env.NODE_ENV === 'development' && showSearchDebug && (
        <details className="mt-4 text-xs text-gray-500">
          <summary className="cursor-pointer font-medium">Debug Information</summary>
          <div className="mt-2 space-y-2">
            <div className="p-2 bg-gray-100 rounded overflow-x-auto">
              <strong>Results Structure:</strong>
              <pre className="mt-1">
                {JSON.stringify({
                  resultsKeys: Object.keys(results),
                  documentsCount: documents.length,
                  hasChunks: documents.map(([name, data]) => [name, !!data.chunks]),
                  searchResultsKeys: searchResults ? Object.keys(searchResults) : null
                }, null, 2)}
              </pre>
            </div>
            {searchStats && (
              <div className="p-2 bg-gray-100 rounded overflow-x-auto">
                <strong>Search Engine Stats:</strong>
                <pre className="mt-1">
                  {JSON.stringify(searchStats, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </details>
      )}
    </div>
  );
};

export default ProcessingResults;