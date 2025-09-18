// Format file size in bytes to human readable format
export const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

// Format number with commas
export const formatNumber = (num) => {
  return new Intl.NumberFormat().format(num);
};

// Truncate text to specified length
export const truncateText = (text, maxLength = 100) => {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength) + '...';
};

// Calculate reading time estimate
export const calculateReadingTime = (wordCount) => {
  const wordsPerMinute = 200; // Average reading speed
  const minutes = Math.ceil(wordCount / wordsPerMinute);
  return `${minutes} min read`;
};

// Format processing statistics
export const formatProcessingStats = (stats) => {
  if (!stats) return {};
  
  return {
    totalChunks: formatNumber(stats.total_chunks || 0),
    totalWords: formatNumber(stats.total_words || 0),
    totalCharacters: formatNumber(stats.total_characters || 0),
    avgChunkSize: Math.round(stats.avg_chunk_size || 0),
    maxChunkSize: formatNumber(stats.max_chunk_size || 0),
    minChunkSize: formatNumber(stats.min_chunk_size || 0),
    readingTime: calculateReadingTime(stats.total_words || 0)
  };
};

// Validate file type
export const isValidPDFFile = (file) => {
  if (!file) return false;
  
  const validTypes = ['application/pdf'];
  const validExtensions = ['.pdf'];
  
  const hasValidType = validTypes.includes(file.type);
  const hasValidExtension = validExtensions.some(ext => 
    file.name.toLowerCase().endsWith(ext)
  );
  
  return hasValidType || hasValidExtension;
};

// Get file extension
export const getFileExtension = (filename) => {
  return filename.slice((filename.lastIndexOf('.') - 1 >>> 0) + 2);
};

// Format date
export const formatDate = (dateString) => {
  if (!dateString) return 'Not available';
  
  try {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  } catch {
    return 'Invalid date';
  }
};