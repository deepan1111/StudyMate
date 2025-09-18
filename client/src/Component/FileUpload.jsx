import React, { useState, useCallback } from 'react';
import { Upload, FileText, X, Settings } from 'lucide-react';
import { pdfService } from '../services/api';
import { isValidPDFFile, formatFileSize } from '../utile/formatters';
import LoadingSpinner from './LoadingSpinner';

const FileUpload = ({ 
  onProcessingComplete, 
  onProcessingStart, 
  onProcessingEnd, 
  onError 
}) => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [settings, setSettings] = useState({
    chunkSize: 1000,
    chunkOverlap: 200
  });
  const [showSettings, setShowSettings] = useState(false);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const files = Array.from(e.dataTransfer.files);
      addFiles(files);
    }
  }, []);

  const handleFileSelect = (e) => {
    if (e.target.files) {
      const files = Array.from(e.target.files);
      addFiles(files);
    }
  };

  const addFiles = (files) => {
    const validFiles = files.filter(file => {
      if (!isValidPDFFile(file)) {
        onError?.(`Invalid file type: ${file.name}. Please select PDF files only.`);
        return false;
      }
      if (file.size > 100 * 1024 * 1024) { // 100MB limit
        onError?.(`File too large: ${file.name}. Maximum size is 100MB.`);
        return false;
      }
      return true;
    });

    setSelectedFiles(prev => [...prev, ...validFiles]);
  };

  const removeFile = (index) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const processFiles = async () => {
    if (selectedFiles.length === 0) {
      onError?.('Please select at least one PDF file.');
      return;
    }

    setIsProcessing(true);
    onProcessingStart?.(); // ✅ Notify parent that processing started

    try {
      let result;
      if (selectedFiles.length === 1) {
        console.log(`Uploading single file: ${selectedFiles[0].name}`);
        result = await pdfService.uploadSingle(
          selectedFiles[0], 
          settings.chunkSize, 
          settings.chunkOverlap
        );
        if (result.success !== false) {
          result.filename = result.filename || selectedFiles[0].name;
        }
      } else {
        console.log(`Uploading ${selectedFiles.length} files...`);
        result = await pdfService.uploadMultiple(
          selectedFiles, 
          settings.chunkSize, 
          settings.chunkOverlap
        );
      }

      console.log('Upload completed successfully');
      onProcessingComplete?.(result);
      setSelectedFiles([]); // clear after success

    } catch (error) {
      console.error('Processing error:', error);

      if (error.code === 'ECONNABORTED') {
        onError?.('Upload timed out. The file may be too large or the server is busy. Please try again.');
      } else if (error.message?.includes('Network Error')) {
        onError?.('Network connection lost during upload. Please check your connection and try again.');
      } else {
        onError?.(error.response?.data?.detail || 'Processing failed. Please try again.');
      }
    } finally {
      setIsProcessing(false);
      onProcessingEnd?.(); // ✅ Notify parent that processing ended
    }
  };

  const updateSettings = (key, value) => {
    setSettings(prev => ({
      ...prev,
      [key]: parseInt(value) || 0
    }));
  };

  return (
    <div className="card p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-gray-800">Upload PDF Files</h2>
        <button
          onClick={() => setShowSettings(!showSettings)}
          className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
          title="Processing Settings"
        >
          <Settings className="w-5 h-5" />
        </button>
      </div>

      {/* Settings Panel */}
      {showSettings && (
        <div className="mb-6 p-4 bg-gray-50 rounded-lg border">
          <h3 className="text-sm font-medium text-gray-700 mb-3">Processing Settings</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">
                Chunk Size (words)
              </label>
              <input
                type="number"
                value={settings.chunkSize}
                onChange={(e) => updateSettings('chunkSize', e.target.value)}
                className="input-field text-sm"
                min="100"
                max="2000"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">
                Chunk Overlap (words)
              </label>
              <input
                type="number"
                value={settings.chunkOverlap}
                onChange={(e) => updateSettings('chunkOverlap', e.target.value)}
                className="input-field text-sm"
                min="0"
                max="500"
              />
            </div>
          </div>
        </div>
      )}

      {/* File Drop Zone */}
      <div
        className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          dragActive 
            ? 'border-primary-500 bg-primary-50' 
            : 'border-gray-300 hover:border-gray-400'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          type="file"
          multiple
          accept=".pdf,application/pdf"
          onChange={handleFileSelect}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          disabled={isProcessing}
        />
        
        <div className="space-y-4">
          <Upload className={`mx-auto w-12 h-12 ${dragActive ? 'text-primary-500' : 'text-gray-400'}`} />
          <div>
            <p className={`text-lg font-medium ${dragActive ? 'text-primary-700' : 'text-gray-700'}`}>
              {dragActive ? 'Drop files here' : 'Drag & drop PDF files here'}
            </p>
            <p className="text-sm text-gray-500 mt-1">
              or click to select files • Maximum 100MB per file
            </p>
          </div>
        </div>
      </div>

      {/* Selected Files List */}
      {selectedFiles.length > 0 && (
        <div className="mt-6">
          <h3 className="text-sm font-medium text-gray-700 mb-3">
            Selected Files ({selectedFiles.length})
          </h3>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {selectedFiles.map((file, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <FileText className="w-5 h-5 text-red-500" />
                  <div>
                    <p className="text-sm font-medium text-gray-800 truncate max-w-xs">
                      {file.name}
                    </p>
                    <p className="text-xs text-gray-500">
                      {formatFileSize(file.size)}
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => removeFile(index)}
                  className="p-1 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded transition-colors"
                  disabled={isProcessing}
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Process Button */}
      <div className="mt-6">
        <button
          onClick={processFiles}
          disabled={selectedFiles.length === 0 || isProcessing}
          className={`w-full py-3 px-4 rounded-lg font-medium transition-all duration-200 ${
            selectedFiles.length === 0 || isProcessing
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'btn-primary hover:shadow-lg'
          }`}
        >
          {isProcessing ? (
            <div className="flex items-center justify-center space-x-2">
              <LoadingSpinner size="small" />
              <span>Processing {selectedFiles.length} file{selectedFiles.length > 1 ? 's' : ''}...</span>
            </div>
          ) : (
            `Process ${selectedFiles.length} File${selectedFiles.length === 1 ? '' : 's'}`
          )}
        </button>
      </div>
    </div>
  );
};

export default FileUpload;
