
import React, { useState } from 'react';
import { Type, Settings } from 'lucide-react';
import { pdfService } from '../services/api';
import LoadingSpinner from './LoadingSpinner';

const TextProcessor = ({ onProcessingComplete, onError }) => {
  const [text, setText] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [settings, setSettings] = useState({
    chunkSize: 1000,
    chunkOverlap: 200
  });
  const [showSettings, setShowSettings] = useState(false);

 const processText = async () => {
  if (!text.trim()) {
    onError('Please enter some text to process.');
    return;
  }

  setIsProcessing(true);
  try {
    // Use updated API method signature
    const result = await pdfService.processText(text.trim());
    
    // Handle the response properly
    if (result.success !== false) {
      onProcessingComplete(result);
      setText(''); // Clear text after successful processing
    } else {
      onError(result.message || result.error || 'Text processing failed.');
    }
  } catch (error) {
    console.error('Text processing error:', error);
    onError(error.response?.data?.detail || 'Text processing failed. Please try again.');
  } finally {
    setIsProcessing(false);
  }
};

  const updateSettings = (key, value) => {
    setSettings(prev => ({
      ...prev,
      [key]: parseInt(value) || 0
    }));
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Tab') {
      e.preventDefault();
      const start = e.target.selectionStart;
      const end = e.target.selectionEnd;
      const value = e.target.value;
      setText(value.substring(0, start) + '\t' + value.substring(end));
      setTimeout(() => {
        e.target.selectionStart = e.target.selectionEnd = start + 1;
      }, 0);
    }
  };

  return (
    <div className="card p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-gray-800 flex items-center space-x-2">
          <Type className="w-6 h-6 text-primary-500" />
          <span>Process Text</span>
        </h2>
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

      {/* Text Area */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Enter text to process
        </label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Paste your text here to chunk and analyze..."
          className="w-full h-48 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none resize-none"
          disabled={isProcessing}
        />
        <div className="flex justify-between mt-2 text-xs text-gray-500">
          <span>{text.length} characters</span>
          <span>{text.trim().split(/\s+/).length} words</span>
        </div>
      </div>

      {/* Process Button */}
      <button
        onClick={processText}
        disabled={!text.trim() || isProcessing}
        className={`w-full py-3 px-4 rounded-lg font-medium transition-all duration-200 ${
          !text.trim() || isProcessing
            ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
            : 'btn-primary hover:shadow-lg'
        }`}
      >
        {isProcessing ? (
          <div className="flex items-center justify-center space-x-2">
            <LoadingSpinner size="small" />
            <span>Processing Text...</span>
          </div>
        ) : (
          'Process Text'
        )}
      </button>
    </div>
  );
};

export default TextProcessor;