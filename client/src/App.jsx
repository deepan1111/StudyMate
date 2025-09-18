

// import React, { useState, useEffect, useCallback } from 'react';
// import { BookOpen, Upload, MessageSquare, BarChart3, AlertCircle, CheckCircle, X } from 'lucide-react';
// import FileUpload from './Component/FileUpload';
// import TextProcessor from './Component/TextProcessor';
// import ProcessingResults from './Component/ProcessingResults';
// import StudyMateChat from './Component/StudyMAteChat';
// import { pdfService } from './services/api';
// import './App.css';
// // Import the debug service
// import ErrorBoundary from './Component/components/ErrorBoundary'
// import { debugService } from './services/api';

// function App() {
//   const [activeTab, setActiveTab] = useState('upload');
//   const [processingResults, setProcessingResults] = useState(null);
//   const [uploadedDocuments, setUploadedDocuments] = useState([]);
//   const [notification, setNotification] = useState(null);
//   const [apiStatus, setApiStatus] = useState('checking');

//   useEffect(() => {
//     checkApiHealth();
//   }, []);

//   const checkApiHealth = async () => {
//     try {
//       await pdfService.healthCheck();
//       setApiStatus('connected');
//     } catch (error) {
//       setApiStatus('disconnected');
//       showNotification('Unable to connect to backend API. Please ensure the server is running.', 'error');
//     }
//   };

  
// // Add connection testing function
// const testBackendConnection = async () => {
//   try {
//     const results = await debugService.runConnectionTests();
//     console.log('Connection test results:', results);
    
//     // Set connection status state
//     setConnectionStatus(results.health?.status === 'healthy' ? 'connected' : 'disconnected');
    
//     if (results.health?.status === 'healthy') {
//       console.log('✅ Backend connection successful');
//     } else {
//       console.error('❌ Backend connection issues:', results);
//     }
//   } catch (error) {
//     console.error('❌ Cannot connect to backend:', error);
//     setConnectionStatus('error');
//   }
// };

// // Call this in your App's useEffect
// useEffect(() => {
//   testBackendConnection();
  
//   // Optional: Test connection periodically
//   const interval = setInterval(testBackendConnection, 30000); // Every 30 seconds
//   return () => clearInterval(interval);
// }, []);

//   const showNotification = useCallback((message, type = 'info') => {
//     setNotification({ message, type, id: Date.now() });
//     setTimeout(() => setNotification(null), 5000);
//   }, []);

//   const handleProcessingComplete = useCallback((results) => {
//     setProcessingResults(results);
//     setActiveTab('results');
//     showNotification('Processing completed successfully!', 'success');
//   }, [showNotification]);

//   // ✅ Wrapped in useCallback to avoid re-creating on every render
//   const handleDocumentReady = useCallback((documents) => {
//     setUploadedDocuments(documents);
//     showNotification(`${documents.length} document(s) ready for Q&A!`, 'success');
//   }, [showNotification]);

//   const handleError = useCallback((error) => {
//     showNotification(error, 'error');
//   }, [showNotification]);

//   const dismissNotification = useCallback(() => {
//     setNotification(null);
//   }, []);

//   const tabs = [
//     { id: 'upload', label: 'Upload Documents', icon: Upload, description: 'Upload and process PDF files' },
//     { id: 'text', label: 'Process Text', icon: BarChart3, description: 'Process raw text input' },
//     { id: 'chat', label: 'StudyMate Q&A', icon: MessageSquare, description: 'Ask questions about your documents', disabled: uploadedDocuments.length === 0 },
//     { id: 'results', label: 'Results', icon: BarChart3, description: 'View processing results', disabled: !processingResults }
//   ];

//   return (
//     <ErrorBoundary>
//     <div className="min-h-screen bg-gray-50">
//       {/* Header */}
//       <header className="bg-white shadow-sm border-b border-gray-200">
//         <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
//           <div className="flex items-center justify-between h-16">
//             <div className="flex items-center space-x-3">
//               <div className="p-2 bg-primary-500 rounded-lg">
//                 <BookOpen className="w-8 h-8 text-white" />
//               </div>
//               <div>
//                 <h1 className="text-2xl font-bold text-gray-900">StudyMate</h1>
//                 <p className="text-sm text-gray-600">AI-Powered Academic Assistant</p>
//               </div>
//             </div>

//             {/* API Status */}
//             <div className="flex items-center space-x-2">
//               <div
//                 className={`w-3 h-3 rounded-full ${
//                   apiStatus === 'connected'
//                     ? 'bg-green-500'
//                     : apiStatus === 'disconnected'
//                     ? 'bg-red-500'
//                     : 'bg-yellow-500'
//                 }`}
//               ></div>
//               <span
//                 className={`text-sm ${
//                   apiStatus === 'connected'
//                     ? 'text-green-600'
//                     : apiStatus === 'disconnected'
//                     ? 'text-red-600'
//                     : 'text-yellow-600'
//                 }`}
//               >
//                 {apiStatus === 'connected'
//                   ? 'Connected'
//                   : apiStatus === 'disconnected'
//                   ? 'Disconnected'
//                   : 'Connecting...'}
//               </span>
//             </div>
//           </div>
//         </div>
//       </header>

//       {/* Notification */}
//       {notification && (
//         <div
//           className={`fixed top-20 right-4 max-w-sm z-50 p-4 rounded-lg shadow-lg ${
//             notification.type === 'success'
//               ? 'bg-green-50 text-green-800 border border-green-200'
//               : notification.type === 'error'
//               ? 'bg-red-50 text-red-800 border border-red-200'
//               : 'bg-blue-50 text-blue-800 border border-blue-200'
//           }`}
//         >
//           <div className="flex items-start space-x-3">
//             <div className="flex-shrink-0">
//               {notification.type === 'success' ? (
//                 <CheckCircle className="w-5 h-5 text-green-500" />
//               ) : notification.type === 'error' ? (
//                 <AlertCircle className="w-5 h-5 text-red-500" />
//               ) : (
//                 <AlertCircle className="w-5 h-5 text-blue-500" />
//               )}
//             </div>
//             <div className="flex-1">
//               <p className="text-sm font-medium">{notification.message}</p>
//             </div>
//             <button
//               onClick={dismissNotification}
//               className="flex-shrink-0 text-gray-400 hover:text-gray-600"
//             >
//               <X className="w-4 h-4" />
//             </button>
//           </div>
//         </div>
//       )}

//       {/* Navigation Tabs */}
//       <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6">
//         <div className="border-b border-gray-200">
//           <nav className="flex space-x-8">
//             {tabs.map((tab) => {
//               const Icon = tab.icon;
//               return (
//                 <button
//                   key={tab.id}
//                   onClick={() => !tab.disabled && setActiveTab(tab.id)}
//                   className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
//                     activeTab === tab.id
//                       ? 'border-primary-500 text-primary-600'
//                       : tab.disabled
//                       ? 'border-transparent text-gray-400 cursor-not-allowed'
//                       : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
//                   }`}
//                   disabled={tab.disabled}
//                   title={
//                     tab.disabled
//                       ? `${tab.description} (requires uploaded documents)`
//                       : tab.description
//                   }
//                 >
//                   <Icon className="w-5 h-5" />
//                   <span>{tab.label}</span>
//                   {tab.id === 'chat' && uploadedDocuments.length > 0 && (
//                     <span className="bg-primary-100 text-primary-800 text-xs px-2 py-0.5 rounded-full">
//                       {uploadedDocuments.length}
//                     </span>
//                   )}
//                 </button>
//               );
//             })}
//           </nav>
//         </div>
//       </div>

//       {/* Main Content */}
//       <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
//         {activeTab === 'upload' && (
//           <div className="space-y-6">
//             <div className="text-center">
//               <h2 className="text-3xl font-bold text-gray-900 mb-4">Upload Your Study Materials</h2>
//               <p className="text-lg text-gray-600 max-w-2xl mx-auto">
//                 Upload PDF documents like textbooks, lecture notes, or research papers. StudyMate will
//                 process them and enable conversational Q&A.
//               </p>
//             </div>
//             <FileUpload onProcessingComplete={handleProcessingComplete} onError={handleError} />
//           </div>
//         )}

//         {activeTab === 'text' && (
//           <div className="space-y-6">
//             <div className="text-center">
//               <h2 className="text-3xl font-bold text-gray-900 mb-4">Process Raw Text</h2>
//               <p className="text-lg text-gray-600 max-w-2xl mx-auto">
//                 Paste any text content to analyze and chunk it for better understanding.
//               </p>
//             </div>
//             <TextProcessor onProcessingComplete={handleProcessingComplete} onError={handleError} />
//           </div>
//         )}

//         {activeTab === 'chat' && (
//           <div className="space-y-6">
//             <div className="text-center">
//               <h2 className="text-3xl font-bold text-gray-900 mb-4">Chat with Your Documents</h2>
//               <p className="text-lg text-gray-600 max-w-2xl mx-auto">
//                 Ask natural language questions about your uploaded documents. StudyMate will provide
//                 contextual answers with source references.
//               </p>
//               {uploadedDocuments.length > 0 && (
//                 <div className="mt-4">
//                   <div className="inline-flex items-center space-x-2 bg-green-100 text-green-800 px-4 py-2 rounded-full text-sm">
//                     <CheckCircle className="w-4 h-4" />
//                     <span>{uploadedDocuments.length} document(s) ready for Q&A</span>
//                   </div>
//                 </div>
//               )}
//             </div>
//             <div className="max-w-4xl mx-auto">
//               <StudyMateChat uploadedDocuments={uploadedDocuments} />
//             </div>
//           </div>
//         )}

//         {activeTab === 'results' && (
//           <div className="space-y-6">
//             <div className="text-center">
//               <h2 className="text-3xl font-bold text-gray-900 mb-4">Processing Results</h2>
//               <p className="text-lg text-gray-600 max-w-2xl mx-auto">
//                 Detailed analysis of your processed documents with text chunks and statistics.
//               </p>
//             </div>
//             <ProcessingResults results={processingResults} onDocumentReady={handleDocumentReady} />
//           </div>
//         )}
//       </main>

//       {/* Footer */}
//       <footer className="bg-white border-t border-gray-200 mt-16">
//         <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
//           <div className="text-center">
//             <p className="text-gray-600">
//               StudyMate - AI-Powered Academic Assistant | Built with React, FastAPI, and PyMuPDF
//             </p>
//             <div className="mt-4 flex justify-center space-x-6 text-sm text-gray-500">
//               <span>✨ Conversational Q&A</span>
//               <span>🔍 Semantic Search</span>
//               <span>📚 Multiple PDF Support</span>
//               <span>⚡ Fast Processing</span>
//             </div>
//           </div>
//         </div>
//       </footer>
//     </div>
//     </ErrorBoundary>
//   );
// }

// export default App;
import React, { useState, useEffect, useCallback } from 'react';
import { BookOpen, Upload, MessageSquare, BarChart3, AlertCircle, CheckCircle, X } from 'lucide-react';
import FileUpload from './Component/FileUpload';
import TextProcessor from './Component/TextProcessor';
import ProcessingResults from './Component/ProcessingResults';
import StudyMateChat from './Component/StudyMAteChat';
import { pdfService } from './services/api';
import './App.css';
import ErrorBoundary from './Component/components/ErrorBoundary';
import { debugService } from './services/api';

function App() {
  const [activeTab, setActiveTab] = useState('upload');
  const [processingResults, setProcessingResults] = useState(null);
  const [uploadedDocuments, setUploadedDocuments] = useState([]);
  const [notification, setNotification] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');
  const [connectionStatus, setConnectionStatus] = useState('checking');
  const [backendError, setBackendError] = useState(null);
  
  // ✅ ADD: Missing isProcessing state
  const [isProcessing, setIsProcessing] = useState(false);

  // ✅ FIXED: Combined connection testing function
  const checkApiHealth = async () => {
    try {
      setApiStatus('checking');
      setConnectionStatus('checking');
      setBackendError(null);
      
      // Use the simple health check for UI status
      await pdfService.healthCheck();
      setApiStatus('connected');
      
      // Use debug service for detailed testing (optional)
      const results = await debugService.runConnectionTests();
      console.log('Connection test results:', results);
      
      if (results.health && !results.health.error) {
        console.log('✅ Backend connection successful');
        setConnectionStatus('connected');
      } else {
        console.warn('⚠️ Backend connection issues:', results);
        setConnectionStatus('disconnected');
        setBackendError(results.health?.error || 'Backend health check failed');
      }
      
    } catch (error) {
      console.error('❌ Cannot connect to backend:', error);
      setApiStatus('disconnected');
      setConnectionStatus('error');
      setBackendError(error.message);
      showNotification('Unable to connect to backend API. Please ensure the server is running.', 'error');
    }
  };

  // ✅ FIXED: Single useEffect for connection testing with proper dependency
  useEffect(() => {
    checkApiHealth();
    
    // FIXED: Only check periodically when not uploading
    const interval = setInterval(() => {
      // Don't check health during active operations
      if (!isProcessing) {
        checkApiHealth();
      } else {
        console.log('Skipping health check - processing in progress');
      }
    }, 60000); // Reduced frequency to 60 seconds
    
    return () => clearInterval(interval);
  }, [isProcessing]); // ✅ Now properly depends on isProcessing

  const showNotification = useCallback((message, type = 'info') => {
    setNotification({ message, type, id: Date.now() });
    setTimeout(() => setNotification(null), 5000);
  }, []);

  // ✅ ADD: Processing state handlers
  const handleProcessingStart = useCallback(() => {
    console.log('Processing started - pausing health checks');
    setIsProcessing(true);
  }, []);

  const handleProcessingEnd = useCallback(() => {
    console.log('Processing ended - resuming health checks');
    setIsProcessing(false);
    // Optionally trigger a health check after processing
    setTimeout(checkApiHealth, 1000);
  }, []);

  const handleProcessingComplete = useCallback((results) => {
    setProcessingResults(results);
    setActiveTab('results');
    showNotification('Processing completed successfully!', 'success');
    handleProcessingEnd(); // ✅ End processing state
  }, [showNotification, handleProcessingEnd]);

  const handleDocumentReady = useCallback((documents) => {
    setUploadedDocuments(documents);
    showNotification(`${documents.length} document(s) ready for Q&A!`, 'success');
  }, [showNotification]);

  const handleError = useCallback((error) => {
    showNotification(error, 'error');
    handleProcessingEnd(); // ✅ End processing state on error
  }, [showNotification, handleProcessingEnd]);

  const dismissNotification = useCallback(() => {
    setNotification(null);
  }, []);

  const tabs = [
    { id: 'upload', label: 'Upload Documents', icon: Upload, description: 'Upload and process PDF files' },
    { id: 'text', label: 'Process Text', icon: BarChart3, description: 'Process raw text input' },
    { id: 'chat', label: 'StudyMate Q&A', icon: MessageSquare, description: 'Ask questions about your documents', disabled: uploadedDocuments.length === 0 },
    { id: 'results', label: 'Results', icon: BarChart3, description: 'View processing results', disabled: !processingResults }
  ];

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-gray-50">
        {/* ✅ Enhanced: Connection Status Bar with processing indicator */}
        {(connectionStatus !== 'connected' || isProcessing) && (
          <div className={`w-full p-2 text-center text-sm border-b ${
            isProcessing 
              ? 'bg-blue-50 text-blue-800 border-blue-200' 
              : connectionStatus === 'checking' 
                ? 'bg-yellow-50 text-yellow-800 border-yellow-200' 
                : 'bg-red-50 text-red-800 border-red-200'
          }`}>
            {isProcessing && 'Processing document - please wait...'}
            {!isProcessing && connectionStatus === 'checking' && 'Connecting to StudyMate backend...'}
            {!isProcessing && connectionStatus === 'disconnected' && `Backend disconnected: ${backendError}`}
            {!isProcessing && connectionStatus === 'error' && `Connection error: ${backendError}`}
            {!isProcessing && connectionStatus !== 'checking' && (
              <button 
                onClick={checkApiHealth}
                className="ml-2 underline hover:no-underline font-medium"
              >
                Retry Connection
              </button>
            )}
          </div>
        )}

        {/* Header */}
        <header className="bg-white shadow-sm border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-primary-500 rounded-lg">
                  <BookOpen className="w-8 h-8 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-gray-900">StudyMate</h1>
                  <p className="text-sm text-gray-600">AI-Powered Academic Assistant</p>
                </div>
              </div>

              {/* ✅ ENHANCED: API Status with processing indicator */}
              <div className="flex items-center space-x-3">
                <div className="flex items-center space-x-2">
                  <div
                    className={`w-3 h-3 rounded-full ${
                      isProcessing
                        ? 'bg-blue-500 animate-pulse'
                        : apiStatus === 'connected'
                        ? 'bg-green-500'
                        : apiStatus === 'disconnected'
                        ? 'bg-red-500'
                        : 'bg-yellow-500 animate-pulse'
                    }`}
                  ></div>
                  <span
                    className={`text-sm font-medium ${
                      isProcessing
                        ? 'text-blue-600'
                        : apiStatus === 'connected'
                        ? 'text-green-600'
                        : apiStatus === 'disconnected'
                        ? 'text-red-600'
                        : 'text-yellow-600'
                    }`}
                  >
                    {isProcessing
                      ? 'Processing...'
                      : apiStatus === 'connected'
                      ? 'Connected'
                      : apiStatus === 'disconnected'
                      ? 'Disconnected'
                      : 'Connecting...'}
                  </span>
                </div>
                
                {/* Show document count when connected */}
                {apiStatus === 'connected' && uploadedDocuments.length > 0 && (
                  <div className="text-sm text-gray-600">
                    {uploadedDocuments.length} docs ready
                  </div>
                )}
              </div>
            </div>
          </div>
        </header>

        {/* Rest of your component remains the same */}
        {/* Notification */}
        {notification && (
          <div
            className={`fixed top-20 right-4 max-w-sm z-50 p-4 rounded-lg shadow-lg transition-all duration-300 ${
              notification.type === 'success'
                ? 'bg-green-50 text-green-800 border border-green-200'
                : notification.type === 'error'
                ? 'bg-red-50 text-red-800 border border-red-200'
                : 'bg-blue-50 text-blue-800 border border-blue-200'
            }`}
          >
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0">
                {notification.type === 'success' ? (
                  <CheckCircle className="w-5 h-5 text-green-500" />
                ) : notification.type === 'error' ? (
                  <AlertCircle className="w-5 h-5 text-red-500" />
                ) : (
                  <AlertCircle className="w-5 h-5 text-blue-500" />
                )}
              </div>
              <div className="flex-1">
                <p className="text-sm font-medium">{notification.message}</p>
              </div>
              <button
                onClick={dismissNotification}
                className="flex-shrink-0 text-gray-400 hover:text-gray-600 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}

        {/* Navigation Tabs */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6">
          <div className="border-b border-gray-200">
            <nav className="flex space-x-8">
              {tabs.map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => !tab.disabled && setActiveTab(tab.id)}
                    className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                      activeTab === tab.id
                        ? 'border-primary-500 text-primary-600'
                        : tab.disabled
                        ? 'border-transparent text-gray-400 cursor-not-allowed'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }`}
                    disabled={tab.disabled}
                    title={
                      tab.disabled
                        ? `${tab.description} (requires uploaded documents)`
                        : tab.description
                    }
                  >
                    <Icon className="w-5 h-5" />
                    <span>{tab.label}</span>
                    {tab.id === 'chat' && uploadedDocuments.length > 0 && (
                      <span className="bg-primary-100 text-primary-800 text-xs px-2 py-0.5 rounded-full">
                        {uploadedDocuments.length}
                      </span>
                    )}
                  </button>
                );
              })}
            </nav>
          </div>
        </div>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Show loading state when checking connection */}
          {apiStatus === 'checking' && (
            <div className="flex items-center justify-center py-12">
              <div className="text-center">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500 mx-auto mb-4"></div>
                <p className="text-gray-600">Connecting to StudyMate backend...</p>
              </div>
            </div>
          )}

          {/* Show disconnected state */}
          {apiStatus === 'disconnected' && (
            <div className="text-center py-12">
              <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Backend Connection Lost</h3>
              <p className="text-gray-600 mb-4">
                Cannot connect to the StudyMate backend server. Please ensure it's running on port 8000.
              </p>
              <button
                onClick={checkApiHealth}
                className="px-4 py-2 bg-primary-500 text-white rounded hover:bg-primary-600 transition-colors"
              >
                Retry Connection
              </button>
            </div>
          )}

          {/* Main app content (only show when connected) */}
          {apiStatus === 'connected' && (
            <>
              {activeTab === 'upload' && (
                <div className="space-y-6">
                  <div className="text-center">
                    <h2 className="text-3xl font-bold text-gray-900 mb-4">Upload Your Study Materials</h2>
                    <p className="text-lg text-gray-600 max-w-2xl mx-auto">
                      Upload PDF documents like textbooks, lecture notes, or research papers. StudyMate will
                      process them and enable conversational Q&A.
                    </p>
                  </div>
                  {/* ✅ Pass processing state handlers to FileUpload */}
                  <FileUpload 
                    onProcessingComplete={handleProcessingComplete} 
                    onError={handleError}
                    onProcessingStart={handleProcessingStart}
                    onProcessingEnd={handleProcessingEnd}
                  />
                </div>
              )}

              {activeTab === 'text' && (
                <div className="space-y-6">
                  <div className="text-center">
                    <h2 className="text-3xl font-bold text-gray-900 mb-4">Process Raw Text</h2>
                    <p className="text-lg text-gray-600 max-w-2xl mx-auto">
                      Paste any text content to analyze and chunk it for better understanding.
                    </p>
                  </div>
                  <TextProcessor onProcessingComplete={handleProcessingComplete} onError={handleError} />
                </div>
              )}

              {activeTab === 'chat' && (
                <div className="space-y-6">
                  <div className="text-center">
                    <h2 className="text-3xl font-bold text-gray-900 mb-4">Chat with Your Documents</h2>
                    <p className="text-lg text-gray-600 max-w-2xl mx-auto">
                      Ask natural language questions about your uploaded documents. StudyMate will provide
                      contextual answers with source references.
                    </p>
                    {uploadedDocuments.length > 0 && (
                      <div className="mt-4">
                        <div className="inline-flex items-center space-x-2 bg-green-100 text-green-800 px-4 py-2 rounded-full text-sm">
                          <CheckCircle className="w-4 h-4" />
                          <span>{uploadedDocuments.length} document(s) ready for Q&A</span>
                        </div>
                      </div>
                    )}
                  </div>
                  <div className="max-w-4xl mx-auto">
                    <StudyMateChat uploadedDocuments={uploadedDocuments} />
                  </div>
                </div>
              )}

              {activeTab === 'results' && (
                <div className="space-y-6">
                  <div className="text-center">
                    <h2 className="text-3xl font-bold text-gray-900 mb-4">Processing Results</h2>
                    <p className="text-lg text-gray-600 max-w-2xl mx-auto">
                      Detailed analysis of your processed documents with text chunks and statistics.
                    </p>
                  </div>
                  <ProcessingResults results={processingResults} onDocumentReady={handleDocumentReady} />
                </div>
              )}
            </>
          )}
        </main>

        {/* Footer */}
        <footer className="bg-white border-t border-gray-200 mt-16">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <div className="text-center">
              <p className="text-gray-600">
                StudyMate - AI-Powered Academic Assistant | Built with React, FastAPI, and PyMuPDF
              </p>
              <div className="mt-4 flex justify-center space-x-6 text-sm text-gray-500">
                <span>✨ Conversational Q&A</span>
                <span>🔍 Semantic Search</span>
                <span>📚 Multiple PDF Support</span>
                <span>⚡ Fast Processing</span>
              </div>
              <div className="mt-2 text-xs text-gray-400">
                Backend: {apiStatus === 'connected' ? '✅ Connected' : '❌ Disconnected'} | 
                Documents: {uploadedDocuments.length} |
                {isProcessing && ' 🔄 Processing...'}
              </div>
            </div>
          </div>
        </footer>
      </div>
    </ErrorBoundary>
  );
}

export default App;