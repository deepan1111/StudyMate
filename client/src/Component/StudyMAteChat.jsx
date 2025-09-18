// import React, { useState, useRef, useEffect } from 'react';
// import { Send, MessageSquare, Book, User, Bot, Clock, Copy, Check } from 'lucide-react';
// import { studyMateService } from '../services/api';
// import LoadingSpinner from './LoadingSpinner';

// const StudyMateChat = ({ uploadedDocuments = [] }) => {
//   const [messages, setMessages] = useState([]);
//   const [currentQuestion, setCurrentQuestion] = useState('');
//   const [isAsking, setIsAsking] = useState(false);
//   const [sessionId, setSessionId] = useState(null);
//   const [copiedMessageId, setCopiedMessageId] = useState(null);
//   const messagesEndRef = useRef(null);

//   useEffect(() => {
//     // Create a new session when component mounts
//     createNewSession();
//   }, []);

//   useEffect(() => {
//     // Scroll to bottom when new messages are added
//     scrollToBottom();
//   }, [messages]);

//   const scrollToBottom = () => {
//     messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
//   };

//   const createNewSession = async () => {
//     try {
//       const response = await studyMateService.createSession();
//       setSessionId(response.session_id);
//       setMessages([{
//         id: Date.now(),
//         type: 'bot',
//         content: "Hello! I'm StudyMate, your AI academic assistant. Upload your PDFs and ask me any questions about your study materials!",
//         timestamp: new Date(),
//         sources: []
//       }]);
//     } catch (error) {
//       console.error('Failed to create session:', error);
//     }
//   };

//   const askQuestion = async () => {
//     if (!currentQuestion.trim() || isAsking) return;

//     if (uploadedDocuments.length === 0) {
//       setMessages(prev => [...prev, {
//         id: Date.now(),
//         type: 'bot',
//         content: "Please upload some PDF documents first so I can answer questions about them!",
//         timestamp: new Date(),
//         sources: []
//       }]);
//       return;
//     }

//     const userMessage = {
//       id: Date.now(),
//       type: 'user',
//       content: currentQuestion.trim(),
//       timestamp: new Date()
//     };

//     setMessages(prev => [...prev, userMessage]);
//     setIsAsking(true);

//     // Add typing indicator
//     const typingMessage = {
//       id: Date.now() + 1,
//       type: 'bot',
//       content: '',
//       isTyping: true,
//       timestamp: new Date()
//     };
//     setMessages(prev => [...prev, typingMessage]);

//     try {
//       const response = await studyMateService.askQuestion(currentQuestion.trim(), sessionId);
      
//       // Remove typing indicator and add actual response
//       setMessages(prev => prev.filter(msg => !msg.isTyping).concat({
//         id: Date.now() + 2,
//         type: 'bot',
//         content: response.answer,
//         timestamp: new Date(),
//         sources: response.sources || [],
//         confidence: response.confidence,
//         processingTime: response.processing_time
//       }));
      
//     } catch (error) {
//       console.error('Question asking failed:', error);
//       setMessages(prev => prev.filter(msg => !msg.isTyping).concat({
//         id: Date.now() + 2,
//         type: 'bot',
//         content: "I'm sorry, I encountered an error while processing your question. Please try again.",
//         timestamp: new Date(),
//         sources: [],
//         isError: true
//       }));
//     } finally {
//       setIsAsking(false);
//     }

//     setCurrentQuestion('');
//   };

//   const copyToClipboard = async (text, messageId) => {
//     try {
//       await navigator.clipboard.writeText(text);
//       setCopiedMessageId(messageId);
//       setTimeout(() => setCopiedMessageId(null), 2000);
//     } catch (error) {
//       console.error('Failed to copy:', error);
//     }
//   };

//   const handleKeyPress = (e) => {
//     if (e.key === 'Enter' && !e.shiftKey) {
//       e.preventDefault();
//       askQuestion();
//     }
//   };

//   const formatTimestamp = (timestamp) => {
//     return new Date(timestamp).toLocaleTimeString('en-US', {
//       hour: '2-digit',
//       minute: '2-digit'
//     });
//   };

//   const suggestedQuestions = [
//     "What are the main topics covered in this document?",
//     "Can you summarize the key points?",
//     "What are the important definitions mentioned?",
//     "Explain the methodology used in this research.",
//     "What are the conclusions drawn in this study?"
//   ];

//   return (
//     <div className="card p-0 h-full flex flex-col max-h-[600px]">
//       {/* Header */}
//       <div className="p-4 border-b border-gray-200 bg-gradient-to-r from-primary-500 to-primary-600 text-white rounded-t-lg">
//         <div className="flex items-center space-x-3">
//           <div className="p-2 bg-white/20 rounded-lg">
//             <MessageSquare className="w-5 h-5" />
//           </div>
//           <div>
//             <h2 className="text-lg font-semibold">StudyMate Assistant</h2>
//             <p className="text-primary-100 text-sm">
//               {uploadedDocuments.length > 0 
//                 ? `Ready to answer questions about ${uploadedDocuments.length} document(s)`
//                 : 'Upload documents to start chatting'
//               }
//             </p>
//           </div>
//         </div>
//       </div>

//       {/* Messages Container */}
//       <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0">
//         {messages.map((message) => (
//           <div key={message.id} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
//             <div className={`flex max-w-[80%] ${message.type === 'user' ? 'flex-row-reverse' : 'flex-row'} space-x-3`}>
//               {/* Avatar */}
//               <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
//                 message.type === 'user' 
//                   ? 'bg-primary-500 text-white ml-3' 
//                   : 'bg-gray-200 text-gray-600 mr-3'
//               }`}>
//                 {message.type === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
//               </div>

//               {/* Message Content */}
//               <div className={`flex-1 ${message.type === 'user' ? 'mr-3' : 'ml-3'}`}>
//                 <div className={`p-3 rounded-lg ${
//                   message.type === 'user' 
//                     ? 'bg-primary-500 text-white' 
//                     : message.isError 
//                       ? 'bg-red-50 text-red-800 border border-red-200'
//                       : 'bg-gray-100 text-gray-800'
//                 }`}>
//                   {message.isTyping ? (
//                     <div className="flex items-center space-x-2">
//                       <LoadingSpinner size="small" />
//                       <span className="text-sm">StudyMate is thinking...</span>
//                     </div>
//                   ) : (
//                     <>
//                       <p className="whitespace-pre-wrap">{message.content}</p>
                      
//                       {/* Sources */}
//                       {message.sources && message.sources.length > 0 && (
//                         <div className="mt-3 pt-3 border-t border-gray-300">
//                           <div className="flex items-center space-x-1 mb-2">
//                             <Book className="w-4 h-4" />
//                             <span className="text-xs font-medium">Sources:</span>
//                           </div>
//                           <div className="space-y-1">
//                             {message.sources.map((source, index) => (
//                               <div key={index} className="text-xs bg-white/50 p-2 rounded border">
//                                 <div className="font-medium">{source.document}</div>
//                                 <div className="text-gray-600 mt-1">{source.excerpt}...</div>
//                               </div>
//                             ))}
//                           </div>
//                         </div>
//                       )}
                      
//                       {/* Metadata */}
//                       {(message.confidence || message.processingTime) && (
//                         <div className="mt-2 pt-2 border-t border-gray-300 text-xs text-gray-600">
//                           {message.confidence && (
//                             <span>Confidence: {Math.round(message.confidence * 100)}% • </span>
//                           )}
//                           {message.processingTime && (
//                             <span>Response time: {message.processingTime.toFixed(2)}s</span>
//                           )}
//                         </div>
//                       )}
//                     </>
//                   )}
//                 </div>
                
//                 {/* Message Actions */}
//                 {!message.isTyping && message.type === 'bot' && (
//                   <div className="flex items-center space-x-2 mt-2">
//                     <button
//                       onClick={() => copyToClipboard(message.content, message.id)}
//                       className="p-1 text-gray-400 hover:text-gray-600 rounded transition-colors"
//                       title="Copy message"
//                     >
//                       {copiedMessageId === message.id ? (
//                         <Check className="w-4 h-4 text-green-500" />
//                       ) : (
//                         <Copy className="w-4 h-4" />
//                       )}
//                     </button>
//                     <span className="text-xs text-gray-400 flex items-center">
//                       <Clock className="w-3 h-3 mr-1" />
//                       {formatTimestamp(message.timestamp)}
//                     </span>
//                   </div>
//                 )}
//               </div>
//             </div>
//           </div>
//         ))}
//         <div ref={messagesEndRef} />
//       </div>

//       {/* Suggested Questions */}
//       {messages.length <= 1 && uploadedDocuments.length > 0 && (
//         <div className="p-4 border-t border-gray-200 bg-gray-50">
//           <p className="text-sm text-gray-600 mb-2">Try asking:</p>
//           <div className="flex flex-wrap gap-2">
//             {suggestedQuestions.slice(0, 3).map((question, index) => (
//               <button
//                 key={index}
//                 onClick={() => setCurrentQuestion(question)}
//                 className="text-xs bg-white border border-gray-300 hover:border-primary-500 hover:text-primary-600 px-3 py-1 rounded-full transition-colors"
//               >
//                 {question}
//               </button>
//             ))}
//           </div>
//         </div>
//       )}

//       {/* Input Area */}
//       <div className="p-4 border-t border-gray-200">
//         <div className="flex space-x-3">
//           <div className="flex-1">
//             <textarea
//               value={currentQuestion}
//               onChange={(e) => setCurrentQuestion(e.target.value)}
//               onKeyPress={handleKeyPress}
//               placeholder={uploadedDocuments.length > 0 ? "Ask a question about your documents..." : "Upload documents first to start asking questions..."}
//               className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none resize-none"
//               rows={2}
//               disabled={isAsking || uploadedDocuments.length === 0}
//             />
//           </div>
//           <button
//             onClick={askQuestion}
//             disabled={!currentQuestion.trim() || isAsking || uploadedDocuments.length === 0}
//             className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 flex items-center space-x-2 ${
//               !currentQuestion.trim() || isAsking || uploadedDocuments.length === 0
//                 ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
//                 : 'btn-primary hover:shadow-lg'
//             }`}
//           >
//             {isAsking ? (
//               <LoadingSpinner size="small" />
//             ) : (
//               <Send className="w-4 h-4" />
//             )}
//           </button>
//         </div>
        
//         {uploadedDocuments.length === 0 && (
//           <p className="text-xs text-gray-500 mt-2">
//             Upload PDF documents using the upload section to enable the chat functionality.
//           </p>
//         )}
//       </div>
//     </div>
//   );
// };

// export default StudyMateChat;

import React, { useState, useRef, useEffect } from 'react';
import { Send, MessageSquare, Book, User, Bot, Clock, Copy, Check, AlertCircle } from 'lucide-react';
import { studyMateService } from '../services/api';
import LoadingSpinner from './LoadingSpinner';

const StudyMateChat = ({ uploadedDocuments = [] }) => {
  const [messages, setMessages] = useState([]);
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [isAsking, setIsAsking] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [copiedMessageId, setCopiedMessageId] = useState(null);
  const [connectionError, setConnectionError] = useState(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    createNewSession();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const createNewSession = async () => {
    try {
      const response = await studyMateService.createSession();
      setSessionId(response.session_id);
      setConnectionError(null);
      
      setMessages([{
        id: Date.now(),
        type: 'bot',
        content: "Hello! I'm StudyMate, your AI academic assistant. Upload your PDFs and ask me any questions about your study materials!",
        timestamp: new Date(),
        sources: []
      }]);
    } catch (error) {
      console.error('Failed to create session:', error);
      setConnectionError('Failed to connect to StudyMate backend. Please ensure the backend is running on port 8000.');
      
      setMessages([{
        id: Date.now(),
        type: 'bot',
        content: "I'm having trouble connecting to the backend. Please check that your StudyMate backend is running on http://localhost:8000",
        timestamp: new Date(),
        sources: [],
        isError: true
      }]);
    }
  };

  const askQuestion = async () => {
    if (!currentQuestion.trim() || isAsking) return;

    if (uploadedDocuments.length === 0) {
      setMessages(prev => [...prev, {
        id: Date.now(),
        type: 'bot',
        content: "Please upload some PDF documents first so I can answer questions about them!",
        timestamp: new Date(),
        sources: []
      }]);
      return;
    }

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: currentQuestion.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setIsAsking(true);

    // Add typing indicator
    const typingMessage = {
      id: Date.now() + 1,
      type: 'bot',
      content: '',
      isTyping: true,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, typingMessage]);

    try {
      // ✅ FIXED: Use updated API with proper options
      const response = await studyMateService.askQuestion(
        currentQuestion.trim(), 
        sessionId,
        {
          documentIds: null, // Can be customized later
          maxChunks: 5,
          temperature: 0.3
        }
      );
      
      // ✅ FIXED: Handle actual backend response structure
      const botMessage = {
        id: Date.now() + 2,
        type: 'bot',
        content: response.answer || "I received an empty response.",
        timestamp: new Date(),
        sources: response.sources || [],
        confidence: response.confidence || 0.5,
        processingTime: response.processing_time || 0,
        chunksUsed: response.chunks_used || 0,
        llmStats: response.llm_stats || {}
      };

      // Remove typing indicator and add actual response
      setMessages(prev => prev.filter(msg => !msg.isTyping).concat(botMessage));
      
    } catch (error) {
      console.error('Question asking failed:', error);
      
      // ✅ FIXED: Better error handling with specific messages
      let errorMessage = "I'm sorry, I encountered an error while processing your question.";
      
      if (error.response?.status === 400) {
        errorMessage = error.response.data.detail || "Please check your question and try again.";
      } else if (error.response?.status === 500) {
        errorMessage = "Server error. The backend might be processing your request. Please try again.";
      } else if (error.code === 'NETWORK_ERROR' || !error.response) {
        errorMessage = "Cannot connect to StudyMate backend. Please ensure it's running on port 8000.";
        setConnectionError('Backend connection lost');
      }
      
      setMessages(prev => prev.filter(msg => !msg.isTyping).concat({
        id: Date.now() + 2,
        type: 'bot',
        content: errorMessage,
        timestamp: new Date(),
        sources: [],
        isError: true,
        errorDetails: error.response?.data || error.message
      }));
    } finally {
      setIsAsking(false);
    }

    setCurrentQuestion('');
  };

  const copyToClipboard = async (text, messageId) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedMessageId(messageId);
      setTimeout(() => setCopiedMessageId(null), 2000);
    } catch (error) {
      console.error('Failed to copy:', error);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      askQuestion();
    }
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const suggestedQuestions = [
    "What are the main topics covered in this document?",
    "Can you summarize the key points?",
    "What are the important definitions mentioned?",
    "Explain the methodology used in this research.",
    "What are the conclusions drawn in this study?"
  ];

  return (
    <div className="card p-0 h-full flex flex-col max-h-[600px]">
      {/* Header */}
      <div className={`p-4 border-b border-gray-200 rounded-t-lg ${
        connectionError ? 'bg-gradient-to-r from-red-500 to-red-600' : 'bg-gradient-to-r from-primary-500 to-primary-600'
      } text-white`}>
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-white/20 rounded-lg">
            {connectionError ? (
              <AlertCircle className="w-5 h-5" />
            ) : (
              <MessageSquare className="w-5 h-5" />
            )}
          </div>
          <div>
            <h2 className="text-lg font-semibold">
              StudyMate Assistant
              {connectionError && (
                <span className="ml-2 text-sm font-normal opacity-90">
                  (Connection Issue)
                </span>
              )}
            </h2>
            <p className="text-primary-100 text-sm">
              {connectionError ? (
                connectionError
              ) : uploadedDocuments.length > 0 ? (
                `Ready to answer questions about ${uploadedDocuments.length} document(s)`
              ) : (
                'Upload documents to start chatting'
              )}
            </p>
          </div>
        </div>
        
        {connectionError && (
          <button
            onClick={createNewSession}
            className="mt-2 px-3 py-1 bg-white/20 hover:bg-white/30 rounded text-sm transition-colors"
          >
            Retry Connection
          </button>
        )}
      </div>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0">
        {messages.map((message) => (
          <div key={message.id} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`flex max-w-[80%] ${message.type === 'user' ? 'flex-row-reverse' : 'flex-row'} space-x-3`}>
              {/* Avatar */}
              <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                message.type === 'user' 
                  ? 'bg-primary-500 text-white ml-3' 
                  : message.isError
                    ? 'bg-red-200 text-red-600 mr-3'
                    : 'bg-gray-200 text-gray-600 mr-3'
              }`}>
                {message.type === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
              </div>

              {/* Message Content */}
              <div className={`flex-1 ${message.type === 'user' ? 'mr-3' : 'ml-3'}`}>
                <div className={`p-3 rounded-lg ${
                  message.type === 'user' 
                    ? 'bg-primary-500 text-white' 
                    : message.isError 
                      ? 'bg-red-50 text-red-800 border border-red-200'
                      : 'bg-gray-100 text-gray-800'
                }`}>
                  {message.isTyping ? (
                    <div className="flex items-center space-x-2">
                      <LoadingSpinner size="small" />
                      <span className="text-sm">StudyMate is thinking...</span>
                    </div>
                  ) : (
                    <>
                      <p className="whitespace-pre-wrap">{message.content}</p>
                      
                      {/* ✅ FIXED: Sources handling */}
                      {message.sources && message.sources.length > 0 && (
                        <div className="mt-3 pt-3 border-t border-gray-300">
                          <div className="flex items-center space-x-1 mb-2">
                            <Book className="w-4 h-4" />
                            <span className="text-xs font-medium">Sources:</span>
                          </div>
                          <div className="space-y-1">
                            {message.sources.map((source, index) => (
                              <div key={index} className="text-xs bg-white/50 p-2 rounded border">
                                <div className="font-medium">{source.document}</div>
                                <div className="text-gray-600 mt-1">
                                  {source.excerpt || source.text?.substring(0, 100)}...
                                </div>
                                {source.page_numbers && source.page_numbers.length > 0 && (
                                  <div className="text-gray-500 mt-1">
                                    Pages: {source.page_numbers.join(', ')}
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      {/* ✅ FIXED: Enhanced metadata display */}
                      {(message.confidence || message.processingTime || message.chunksUsed) && (
                        <div className="mt-2 pt-2 border-t border-gray-300 text-xs text-gray-600">
                          {message.confidence && (
                            <span>Confidence: {Math.round(message.confidence * 100)}%</span>
                          )}
                          {message.processingTime && (
                            <span> • Response time: {message.processingTime.toFixed(2)}s</span>
                          )}
                          {message.chunksUsed && (
                            <span> • Sources used: {message.chunksUsed}</span>
                          )}
                          {message.llmStats?.model_used && (
                            <div className="mt-1">Model: {message.llmStats.model_used}</div>
                          )}
                        </div>
                      )}

                      {/* Error details for debugging */}
                      {message.isError && message.errorDetails && (
                        <details className="mt-2 text-xs">
                          <summary className="cursor-pointer text-red-600">Technical Details</summary>
                          <pre className="mt-1 p-2 bg-red-100 rounded text-red-800 overflow-x-auto">
                            {JSON.stringify(message.errorDetails, null, 2)}
                          </pre>
                        </details>
                      )}
                    </>
                  )}
                </div>
                
                {/* Message Actions */}
                {!message.isTyping && message.type === 'bot' && (
                  <div className="flex items-center space-x-2 mt-2">
                    <button
                      onClick={() => copyToClipboard(message.content, message.id)}
                      className="p-1 text-gray-400 hover:text-gray-600 rounded transition-colors"
                      title="Copy message"
                    >
                      {copiedMessageId === message.id ? (
                        <Check className="w-4 h-4 text-green-500" />
                      ) : (
                        <Copy className="w-4 h-4" />
                      )}
                    </button>
                    <span className="text-xs text-gray-400 flex items-center">
                      <Clock className="w-3 h-3 mr-1" />
                      {formatTimestamp(message.timestamp)}
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Suggested Questions */}
      {messages.length <= 1 && uploadedDocuments.length > 0 && !connectionError && (
        <div className="p-4 border-t border-gray-200 bg-gray-50">
          <p className="text-sm text-gray-600 mb-2">Try asking:</p>
          <div className="flex flex-wrap gap-2">
            {suggestedQuestions.slice(0, 3).map((question, index) => (
              <button
                key={index}
                onClick={() => setCurrentQuestion(question)}
                className="text-xs bg-white border border-gray-300 hover:border-primary-500 hover:text-primary-600 px-3 py-1 rounded-full transition-colors"
              >
                {question}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input Area */}
      <div className="p-4 border-t border-gray-200">
        <div className="flex space-x-3">
          <div className="flex-1">
            <textarea
              value={currentQuestion}
              onChange={(e) => setCurrentQuestion(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={
                connectionError 
                  ? "Fix connection to start asking questions..."
                  : uploadedDocuments.length > 0 
                    ? "Ask a question about your documents..." 
                    : "Upload documents first to start asking questions..."
              }
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none resize-none"
              rows={2}
              disabled={isAsking || uploadedDocuments.length === 0 || connectionError}
            />
          </div>
          <button
            onClick={askQuestion}
            disabled={!currentQuestion.trim() || isAsking || uploadedDocuments.length === 0 || connectionError}
            className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 flex items-center space-x-2 ${
              !currentQuestion.trim() || isAsking || uploadedDocuments.length === 0 || connectionError
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'btn-primary hover:shadow-lg'
            }`}
          >
            {isAsking ? (
              <LoadingSpinner size="small" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </div>
        
        {uploadedDocuments.length === 0 && !connectionError && (
          <p className="text-xs text-gray-500 mt-2">
            Upload PDF documents using the upload section to enable the chat functionality.
          </p>
        )}
      </div>
    </div>
  );
};

export default StudyMateChat;