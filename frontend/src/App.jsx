// frontend/src/App.jsx

import React, { useState, useRef, useEffect } from 'react';
import { Upload, Send, BarChart3, Moon, Sun } from 'lucide-react';

const API_URL = "http://localhost:8000/api/analyze_chart";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState("");
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(true);
  
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async () => {
    if (!query || !imageFile || isLoading) return;

    setIsLoading(true);

    const userMessage = {
      role: 'user',
      content: query,
      image: imagePreview
    };
    setMessages(prev => [...prev, userMessage]);

    const formData = new FormData();
    formData.append('query', query);
    formData.append('image_file', imageFile);

    const aiMessage = { role: 'ai', content: '' };
    setMessages(prev => [...prev, aiMessage]);

    setQuery("");
    setImageFile(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        
        setMessages(prev => {
          const lastMessage = prev[prev.length - 1];
          if (lastMessage.role === 'ai') {
            const updatedMessage = {
              ...lastMessage,
              content: lastMessage.content + chunk,
            };
            return [...prev.slice(0, -1), updatedMessage];
          }
          return prev;
        });
      }

    } catch (error) {
      console.error("Fetch error:", error);
      setMessages(prev => {
        const lastMessage = prev[prev.length - 1];
        const updatedMessage = {
          ...lastMessage,
          content: `Maaf, terjadi kesalahan: ${error.message}`,
        };
        return [...prev.slice(0, -1), updatedMessage];
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className={`app-container ${isDarkMode ? 'dark-mode' : 'light-mode'}`}>
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="header-left">
            <BarChart3 className="header-icon" />
            <h1>AI Chart Analyst</h1>
          </div>
          
          <button
            onClick={() => setIsDarkMode(!isDarkMode)}
            className="theme-toggle"
          >
            {isDarkMode ? <Sun className="icon" /> : <Moon className="icon" />}
          </button>
        </div>
      </header>

      {/* Chat Area */}
      <main className="chat-area">
        <div className="messages-container">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`message ${msg.role}`}
            >
              <div className="message-content">
                {msg.role === 'user' && msg.image && (
                  <div className="message-image">
                    <img src={msg.image} alt="Chart preview" />
                  </div>
                )}
                <p>{msg.content}</p>
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="message ai">
              <div className="message-content">
                <div className="loading-indicator">
                  <div className="loading-dots">
                    <div className="dot"></div>
                    <div className="dot"></div>
                    <div className="dot"></div>
                  </div>
                  <span>AI sedang menganalisis...</span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Footer Input */}
      <footer className="footer">
        <div className="footer-content">
          {imageFile && (
            <div className="file-preview">
              <div className="preview-dot"></div>
              <span>
                Siap menganalisis: <strong>{imageFile.name}</strong>
              </span>
            </div>
          )}
          
          <div className="chat-form">
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="upload-button"
              title="Upload Chart Image"
            >
              <Upload className="icon" />
            </button>
            
            <input
              type="file"
              accept="image/png, image/jpeg"
              ref={fileInputRef}
              onChange={handleFileChange}
              className="hidden-file-input"
            />

            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={imagePreview ? "Ketik pertanyaan Anda..." : "Silakan upload gambar chart dulu..."}
              disabled={!imageFile}
              className="text-input"
            />

            <button
              type="button"
              onClick={handleSubmit}
              disabled={!query || !imageFile || isLoading}
              className="send-button"
            >
              <Send className="icon" />
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}