import React, { useState, useRef, useEffect } from 'react';
import { Upload, Send, BarChart3, Moon, Sun, X } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './App.css';

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

  const removeImage = () => {
    setImageFile(null);
    setImagePreview(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleSubmit = async () => {
    if ((!query && !imageFile) || isLoading) return;

    setIsLoading(true);

    // 1. Simpan referensi lokal sebelum state direset
    const currentQuery = query || "Analisis gambar ini";
    const currentImageFile = imageFile;
    const currentImagePreview = imagePreview;

    // 2. Tambahkan pesan User ke UI dan siapkan payload history
    const userMessage = {
      role: 'user',
      content: currentQuery,
      image: currentImagePreview
    };

    const currentMessages = [...messages, userMessage];
    // Update UI immediately so user sees their message
    setMessages(currentMessages);

    // 3. Siapkan History untuk dikirim (pakai currentMessages agar tidak ketinggalan pesan baru)
    const historyPayload = currentMessages.map(msg => ({
      role: msg.role,
      content: msg.content || "",
      // sertakan flag image kalau ada (berguna untuk debugging di backend)
      hasImage: !!msg.image
    }));

    // 4. Buat FormData SEBELUM reset
    const formData = new FormData();
    formData.append('query', currentQuery);
    formData.append('history', JSON.stringify(historyPayload));

    if (currentImageFile) {
      formData.append('image_file', currentImageFile);
    }

    // 5. Reset Input (SETELAH FormData dibuat)
    setQuery("");
    setImageFile(null);
    // Debug: log what's about to be sent (FormData is not directly printable so we log values)
      try {
        console.debug("[DEBUG] Outgoing request:", {
          query: formData.get('query'),
          history: formData.get('history'),
          image_present: !!formData.get('image_file')
        });
      } catch (e) {
        console.warn("[DEBUG] Failed to log FormData directly", e);
      }
    setImagePreview(null);
    if (fileInputRef.current) fileInputRef.current.value = "";

    const aiMessage = { role: 'ai', content: '' };
    // Ensure placeholder is appended to the UI
    setMessages([...currentMessages, aiMessage]);

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let errorMsg = `Server error: ${response.status}`;
        try {
          const errorData = await response.json();
          errorMsg += ` - ${errorData.detail || ''}`;
        } catch (e) {
          // Ignore JSON parse error
        }
        throw new Error(errorMsg);
      }

      // 6. Baca Stream Response
      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        
        // Update pesan AI terakhir dengan chunk baru (IMMUTABLE)
        setMessages(prev => {
          const newMsgs = [...prev];
          const lastMsg = newMsgs[newMsgs.length - 1];
          if (lastMsg && lastMsg.role === 'ai') {
            newMsgs[newMsgs.length - 1] = {
              ...lastMsg,
              content: lastMsg.content + chunk
            };
          }
          return newMsgs;
        });
      }

    } catch (error) {
      console.error("Fetch error:", error);
      setMessages(prev => {
        const newMsgs = [...prev];
        const lastMsg = newMsgs[newMsgs.length - 1];
        if (lastMsg && lastMsg.role === 'ai') {
          newMsgs[newMsgs.length - 1] = {
            ...lastMsg,
            content: `⚠️ Maaf, terjadi kesalahan: ${error.message}`
          };
        }
        return newMsgs;
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
            title="Ganti Tema"
          >
            {isDarkMode ? <Sun className="icon" /> : <Moon className="icon" />}
          </button>
        </div>
      </header>

      {/* Chat Area */}
      <main className="chat-area">
        <div className="messages-container">
          {messages.length === 0 && (
            <div className="welcome-message">
              <h2>Selamat Datang! 👋</h2>
              <p>Saya siap membantu menganalisis chart trading Anda.</p>
              <p>Silakan upload gambar chart atau tanya tentang konsep teknikal.</p>
            </div>
          )}

          {messages.map((msg, index) => (
            <div key={index} className={`message ${msg.role}`}>
              <div className="message-content">
                {msg.role === 'user' && msg.image && (
                  <div className="message-image">
                    <img src={msg.image} alt="Uploaded chart" />
                  </div>
                )}
                
                <div className="markdown-body">
                  <ReactMarkdown 
                    remarkPlugins={[remarkGfm]}
                    components={{
                      h1: ({node, ...props}) => <h1 style={{fontSize: '1.5em', fontWeight: 'bold', margin: '0.5em 0'}} {...props} />,
                      h2: ({node, ...props}) => <h2 style={{fontSize: '1.3em', fontWeight: 'bold', margin: '0.8em 0 0.4em', borderBottom: '1px solid rgba(255,255,255,0.2)', paddingBottom: '4px'}} {...props} />,
                      ul: ({node, ...props}) => <ul style={{paddingLeft: '20px', margin: '0.5em 0'}} {...props} />,
                      li: ({node, ...props}) => <li style={{margin: '0.3em 0'}} {...props} />,
                      p: ({node, ...props}) => <p style={{margin: '0.5em 0', lineHeight: '1.6'}} {...props} />,
                      strong: ({node, ...props}) => <strong style={{color: isDarkMode ? '#fbbf24' : '#d97706'}} {...props} />,
                    }}
                  >
                    {msg.content}
                  </ReactMarkdown>
                </div>
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
                  <span>Sedang berpikir...</span>
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
          {imagePreview && (
            <div className="image-preview-container">
              <div className="preview-box">
                <img src={imagePreview} alt="Preview" />
                <button onClick={removeImage} className="remove-image-btn" title="Hapus gambar">
                  <X size={16} />
                </button>
              </div>
              <span className="file-name">{imageFile?.name}</span>
            </div>
          )}
          
          <div className="chat-form">
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="upload-button"
              title="Upload Gambar Chart"
            >
              <Upload className="icon" />
            </button>
            
            <input
              type="file"
              accept="image/png, image/jpeg, image/jpg"
              ref={fileInputRef}
              onChange={handleFileChange}
              className="hidden-file-input"
            />

            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={imageFile ? "Tambahkan pertanyaan tentang gambar ini..." : "Ketik pesan Anda..."}
              className="text-input"
              disabled={isLoading}
            />

            <button
              type="button"
              onClick={handleSubmit}
              disabled={(!query && !imageFile) || isLoading}
              className="send-button"
              title="Kirim"
            >
              <Send className="icon" />
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}