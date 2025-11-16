// frontend/src/App.jsx (VERSI DIPERBARUI)

import React, { useState, useRef, useEffect } from 'react';
// Impor ikon-ikon Anda
import { Upload, Send, BarChart3, Moon, Sun } from 'lucide-react';

const API_URL = "http://localhost:8000/api/analyze_chart";

export default function App() {
  // Semua state Anda (termasuk dark mode) tetap ada
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

  // Perbarui handleFileChange agar menghapus chat lama
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
      
      // Hapus riwayat chat saat gambar baru diunggah
      setMessages([]);
    }
  };

  // --- FUNGSI handleSubmit YANG DIPERBARUI ---
  const handleSubmit = async () => {
    // [PERBAIKAN 1] Izinkan submit jika query ada (gambar opsional)
    if (!query || isLoading) return;

    setIsLoading(true);

    // [PERBAIKAN 2] Tampilkan gambar di pesan HANYA jika ini adalah pesan pertama
    // tentang gambar itu. Kita cek apakah imagePreview ada.
    const userMessage = {
      role: 'user',
      content: query,
      // Jika ada imagePreview, ini adalah pesan pertama tentang gambar itu
      // Pesan selanjutnya (follow-up) tidak akan menyertakan gambar
      image: imagePreview 
    };
    setMessages(prev => [...prev, userMessage]);

    const formData = new FormData();
    formData.append('query', query);
    
    // [PERBAIKAN 3] Hanya tambahkan gambar jika ada
    if (imageFile) {
      formData.append('image_file', imageFile);
    }

    const aiMessage = { role: 'ai', content: '' };
    setMessages(prev => [...prev, aiMessage]);

    // [PERBAIKAN 4] HANYA reset query. JANGAN reset gambar.
    // Ini memungkinkan pertanyaan follow-up.
    setQuery("");

    // [PERBAIKAN 5] Setelah gambar dikirim SATU KALI, 
    // kita hapus dari state agar pengiriman berikutnya
    // adalah teks-saja (follow-up).
    if (imageFile) {
      setImageFile(null);
      setImagePreview(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        // Tampilkan error yang lebih jelas
        const errorText = await response.text();
        let errorJson = {};
        try {
          errorJson = JSON.parse(errorText);
        } catch(e) {}
        
        const detail = errorJson.detail || errorText || `HTTP error! status: ${response.status}`;
        throw new Error(detail);
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
  // --- AKHIR FUNGSI handleSubmit ---

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  // --- KODE JSX (HTML) JUGA PERLU DIPERBAIKI ---
  return (
    <div className={`app-container ${isDarkMode ? 'dark-mode' : 'light-mode'}`}>
      {/* Header (Kode Anda tidak berubah) */}
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

      {/* Chat Area (Kode Anda tidak berubah) */}
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

      {/* Footer Input (Kode Anda DIPERBARUI) */}
      <footer className="footer">
        <div className="footer-content">
          {imagePreview && ( // Ubah imageFile menjadi imagePreview
            <div className="file-preview">
              <div className="preview-dot"></div>
              <span>
                Siap menganalisis: <strong>{imageFile?.name}</strong>
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
              // [PERBAIKAN 6] Ubah placeholder dan logik disabled
              placeholder={imagePreview ? "Tanya tentang gambar ini..." : "Tanya tentang konsep trading..."}
              disabled={isLoading} // Izinkan input teks kapan saja
              className="text-input"
            />

            <button
              type="button"
              onClick={handleSubmit}
              // [PERBAIKAN 7] Izinkan submit hanya dengan teks
              disabled={!query || isLoading}
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