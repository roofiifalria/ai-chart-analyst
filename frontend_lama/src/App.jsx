import React, { useState, useRef, useEffect } from 'react';

// URL Backend API kita
const API_URL = "http://localhost:8000/api/analyze_chart";

export default function App() {
  const [messages, setMessages] = useState([]); // Menyimpan semua pesan chat
  const [query, setQuery] = useState(""); // Teks di input box
  const [imageFile, setImageFile] = useState(null); // File gambar
  const [imagePreview, setImagePreview] = useState(null); // URL preview gambar
  const [isLoading, setIsLoading] = useState(false); // Status loading
  
  const fileInputRef = useRef(null); // Ref untuk tombol upload
  const messagesEndRef = useRef(null); // Ref untuk auto-scroll

  // Auto-scroll ke pesan terbaru
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Handle perubahan file
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      // Buat URL preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  // Handle pengiriman form
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query || !imageFile || isLoading) return;

    setIsLoading(true);

    // 1. Tambahkan pesan pengguna ke chat
    const userMessage = {
      role: 'user',
      content: query,
      image: imagePreview
    };
    setMessages(prev => [...prev, userMessage]);

    // 2. Buat FormData untuk dikirim
    const formData = new FormData();
    formData.append('query', query);
    formData.append('image_file', imageFile);

    // 3. Tambahkan pesan AI (kosong) untuk diisi oleh streaming
    const aiMessage = { role: 'ai', content: '' };
    setMessages(prev => [...prev, aiMessage]);

    // Reset input
    setQuery("");
    setImageFile(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = ""; // Reset input file
    }

    try {
      // 4. Panggil API (Streaming)
      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');

      // 5. Baca stream
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        
        // Perbarui pesan AI terakhir dengan chunk baru
        setMessages(prev => {
          const lastMessage = prev[prev.length - 1];
          // Pastikan kita hanya memperbarui pesan AI
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
      // Tampilkan error di chat
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

  return (
    <div className="flex flex-col h-screen bg-gray-900 text-white font-sans">
      {/* Header */}
      <header className="bg-gray-800 shadow-md p-4">
        <h1 className="text-xl font-bold text-center">AI Chart Analyst</h1>
      </header>

      {/* Chat Area */}
      <main className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, index) => (
          <div key={index} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`rounded-lg p-3 max-w-lg ${msg.role === 'user' ? 'bg-blue-600' : 'bg-gray-700'}`}>
              {/* Tampilkan gambar jika ada di pesan pengguna */}
              {msg.role === 'user' && msg.image && (
                <img src={msg.image} alt="Chart preview" className="rounded-md mb-2 max-h-48" />
              )}
              {/* Tampilkan teks, ganti baris baru \n dengan <br> (ini versi sederhana) */}
              <p className="whitespace-pre-wrap">{msg.content}</p>
            </div>
          </div>
        ))}
        {/* Indikator loading */}
        {isLoading && (
          <div className="flex justify-start">
            <div className="rounded-lg p-3 bg-gray-700">
              <p className="animate-pulse">AI sedang menganalisis...</p>
            </div>
          </div>
        )}
        {/* Ref untuk auto-scroll */}
        <div ref={messagesEndRef} />
      </main>

      {/* Input Form Area */}
      <footer className="bg-gray-800 p-4 shadow-inner">
        <form onSubmit={handleSubmit} className="flex items-center space-x-3">
          {/* Tombol Upload */}
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="p-2 rounded-full bg-gray-700 hover:bg-gray-600 focus:outline-none"
            title="Upload Chart Image"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
            </svg>
          </button>
          <input
            type="file"
            accept="image/png, image/jpeg"
            ref={fileInputRef}
            onChange={handleFileChange}
            className="hidden"
          />

          {/* Input Teks */}
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={imagePreview ? "Ketik pertanyaan Anda..." : "Silakan upload gambar chart dulu..."}
            className="flex-1 p-3 rounded-lg bg-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={!imageFile}
          />

          {/* Tombol Kirim */}
          <button
            type="submit"
            disabled={!query || !imageFile || isLoading}
            className="p-3 rounded-lg bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Kirim
          </button>
        </form>
        {/* Preview nama file */}
        {imageFile && (
          <p className="text-xs text-gray-400 mt-2 text-center">
            Siap menganalisis: {imageFile.name}
          </p>
        )}
      </footer>
    </div>
  );
}