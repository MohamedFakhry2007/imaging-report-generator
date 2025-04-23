import "./App.css";
import { useState, useRef } from "react";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [story, setStory] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const fileInputRef = useRef(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      // Reset states
      setSelectedImage(file);
      setStory("");
      setError("");
      
      // Create a preview URL
      const previewUrl = URL.createObjectURL(file);
      setImagePreview(previewUrl);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleImageChange({ target: { files: e.dataTransfer.files } });
    }
  };

  const handleGenerateStory = async () => {
    if (!selectedImage) {
      setError("يرجى اختيار صورة أولاً");
      return;
    }

    setIsLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", selectedImage);

      const response = await fetch(`${BACKEND_URL}/api/generate-story`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "فشل في إنشاء القصة");
      }

      const data = await response.json();
      setStory(data.story);
    } catch (err) {
      console.error("Error:", err);
      setError(err.message || "حدث خطأ أثناء إنشاء القصة");
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setStory("");
    setError("");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div className="app-container" dir="rtl">
      <header className="app-header">
        <h1 className="app-title">قصة من صورة</h1>
        <p className="app-subtitle">حول صورتك إلى قصة عربية فريدة بلمسة واحدة</p>
      </header>

      <main className="app-main">
        <section className="upload-section">
          <div 
            className="upload-area"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current && fileInputRef.current.click()}
          >
            {imagePreview ? (
              <div className="image-preview-container">
                <img src={imagePreview} alt="صورتك" className="image-preview" />
                <button className="reset-button" onClick={(e) => { e.stopPropagation(); handleReset(); }}>
                  اختر صورة أخرى
                </button>
              </div>
            ) : (
              <>
                <div className="upload-icon">📷</div>
                <p className="upload-text">انقر أو اسحب صورة هنا</p>
                <p className="upload-subtext">JPG, PNG, WEBP</p>
              </>
            )}
            <input
              type="file"
              ref={fileInputRef}
              accept="image/jpeg,image/png,image/webp"
              onChange={handleImageChange}
              className="file-input"
            />
          </div>

          <button 
            className={`generate-button ${!selectedImage ? 'disabled' : ''}`}
            onClick={handleGenerateStory}
            disabled={!selectedImage || isLoading}
          >
            {isLoading ? "جاري كتابة القصة..." : "أنشئ القصة"}
          </button>

          {error && <div className="error-message">{error}</div>}
        </section>

        {story && (
          <section className="story-section">
            <h2 className="story-title">قصتك الجديدة</h2>
            <div className="story-content">
              {story.split('\n').map((paragraph, index) => (
                paragraph ? <p key={index}>{paragraph}</p> : <br key={index} />
              ))}
            </div>
          </section>
        )}
      </main>

      <footer className="app-footer">
        <p>تطبيق قصة من صورة © {new Date().getFullYear()}</p>
      </footer>
    </div>
  );
}

export default App;
