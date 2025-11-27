// frontend/src/App.js
import "./App.css";
import { useState, useRef } from "react";

// Ideally, store this in .env
const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [report, setReport] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const fileInputRef = useRef(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      setReport("");
      setError("");
      setImagePreview(URL.createObjectURL(file));
    }
  };

  const handleGenerateReport = async () => {
    if (!selectedImage) return;

    setIsLoading(true);
    setError("");
    setReport("");

    try {
      const formData = new FormData();
      formData.append("file", selectedImage);

      // Note the updated endpoint
      const response = await fetch(`${BACKEND_URL}/api/generate-report`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Failed to generate report");
      }

      const data = await response.json();
      setReport(data.report);
    } catch (err) {
      console.error(err);
      setError("Error analyzing image. Please ensure it is a valid medical scan.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setReport("");
    setError("");
  };

  return (
    <div className="min-h-screen bg-gray-50 text-gray-800 font-sans">
      {/* Disclaimer Banner */}
      <div className="bg-yellow-100 border-b border-yellow-200 p-2 text-center text-xs text-yellow-800 font-medium">
        ⚠️ RESEARCH PROTOTYPE ONLY. NOT FOR CLINICAL DIAGNOSIS.
      </div>

      <header className="bg-white shadow-sm p-6 mb-8">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <h1 className="text-2xl font-bold text-blue-900">
            MedVision <span className="text-blue-400 font-light">Assist</span>
          </h1>
          <span className="text-sm text-gray-500">AI-Powered Radiology Triage</span>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 grid grid-cols-1 md:grid-cols-2 gap-8">
        
        {/* Left Column: Upload */}
        <section className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 h-fit">
          <h2 className="text-lg font-semibold mb-4 text-gray-700">Scan Upload</h2>
          
          <div 
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
              ${imagePreview ? 'border-blue-200 bg-blue-50' : 'border-gray-300 hover:border-blue-400'}`}
            onClick={() => fileInputRef.current.click()}
          >
            {imagePreview ? (
              <div className="relative">
                <img src={imagePreview} alt="Preview" className="max-h-64 mx-auto rounded shadow-sm" />
                <button 
                  onClick={(e) => { e.stopPropagation(); handleReset(); }}
                  className="mt-4 text-sm text-red-500 hover:text-red-700 underline"
                >
                  Remove Scan
                </button>
              </div>
            ) : (
              <div className="py-8">
                <div className="text-4xl mb-2">☢️</div>
                <p className="font-medium text-gray-600">Drop X-Ray / CT Slice here</p>
                <p className="text-xs text-gray-400 mt-2">Supported: JPG, PNG</p>
              </div>
            )}
            <input 
              type="file" 
              ref={fileInputRef} 
              className="hidden" 
              accept="image/*" 
              onChange={handleImageChange} 
            />
          </div>

          <button 
            className={`w-full mt-6 py-3 rounded-lg font-medium transition-all
              ${!selectedImage || isLoading 
                ? 'bg-gray-200 text-gray-400 cursor-not-allowed' 
                : 'bg-blue-600 text-white hover:bg-blue-700 shadow-md'}`}
            onClick={handleGenerateReport}
            disabled={!selectedImage || isLoading}
          >
            {isLoading ? "Analyzing Anatomy..." : "Generate Preliminary Report"}
          </button>
          
          {error && <div className="mt-4 p-3 bg-red-50 text-red-700 text-sm rounded">{error}</div>}
        </section>

        {/* Right Column: Report */}
        <section className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 min-h-[400px]">
          <h2 className="text-lg font-semibold mb-4 text-gray-700">Findings</h2>
          
          {report ? (
            <div className="prose prose-sm text-gray-600 whitespace-pre-wrap leading-relaxed animate-fade-in">
              {report}
            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-gray-300">
              <div className="text-3xl mb-2">📄</div>
              <p>No report generated yet.</p>
            </div>
          )}
        </section>

      </main>
    </div>
  );
}

export default App;