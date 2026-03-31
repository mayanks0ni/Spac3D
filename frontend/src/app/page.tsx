"use client";

import { useState } from "react";
import "./globals.css";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);

    const formData = new FormData();
    formData.append("image", file);

    try {
        const res = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResult(data);
    } catch (error) {
      console.error(error);
      setResult({ error: "Upload failed. Is the backend running?" });
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="container">
      <div className="card">
        <h1 className="title">Pseudo-3D Pipeline</h1>
        <p className="subtitle">Upload a single 2D image to generate a 3D point cloud & segmentation.</p>
        
        <div className="upload-box">
          <input type="file" onChange={handleFileChange} accept="image/*" className="file-input" id="file-upload"/>
          <label htmlFor="file-upload" className="file-label">
            {file ? file.name : "Choose an Image"}
          </label>
        </div>

        <button 
          onClick={handleUpload} 
          disabled={!file || uploading} 
          className="upload-button"
        >
          {uploading ? "Processing..." : "Generate 3D (Day 1-2 Demo)"}
        </button>

        {result && (
          <div className="result-box">
            <h3 className="result-title">Response from FastAPI:</h3>
            <pre className="result-json">{JSON.stringify(result, null, 2)}</pre>
          </div>
        )}
      </div>
    </div>
  );
}
