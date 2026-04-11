import React, { useState, useCallback, useRef } from 'react'

interface Detection {
  label: string
  confidence: number
  bbox: [number, number, number, number]
}

interface FurnitureItem {
  id: string
  name: string
  style: string
  image_url: string
  similarity_score: number
}

interface AnalysisResult {
  status: string
  detections: Detection[]
  style: string
  confidence: number
  recommendations: FurnitureItem[]
  image_size: { width: number; height: number }
}

const STYLE_ACCENTS: Record<string, string> = {
  Minimalist: '#A8DADC',
  Modern: '#E9C46A',
  Bohemian: '#E76F51',
  Industrial: '#8D99AE',
  Scandinavian: '#95D5B2',
  Traditional: '#C77DFF',
}

const API_URL = 'http://127.0.0.1:8000'

export default function App() {
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const accentColor = result ? (STYLE_ACCENTS[result.style] ?? '#A8DADC') : '#A8DADC'

  // Update CSS --accent variable dynamically when style is detected
  React.useEffect(() => {
    const root = document.documentElement
    root.style.setProperty('--accent', accentColor)
    root.style.setProperty('--accent-dim', `${accentColor}1a`)
    root.style.setProperty('--accent-glow', `${accentColor}30`)
  }, [accentColor])

  const handleFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Please upload an image file (JPG, PNG, WEBP)')
      return
    }
    setImageFile(file)
    setImagePreview(URL.createObjectURL(file))
    setResult(null)
    setError(null)
  }

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }, [])

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleAnalyze = async () => {
    if (!imageFile) return
    setLoading(true)
    setError(null)
    try {
      const formData = new FormData()
      formData.append('image', imageFile)
      const response = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        body: formData,
      })
      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        throw new Error(err.detail ?? `Server error ${response.status}`)
      }
      const data: AnalysisResult = await response.json()
      setResult(data)
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error'
      setError(`Network Error: ${message} | URL: ${API_URL}/analyze`)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setImageFile(null)
    setImagePreview(null)
    setResult(null)
    setError(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  return (
    <div className="app-wrapper">
      <nav className="navbar">
        <div className="navbar-logo">
          <div className="logo-icon" />
          <span className="logo-text">SPAC3D</span>
        </div>
        <span className="navbar-tagline">
          AI Room Analysis &amp; Furniture Intelligence
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, color: 'var(--text-muted)' }}>
          <span className="live-dot" />
          Connected
        </div>
      </nav>

      <main className="main-content">

        <div className="glass-panel">
          <p className="panel-title">Room Image</p>

          {!imagePreview ? (
            /* Drop Zone */
            <div
              id="upload-dropzone"
              className={`upload-zone ${isDragOver ? 'drag-over' : ''}`}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={() => setIsDragOver(false)}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                style={{ display: 'none' }}
                onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
              />
              <div className="upload-icon" />
              <p className="upload-title">Drop your room photo here</p>
              <p className="upload-sub">
                or <span>click to browse</span>
              </p>
              <p className="upload-sub" style={{ fontSize: 12, marginTop: 4 }}>
                JPG · PNG · WEBP · max 20MB
              </p>
            </div>
          ) : (
            <ImageWithBboxes
              src={imagePreview}
              detections={result?.detections ?? []}
              imageSize={result?.image_size}
              accentColor={accentColor}
            />
          )}

          {error && (
            <div className="error-banner">
              {error}
            </div>
          )}

          {imagePreview && (
            <>
              <button
                id="btn-analyze"
                className="btn-analyze"
                onClick={handleAnalyze}
                disabled={loading}
              >
                {loading ? (
                  <><div className="spinner" /> Analyzing Room...</>
                ) : (
                  <>Analyze Room</>
                )}
              </button>
              <button className="btn-reset" onClick={handleReset}>
                Clear &amp; Start Over
              </button>
            </>
          )}
        </div>

        <div className="glass-panel" style={{ overflowY: 'auto', maxHeight: 'calc(100vh - 180px)' }}>
          <p className="panel-title">Analysis Results</p>

          {!result && !loading && (
            <div className="empty-state">
              <div className="empty-state-icon" />
              <p className="empty-state-text">
                Upload a room photo and click Analyze<br />to get style detection &amp; furniture recommendations
              </p>
            </div>
          )}

          {loading && (
            <div className="empty-state">
              <div style={{ fontSize: 40, opacity: 0.3 }} />
              <p className="empty-state-text">
                Running ML pipeline...<br />
                <span style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 6, display: 'block' }}>
                  YOLOv8 → ResNet50 → PCA → kNN → SVM
                </span>
              </p>
            </div>
          )}

          {result && (
            <div className="fade-in-up">
            <div className="style-result">
                <p className="style-label">Detected Room Style</p>
                <p className="style-name">{result.style}</p>
                <div className="confidence-row">
                  <span className="confidence-label">SVM Confidence</span>
                  <span className="confidence-value">{(result.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="confidence-bar-track">
                  <div
                    className="confidence-bar-fill"
                    style={{ width: `${result.confidence * 100}%` }}
                  />
                </div>
              </div>

              {result.detections.length > 0 && (
                <div className="detections-section">
                  <p className="detections-header">
                    {result.detections.length} object{result.detections.length !== 1 ? 's' : ''} detected
                  </p>
                  <div className="detection-chips">
                    {result.detections.map((d, i) => (
                      <div className="detection-chip" key={i}>
                        <span className="chip-dot" />
                        {d.label}
                        <span style={{ opacity: 0.5, fontSize: 11 }}>
                          {(d.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {result.recommendations.length > 0 && (
                <>
                  <p className="recommendations-title">
                    Furniture Recommendations
                  </p>
                  <div className="furniture-grid">
                    {result.recommendations.map((item) => (
                      <FurnitureCard key={item.id} item={item} />
                    ))}
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  )
}

/* ─────────────────────────────────────────────────────────────────── */
/* Image with SVG bounding box overlay                                 */
/* ─────────────────────────────────────────────────────────────────── */
function ImageWithBboxes({
  src,
  detections,
  imageSize,
  accentColor,
}: {
  src: string
  detections: Detection[]
  imageSize?: { width: number; height: number }
  accentColor: string
}) {
  const imgRef = useRef<HTMLImageElement>(null)
  const [displaySize, setDisplaySize] = useState({ w: 0, h: 0 })

  const onLoad = () => {
    if (imgRef.current) {
      setDisplaySize({
        w: imgRef.current.offsetWidth,
        h: imgRef.current.offsetHeight,
      })
    }
  }

  // Scale pixel coords from original image size to displayed size
  const scaleX = displaySize.w / ((imageSize?.width ?? displaySize.w) || 1)
  const scaleY = displaySize.h / ((imageSize?.height ?? displaySize.h) || 1)

  return (
    <div className="image-container">
      <img ref={imgRef} src={src} alt="Room" onLoad={onLoad} style={{ width: '100%' }} />

      {detections.length > 0 && displaySize.w > 0 && (
        <svg
          className="bbox-overlay"
          viewBox={`0 0 ${displaySize.w} ${displaySize.h}`}
          xmlns="http://www.w3.org/2000/svg"
        >
          {detections.map((det, i) => {
            const [x1, y1, x2, y2] = det.bbox
            const rx = x1 * scaleX
            const ry = y1 * scaleY
            const rw = (x2 - x1) * scaleX
            const rh = (y2 - y1) * scaleY
            return (
              <g key={i}>
                <rect
                  x={rx} y={ry} width={rw} height={rh}
                  fill="none"
                  stroke={accentColor}
                  strokeWidth="2"
                  strokeDasharray="6 3"
                  opacity="0.9"
                />
                <rect
                  x={rx} y={ry - 22} width={det.label.length * 8 + 20} height={20}
                  fill={accentColor}
                  rx="4"
                />
                <text
                  x={rx + 8} y={ry - 7}
                  fill="#0a0a0f"
                  fontSize="11"
                  fontWeight="700"
                  fontFamily="Inter, sans-serif"
                  textAnchor="start"
                >
                  {det.label} {(det.confidence * 100).toFixed(0)}%
                </text>
              </g>
            )
          })}
        </svg>
      )}
    </div>
  )
}

/* ─────────────────────────────────────────────────────────────────── */
/* Furniture recommendation card                                        */
/* ─────────────────────────────────────────────────────────────────── */
function FurnitureCard({ item }: { item: FurnitureItem }) {
  return (
    <div className="furniture-card">
      <img
        src={item.image_url}
        alt={item.name}
        className="furniture-card-img"
        onError={(e) => {
          ;(e.target as HTMLImageElement).src = `https://picsum.photos/seed/${item.id}/400/300`
        }}
      />
      <div className="furniture-card-body">
        <p className="furniture-card-name">{item.name}</p>
        <div className="furniture-card-meta">
          <span className="furniture-style-tag">{item.style}</span>
          <div className="furniture-similarity">
            <div className="similarity-bar">
              <div
                className="similarity-fill"
                style={{ width: `${item.similarity_score * 100}%` }}
              />
            </div>
            <span className="similarity-score">
              {(item.similarity_score * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
