import React, { useRef, useState } from 'react'
import type { ServerFile } from '../services/api'

interface FileUploadProps {
  onFileUpload: (file: File) => void
  onLoadSample: (key: 'churn' | 'housing' | 'sales_ts' | 'customers') => void
  onSelectExisting: (fileName: string) => void
  serverFiles: ServerFile[]
  isLoading: boolean
  error: string | null
  onClearError: () => void
}

export const FileUpload: React.FC<FileUploadProps> = ({
  onFileUpload,
  onLoadSample,
  onSelectExisting,
  serverFiles,
  isLoading,
  error,
  onClearError,
}) => {
  const [isDragOver, setIsDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndSubmit(e.dataTransfer.files[0])
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndSubmit(e.target.files[0])
    }
    e.target.value = ''
  }

  const handleDropzoneClick = (e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('button')) {
      return
    }
    if (!isLoading) {
      fileInputRef.current?.click()
    }
  }

  const validateAndSubmit = (file: File) => {
    onClearError()
    const validExts = ['.csv', '.xlsx', '.xls']
    const hasValidExt = validExts.some((ext) => file.name.toLowerCase().endsWith(ext))
    if (!hasValidExt) {
      alert('Please upload a valid .csv or .xlsx dataset file.')
      return
    }
    onFileUpload(file)
  }

  return (
    <div className="upload-container">
      {/* Hidden file input placed outside dropzone */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".csv,.xlsx,.xls"
        onChange={handleFileChange}
        style={{ display: 'none' }}
        disabled={isLoading}
      />

      {/* Error notification banner */}
      {error && (
        <div className="error-banner">
          <div className="error-content">
            <svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" strokeWidth="2" fill="none" className="error-icon">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
            <div>
              <p className="error-title">Upload or Processing Error</p>
              <p className="error-message">{error}</p>
            </div>
          </div>
          <button type="button" className="close-btn" onClick={onClearError} title="Dismiss">×</button>
        </div>
      )}

      {/* Windows Dialog Freeze Workaround Tip */}
      <div className="drag-tip-banner">
        <span className="tip-badge">Drag & Drop Recommended</span>
        <span className="tip-text">
          If clicking "Choose File" freezes Windows File Explorer on "Working on it...", simply <strong>drag and drop your file directly from your folder</strong> into the dropzone below!
        </span>
      </div>

      {/* Main Upload Dropzone */}
      <div
        className={`dropzone ${isDragOver ? 'dropzone-active' : ''} ${isLoading ? 'dropzone-loading' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleDropzoneClick}
      >
        {isLoading ? (
          <div className="loading-state">
            <div className="spinner"></div>
            <p className="loading-title">Ingesting & Processing Dataset...</p>
            <p className="loading-desc">
              Streaming file into backend and computing descriptive statistics...
            </p>
          </div>
        ) : (
          <div className="dropzone-inner">
            <div className="upload-icon-wrapper">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="upload-icon">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </div>
            <h3 className="upload-title">Drop your dataset file here, or browse</h3>
            <p className="upload-subtitle">
              Supports <strong>Drag & Drop</strong> for CSV or Excel files. Bypasses the Windows Open file dialog!
            </p>
            <button
              type="button"
              className="btn btn-primary btn-upload"
              onClick={(e) => {
                e.stopPropagation()
                fileInputRef.current?.click()
              }}
            >
              Choose File
            </button>
          </div>
        )}
      </div>

      {/* Existing Files on Server (Bypasses local file dialog completely!) */}
      {serverFiles.length > 0 && (
        <div className="samples-card">
          <div className="samples-header">
            <span className="samples-label">📁 Or Load from Already Uploaded Files on Server (No File Picker Needed):</span>
          </div>
          <div className="samples-grid">
            {serverFiles.map((f) => (
              <button
                key={f.file_name}
                type="button"
                className="sample-pill-btn"
                disabled={isLoading}
                onClick={() => onSelectExisting(f.file_name)}
                title={`Load ${f.file_name} (${f.size_display}) directly from server`}
              >
                <span className="sample-icon">📊</span>
                <div className="sample-info">
                  <span className="sample-name">{f.file_name}</span>
                  <span className="sample-detail">{f.size_display} on server</span>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Instant Test Sample Datasets */}
      <div className="samples-card">
        <div className="samples-header">
          <span className="samples-label">Instant 1-Click Test Datasets:</span>
        </div>
        <div className="samples-grid">
          <button
            type="button"
            className="sample-pill-btn"
            disabled={isLoading}
            onClick={() => onLoadSample('churn')}
          >
            <span className="sample-icon">👥</span>
            <div className="sample-info">
              <span className="sample-name">Customer Churn</span>
              <span className="sample-detail">Classification: Random Forest, XGBoost</span>
            </div>
          </button>

          <button
            type="button"
            className="sample-pill-btn"
            disabled={isLoading}
            onClick={() => onLoadSample('housing')}
          >
            <span className="sample-icon">🏡</span>
            <div className="sample-info">
              <span className="sample-name">Housing Valuation</span>
              <span className="sample-detail">Regression: Neural Networks, Boosted Trees</span>
            </div>
          </button>

          <button
            type="button"
            className="sample-pill-btn"
            disabled={isLoading}
            onClick={() => onLoadSample('sales_ts')}
          >
            <span className="sample-icon">📈</span>
            <div className="sample-info">
              <span className="sample-name">Daily Demand & Revenue</span>
              <span className="sample-detail">Time-Series: LSTM Deep Learning</span>
            </div>
          </button>

          <button
            type="button"
            className="sample-pill-btn"
            disabled={isLoading}
            onClick={() => onLoadSample('customers')}
          >
            <span className="sample-icon">🛍️</span>
            <div className="sample-info">
              <span className="sample-name">Customer Segmentation</span>
              <span className="sample-detail">Pattern Discovery: K-Means & PCA</span>
            </div>
          </button>
        </div>
      </div>
    </div>
  )
}
