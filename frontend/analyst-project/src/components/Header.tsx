import React, { useState } from 'react'

interface HeaderProps {
  backendOnline: boolean
  backendStatusText: string
  backendUrl: string
  onUpdateBackendUrl: (url: string) => void
  onRefreshHealth: () => void
  onLoadSample: (key: 'churn' | 'housing' | 'sales_ts' | 'customers') => void
  hasData: boolean
  onReset: () => void
}

export const Header: React.FC<HeaderProps> = ({
  backendOnline,
  backendStatusText,
  backendUrl,
  onUpdateBackendUrl,
  onRefreshHealth,
  onLoadSample,
  hasData,
  onReset,
}) => {
  const [showConfig, setShowConfig] = useState(false)
  const [showSampleDropdown, setShowSampleDropdown] = useState(false)
  const [tempUrl, setTempUrl] = useState(backendUrl)

  const handleSaveConfig = (e: React.FormEvent) => {
    e.preventDefault()
    onUpdateBackendUrl(tempUrl)
    setShowConfig(false)
  }

  return (
    <header className="app-header">
      <div className="header-container">
        <div className="brand-group">
          <div className="brand-logo">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="logo-svg">
              <path d="M3 3v18h18" />
              <path d="m19 9-5 5-4-4-3 3" />
            </svg>
          </div>
          <div>
            <div className="brand-title-row">
              <h1 className="brand-title">InsightAI</h1>
              <span className="brand-badge">ML & DL Engine</span>
            </div>
            <p className="brand-subtitle">Descriptive Statistics, Predictive ML, Clustering & LSTM Forecasting</p>
          </div>
        </div>

        <div className="header-actions">
          {/* Quick sample dropdown */}
          <div className="sample-menu-wrapper" style={{ position: 'relative' }}>
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              title="Load instant pre-packaged datasets"
              onClick={() => setShowSampleDropdown(!showSampleDropdown)}
            >
              <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" strokeWidth="2" fill="none">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
                <line x1="16" y1="13" x2="8" y2="13" />
                <line x1="16" y1="17" x2="8" y2="17" />
              </svg>
              <span>Load Sample Data ▾</span>
            </button>

            {showSampleDropdown && (
              <div className="sample-dropdown-menu">
                <button
                  type="button"
                  className="dropdown-item"
                  onClick={() => {
                    onLoadSample('churn')
                    setShowSampleDropdown(false)
                  }}
                >
                  <span className="dropdown-icon">👥</span>
                  <div>
                    <strong>Customer Churn</strong>
                    <span className="dropdown-sub">Classification: RF & XGBoost</span>
                  </div>
                </button>

                <button
                  type="button"
                  className="dropdown-item"
                  onClick={() => {
                    onLoadSample('housing')
                    setShowSampleDropdown(false)
                  }}
                >
                  <span className="dropdown-icon">🏡</span>
                  <div>
                    <strong>Housing Market</strong>
                    <span className="dropdown-sub">Regression: Neural Net & RF</span>
                  </div>
                </button>

                <button
                  type="button"
                  className="dropdown-item"
                  onClick={() => {
                    onLoadSample('sales_ts')
                    setShowSampleDropdown(false)
                  }}
                >
                  <span className="dropdown-icon">📈</span>
                  <div>
                    <strong>Daily Demand & Sales</strong>
                    <span className="dropdown-sub">Time-Series: LSTM Deep Learning</span>
                  </div>
                </button>

                <button
                  type="button"
                  className="dropdown-item"
                  onClick={() => {
                    onLoadSample('customers')
                    setShowSampleDropdown(false)
                  }}
                >
                  <span className="dropdown-icon">🛍️</span>
                  <div>
                    <strong>Customer Segmentation</strong>
                    <span className="dropdown-sub">Unsupervised: K-Means & PCA</span>
                  </div>
                </button>
              </div>
            )}
          </div>

          {/* Reset button if dataset loaded */}
          {hasData && (
            <button type="button" className="btn btn-ghost btn-sm" onClick={onReset} title="Clear current analysis">
              <svg viewBox="0 0 24 24" width="15" height="15" stroke="currentColor" strokeWidth="2" fill="none">
                <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                <path d="M3 3v5h5" />
              </svg>
              <span>New Analysis</span>
            </button>
          )}

          {/* Backend Status Indicator Pill */}
          <div
            className={`status-pill ${backendOnline ? 'status-online' : 'status-offline'}`}
            onClick={() => setShowConfig(!showConfig)}
            title={`Click to configure backend URL. Status: ${backendStatusText}`}
          >
            <span className="status-dot"></span>
            <span className="status-label">
              {backendOnline ? 'FastAPI Online' : 'FastAPI Offline'}
            </span>
            <svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" strokeWidth="2" fill="none" className="status-icon">
              <circle cx="12" cy="12" r="3"></circle>
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
            </svg>
          </div>
        </div>
      </div>

      {/* Backend Configuration Modal/Flyout */}
      {showConfig && (
        <div className="config-popover">
          <div className="config-header">
            <h4>Backend API Configuration</h4>
            <button type="button" className="close-btn" onClick={() => setShowConfig(false)}>×</button>
          </div>
          <form onSubmit={handleSaveConfig} className="config-form">
            <label htmlFor="backend-url-input">FastAPI Server URL:</label>
            <div className="config-input-group">
              <input
                id="backend-url-input"
                type="text"
                value={tempUrl}
                onChange={(e) => setTempUrl(e.target.value)}
                placeholder="http://127.0.0.1:8000"
                className="config-input"
              />
              <button type="submit" className="btn btn-primary btn-sm">Save</button>
            </div>
            <p className="config-hint">
              Default is <code>http://127.0.0.1:8000</code>. To run backend:
              <br />
              <code className="command-snippet">uvicorn app.main:app --reload</code>
            </p>
            <div className="config-actions">
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => {
                  onRefreshHealth()
                }}
              >
                Test Connection
              </button>
            </div>
          </form>
        </div>
      )}
    </header>
  )
}
