import { useEffect, useMemo, useState } from 'react'
import './App.css'
import { Header } from './components/Header'
import { FileUpload } from './components/FileUpload'
import { SummaryCards } from './components/SummaryCards'
import { StatisticsTable } from './components/StatisticsTable'
import { VisualAnalytics } from './components/VisualAnalytics'
import { PredictiveAnalytics } from './components/PredictiveAnalytics'
import { PatternDiscovery } from './components/PatternDiscovery'
import { TimeSeriesForecasting } from './components/TimeSeriesForecasting'
import { DataPreview } from './components/DataPreview'
import { DataQualityAudit } from './components/DataQualityAudit'
import { ExportModal, type ExportTabKey } from './components/ExportModal'
import type { DatasetResult } from './types/analytics'
import type {
  PatternDiscoveryResult,
  PredictiveModelResult,
  TimeSeriesForecastResult,
} from './types/ml'
import {
  checkBackendHealth,
  fetchServerDatasets,
  formatColumnStatistics,
  getBackendBaseUrl,
  selectExistingDataset,
  setBackendBaseUrl,
  uploadDataset,
  type ServerFile,
} from './services/api'
import { createSampleFile, parseCsvText, SAMPLE_DATASETS } from './utils/csvParser'

type MainTab =
  | 'statistics'
  | 'visuals'
  | 'predictive'
  | 'patterns'
  | 'timeseries'
  | 'preview'
  | 'quality'

function App() {
  const [backendUrl, setBackendUrlState] = useState<string>(getBackendBaseUrl())
  const [backendOnline, setBackendOnline] = useState<boolean>(false)
  const [backendStatusText, setBackendStatusText] = useState<string>('Checking backend...')
  const [dataset, setDataset] = useState<DatasetResult | null>(null)
  const [clientPreview, setClientPreview] = useState<{
    headers: string[]
    rows: Record<string, unknown>[]
    totalRows: number
  } | null>(null)
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<MainTab>('statistics')
  const [selectedColumn, setSelectedColumn] = useState<string | null>(null)
  const [isExportOpen, setIsExportOpen] = useState<boolean>(false)
  const [serverFiles, setServerFiles] = useState<ServerFile[]>([])

  // Preserved Machine Learning & Analytics Results Across Tabs
  const [predictiveResult, setPredictiveResult] = useState<PredictiveModelResult | null>(null)
  const [patternResult, setPatternResult] = useState<PatternDiscoveryResult | null>(null)
  const [timeSeriesResult, setTimeSeriesResult] = useState<TimeSeriesForecastResult | null>(null)

  // Verify health on mount and periodic interval
  useEffect(() => {
    let isMounted = true

    const verifyHealth = async () => {
      const res = await checkBackendHealth(backendUrl)
      if (isMounted) {
        setBackendOnline(res.isOnline)
        setBackendStatusText(res.message)
        if (res.isOnline) {
          fetchServerDatasets(backendUrl).then((files) => {
            if (isMounted) setServerFiles(files)
          })
        }
      }
    }

    verifyHealth()
    const interval = setInterval(verifyHealth, 10000)

    return () => {
      isMounted = false
      clearInterval(interval)
    }
  }, [backendUrl])

  const handleUpdateBackendUrl = (newUrl: string) => {
    const trimmed = newUrl.trim().replace(/\/$/, '')
    setBackendBaseUrl(trimmed)
    setBackendUrlState(trimmed)
    checkBackendHealth(trimmed).then((res) => {
      setBackendOnline(res.isOnline)
      setBackendStatusText(res.message)
    })
  }

  const handleFileUpload = async (file: File, autoSelectTab?: MainTab) => {
    setIsLoading(true)
    setError(null)

    if (file.name.toLowerCase().endsWith('.csv')) {
      try {
        // Slice ONLY the first 64KB for fast preview without freezing Chrome on 20MB+ files
        const chunk = file.slice(0, 65536)
        const text = await chunk.text()
        const parsed = parseCsvText(text, 15)
        setClientPreview(parsed)
      } catch {
        // ignore
      }
    } else {
      setClientPreview(null)
    }

    try {
      const result = await uploadDataset(file, backendUrl)
      setDataset(result)

      const statsList = formatColumnStatistics(result.statistics)
      if (statsList.length > 0) {
        setSelectedColumn(statsList[0].column)
      }

      if (autoSelectTab) {
        setActiveTab(autoSelectTab)
      }
    } catch (err: unknown) {
      console.error('Upload error:', err)
      const errorMsg =
        err instanceof Error
          ? err.message
          : 'Failed to upload dataset. Ensure FastAPI backend is running with uvicorn app.main:app --reload'
      setError(errorMsg)
    } finally {
      setIsLoading(false)
    }
  }

  const handleLoadSample = async (key: 'churn' | 'housing' | 'sales_ts' | 'customers') => {
    const sampleDef = SAMPLE_DATASETS[key]
    const file = createSampleFile(key)
    const parsed = parseCsvText(sampleDef.content, 15)
    setClientPreview(parsed)

    let targetTab: MainTab = 'statistics'
    if (sampleDef.recommendedMode === 'predictive') targetTab = 'predictive'
    else if (sampleDef.recommendedMode === 'timeseries') targetTab = 'timeseries'
    else if (sampleDef.recommendedMode === 'patterns') targetTab = 'patterns'

    await handleFileUpload(file, targetTab)
  }

  const handleSelectExisting = async (fileName: string) => {
    setIsLoading(true)
    setError(null)
    setClientPreview(null)

    try {
      const result = await selectExistingDataset(fileName, backendUrl)
      setDataset(result)

      const statsList = formatColumnStatistics(result.statistics)
      if (statsList.length > 0) {
        setSelectedColumn(statsList[0].column)
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to load dataset from server.')
    } finally {
      setIsLoading(false)
    }
  }

  const handleReset = () => {
    setDataset(null)
    setClientPreview(null)
    setError(null)
    setSelectedColumn(null)
    setActiveTab('statistics')
    setPredictiveResult(null)
    setPatternResult(null)
    setTimeSeriesResult(null)
  }

  const columnRows = useMemo(() => {
    if (!dataset?.statistics) return []
    return formatColumnStatistics(dataset.statistics)
  }, [dataset])

  const previewData = useMemo(() => {
    if (dataset?.preview && dataset.preview.length > 0) {
      return {
        rows: dataset.preview,
        columns: dataset.columns_list || Object.keys(dataset.preview[0] || {}),
        totalRows: dataset.rows,
        totalColumns: dataset.columns,
      }
    }
    if (clientPreview && clientPreview.rows.length > 0) {
      return {
        rows: clientPreview.rows,
        columns: clientPreview.headers,
        totalRows: dataset ? dataset.rows : clientPreview.totalRows,
        totalColumns: dataset ? dataset.columns : clientPreview.headers.length,
      }
    }
    return null
  }, [dataset, clientPreview])

  return (
    <div className="app-layout">
      {/* Top Header */}
      <Header
        backendOnline={backendOnline}
        backendStatusText={backendStatusText}
        backendUrl={backendUrl}
        onUpdateBackendUrl={handleUpdateBackendUrl}
        onRefreshHealth={() => {
          checkBackendHealth(backendUrl).then((res) => {
            setBackendOnline(res.isOnline)
            setBackendStatusText(res.message)
          })
        }}
        onLoadSample={handleLoadSample}
        hasData={!!dataset}
        onReset={handleReset}
      />

      <main className="main-content">
        {!dataset ? (
          /* Empty / Upload View */
          <div className="upload-view-wrapper">
            <div className="hero-section">
              <div className="hero-badge">AI, Machine Learning & Deep Learning Engine</div>
              <h2 className="hero-headline">Predictive ML, Pattern Discovery & LSTM Forecasting</h2>
              <p className="hero-subhead">
                Full-stack analytics platform powered by FastAPI & React. Train Random Forest, XGBoost, and Deep Neural Networks,
                discover K-Means clusters and correlation structures, and forecast sequential trends with LSTM.
              </p>
            </div>

            <FileUpload
              onFileUpload={(f) => handleFileUpload(f)}
              onLoadSample={handleLoadSample}
              onSelectExisting={handleSelectExisting}
              serverFiles={serverFiles}
              isLoading={isLoading}
              error={error}
              onClearError={() => setError(null)}
            />
          </div>
        ) : (
          /* Analyzed Dataset Workspace */
          <div className="dashboard-view-wrapper">
            {/* KPI Summary Cards */}
            <SummaryCards dataset={dataset} columnRows={columnRows} />

            {/* Dashboard Navigation Tabs */}
            <div className="dashboard-toolbar">
              <div className="tabs-container">
                <button
                  type="button"
                  className={`main-tab-btn ${activeTab === 'statistics' ? 'main-tab-active' : ''}`}
                  onClick={() => setActiveTab('statistics')}
                >
                  <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" strokeWidth="2" fill="none">
                    <rect x="3" y="3" width="18" height="18" rx="2" />
                    <line x1="3" y1="9" x2="21" y2="9" />
                    <line x1="9" y1="21" x2="9" y2="9" />
                  </svg>
                  <span>Statistics</span>
                  <span className="tab-pill-count">{columnRows.length}</span>
                </button>

                <button
                  type="button"
                  className={`main-tab-btn ${activeTab === 'visuals' ? 'main-tab-active' : ''}`}
                  onClick={() => setActiveTab('visuals')}
                >
                  <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" strokeWidth="2" fill="none">
                    <line x1="18" y1="20" x2="18" y2="10" />
                    <line x1="12" y1="20" x2="12" y2="4" />
                    <line x1="6" y1="20" x2="6" y2="14" />
                  </svg>
                  <span>Distributions</span>
                </button>

                <button
                  type="button"
                  className={`main-tab-btn ${activeTab === 'predictive' ? 'main-tab-active' : ''}`}
                  onClick={() => setActiveTab('predictive')}
                >
                  <span style={{ fontSize: '15px' }}>🌲</span>
                  <span>Predictive ML (RF, XGB, NN)</span>
                </button>

                <button
                  type="button"
                  className={`main-tab-btn ${activeTab === 'patterns' ? 'main-tab-active' : ''}`}
                  onClick={() => setActiveTab('patterns')}
                >
                  <span style={{ fontSize: '15px' }}>🧩</span>
                  <span>Pattern Discovery & Clusters</span>
                </button>

                <button
                  type="button"
                  className={`main-tab-btn ${activeTab === 'timeseries' ? 'main-tab-active' : ''}`}
                  onClick={() => setActiveTab('timeseries')}
                >
                  <span style={{ fontSize: '15px' }}>📈</span>
                  <span>LSTM Forecasting</span>
                </button>

                <button
                  type="button"
                  className={`main-tab-btn ${activeTab === 'preview' ? 'main-tab-active' : ''}`}
                  onClick={() => setActiveTab('preview')}
                >
                  <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" strokeWidth="2" fill="none">
                    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                    <circle cx="12" cy="12" r="3" />
                  </svg>
                  <span>Data Preview</span>
                </button>

                <button
                  type="button"
                  className={`main-tab-btn ${activeTab === 'quality' ? 'main-tab-active' : ''}`}
                  onClick={() => setActiveTab('quality')}
                >
                  <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" strokeWidth="2" fill="none">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                    <polyline points="22 4 12 14.01 9 11.01" />
                  </svg>
                  <span>Data Quality</span>
                </button>
              </div>

              <div className="toolbar-actions">
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  onClick={() => setIsExportOpen(true)}
                >
                  <svg viewBox="0 0 24 24" width="15" height="15" stroke="currentColor" strokeWidth="2" fill="none">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="7 10 12 15 17 10" />
                    <line x1="12" y1="15" x2="12" y2="3" />
                  </svg>
                  <span>Export Report</span>
                </button>
              </div>
            </div>

            {/* Main Tab Content */}
            <div className="tab-content-area">
              {activeTab === 'statistics' && (
                <StatisticsTable
                  rows={columnRows}
                  selectedColumn={selectedColumn}
                  onSelectColumn={(col) => {
                    setSelectedColumn(col)
                    setActiveTab('visuals')
                  }}
                />
              )}

              {activeTab === 'visuals' && (
                <VisualAnalytics
                  rows={columnRows}
                  selectedColumn={selectedColumn}
                  onSelectColumn={setSelectedColumn}
                />
              )}

              {activeTab === 'predictive' && (
                <PredictiveAnalytics
                  key={dataset.file_name}
                  dataset={dataset}
                  backendUrl={backendUrl}
                  initialResult={predictiveResult}
                  onResultTrained={(res) => setPredictiveResult(res)}
                />
              )}

              {activeTab === 'patterns' && (
                <PatternDiscovery
                  key={dataset.file_name}
                  dataset={dataset}
                  backendUrl={backendUrl}
                  initialResult={patternResult}
                  onResultAnalyzed={(res) => setPatternResult(res)}
                />
              )}

              {activeTab === 'timeseries' && (
                <TimeSeriesForecasting
                  key={dataset.file_name}
                  dataset={dataset}
                  backendUrl={backendUrl}
                  initialResult={timeSeriesResult}
                  onResultForecasted={(res) => setTimeSeriesResult(res)}
                />
              )}

              {activeTab === 'preview' && (
                <DataPreview
                  previewRows={previewData?.rows}
                  columnsList={previewData?.columns}
                  totalRows={previewData?.totalRows || dataset.rows}
                  totalColumns={previewData?.totalColumns || dataset.columns}
                />
              )}

              {activeTab === 'quality' && (
                <DataQualityAudit
                  rows={columnRows}
                  totalRows={dataset.rows}
                />
              )}
            </div>
          </div>
        )}
      </main>

      {/* Export Dialog */}
      {dataset && (
        <ExportModal
          dataset={dataset}
          columnRows={columnRows}
          predictiveResult={predictiveResult}
          patternResult={patternResult}
          timeSeriesResult={timeSeriesResult}
          previewRows={previewData?.rows}
          columnsList={previewData?.columns}
          initialTab={activeTab as ExportTabKey}
          isOpen={isExportOpen}
          onClose={() => setIsExportOpen(false)}
        />
      )}
    </div>
  )
}

export default App
