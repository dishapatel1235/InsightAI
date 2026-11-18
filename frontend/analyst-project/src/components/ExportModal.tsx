import React, { useState } from 'react'
import type { ColumnStatRow, DatasetResult } from '../types/analytics'
import type {
  PatternDiscoveryResult,
  PredictiveModelResult,
  TimeSeriesForecastResult,
} from '../types/ml'
import {
  buildMasterReportJson,
  buildMasterReportMarkdown,
  downloadCsv,
  downloadFile,
  downloadJson,
  generateAnomaliesCsv,
  generateClusterSegmentsCsv,
  generateCorrelationMatrixCsv,
  generateDataPreviewCsv,
  generateDataQualityCsv,
  generateDistributionsCsv,
  generateForecastMarkdown,
  generateForecastTimelineCsv,
  generateMlFeatureImportanceCsv,
  generateMlPredictionsCsv,
  generateMlSummaryMarkdown,
  generateStatisticsCsv,
  generateStatisticsMarkdown,
} from '../utils/exportUtils'

export type ExportTabKey =
  | 'all'
  | 'statistics'
  | 'visuals'
  | 'predictive'
  | 'patterns'
  | 'timeseries'
  | 'quality'
  | 'preview'

interface ExportModalProps {
  dataset: DatasetResult
  columnRows: ColumnStatRow[]
  predictiveResult?: PredictiveModelResult | null
  patternResult?: PatternDiscoveryResult | null
  timeSeriesResult?: TimeSeriesForecastResult | null
  previewRows?: Record<string, unknown>[]
  columnsList?: string[]
  initialTab?: ExportTabKey
  isOpen: boolean
  onClose: () => void
}

export const ExportModal: React.FC<ExportModalProps> = ({
  dataset,
  columnRows,
  predictiveResult,
  patternResult,
  timeSeriesResult,
  previewRows,
  columnsList,
  initialTab = 'all',
  isOpen,
  onClose,
}) => {
  const [activeTab, setActiveTab] = useState<ExportTabKey>(initialTab)
  const [feedback, setFeedback] = useState<string | null>(null)

  if (!isOpen) return null

  const baseFileName = dataset.file_name.replace(/\.[^/.]+$/, '')

  const showFeedback = (msg: string) => {
    setFeedback(msg)
    setTimeout(() => setFeedback(null), 3000)
  }

  // ==========================================
  // TAB EXPORT HANDLERS
  // ==========================================

  // 1. Statistics
  const handleExportStatisticsCsv = () => {
    const { headers, rows } = generateStatisticsCsv(columnRows)
    downloadCsv(headers, rows, `${baseFileName}_descriptive_statistics.csv`)
    showFeedback('Downloaded Statistics CSV!')
  }

  const handleExportStatisticsJson = () => {
    downloadJson(dataset.statistics, `${baseFileName}_statistics.json`)
    showFeedback('Downloaded Statistics JSON!')
  }

  const handleCopyStatisticsMd = () => {
    const md = generateStatisticsMarkdown(dataset, columnRows)
    navigator.clipboard.writeText(md)
    showFeedback('Copied Statistics Markdown to clipboard!')
  }

  // 2. Distributions
  const handleExportDistributionsCsv = () => {
    const { headers, rows } = generateDistributionsCsv(columnRows)
    downloadCsv(headers, rows, `${baseFileName}_distributions_analysis.csv`)
    showFeedback('Downloaded Distributions CSV!')
  }

  const handleExportDistributionsJson = () => {
    const data = columnRows.map((r) => ({
      column: r.column,
      mean: r.mean,
      median: r.median,
      std: r.std,
      min: r.min,
      max: r.max,
      skewness: r.skewness,
      kurtosis: r.kurtosis,
    }))
    downloadJson(data, `${baseFileName}_distributions.json`)
    showFeedback('Downloaded Distributions JSON!')
  }

  // 3. Predictive ML
  const handleExportMlPredictionsCsv = () => {
    if (!predictiveResult) return
    const res = generateMlPredictionsCsv(predictiveResult)
    if (res) {
      downloadCsv(res.headers, res.rows, `${baseFileName}_ml_predictions.csv`)
      showFeedback('Downloaded ML Predictions CSV!')
    }
  }

  const handleExportMlFeatureImportanceCsv = () => {
    if (!predictiveResult) return
    const res = generateMlFeatureImportanceCsv(predictiveResult)
    if (res) {
      downloadCsv(res.headers, res.rows, `${baseFileName}_feature_importances.csv`)
      showFeedback('Downloaded Feature Importances CSV!')
    }
  }

  const handleExportMlModelJson = () => {
    if (!predictiveResult) return
    downloadJson(predictiveResult, `${baseFileName}_predictive_model_${predictiveResult.model_name}.json`)
    showFeedback('Downloaded ML Model JSON!')
  }

  const handleCopyMlMd = () => {
    if (!predictiveResult) return
    const md = generateMlSummaryMarkdown(predictiveResult)
    navigator.clipboard.writeText(md)
    showFeedback('Copied ML Summary Markdown!')
  }

  // 4. Pattern Discovery
  const handleExportClustersCsv = () => {
    if (!patternResult) return
    const res = generateClusterSegmentsCsv(patternResult)
    if (res) {
      downloadCsv(res.headers, res.rows, `${baseFileName}_cluster_segments.csv`)
      showFeedback('Downloaded Cluster Segments CSV!')
    }
  }

  const handleExportCorrelationCsv = () => {
    if (!patternResult) return
    const res = generateCorrelationMatrixCsv(patternResult)
    if (res) {
      downloadCsv(res.headers, res.rows, `${baseFileName}_correlation_matrix.csv`)
      showFeedback('Downloaded Correlation Matrix CSV!')
    }
  }

  const handleExportAnomaliesCsv = () => {
    if (!patternResult) return
    const res = generateAnomaliesCsv(patternResult)
    if (res) {
      downloadCsv(res.headers, res.rows, `${baseFileName}_detected_anomalies.csv`)
      showFeedback('Downloaded Anomalies CSV!')
    }
  }

  const handleExportPatternsJson = () => {
    if (!patternResult) return
    downloadJson(patternResult, `${baseFileName}_pattern_discovery.json`)
    showFeedback('Downloaded Pattern Discovery JSON!')
  }

  // 5. Time-Series Forecasting
  const handleExportForecastTimelineCsv = () => {
    if (!timeSeriesResult) return
    const res = generateForecastTimelineCsv(timeSeriesResult)
    if (res) {
      downloadCsv(res.headers, res.rows, `${baseFileName}_forecast_timeline.csv`)
      showFeedback('Downloaded Forecast Timeline CSV!')
    }
  }

  const handleExportForecastJson = () => {
    if (!timeSeriesResult) return
    downloadJson(timeSeriesResult, `${baseFileName}_time_series_forecast.json`)
    showFeedback('Downloaded Forecast JSON!')
  }

  const handleCopyForecastMd = () => {
    if (!timeSeriesResult) return
    const md = generateForecastMarkdown(timeSeriesResult)
    navigator.clipboard.writeText(md)
    showFeedback('Copied Forecast Markdown!')
  }

  // 6. Data Quality
  const handleExportQualityCsv = () => {
    const { headers, rows } = generateDataQualityCsv(columnRows, dataset.rows)
    downloadCsv(headers, rows, `${baseFileName}_data_quality_audit.csv`)
    showFeedback('Downloaded Data Quality CSV!')
  }

  const handleExportQualityJson = () => {
    const qualitySummary = {
      file_name: dataset.file_name,
      total_rows: dataset.rows,
      columns_count: columnRows.length,
      audit: columnRows.map((r) => ({
        column: r.column,
        missing_count: r.missing_values,
        missing_pct: dataset.rows > 0 ? (r.missing_values / dataset.rows) * 100 : 0,
        skewness: r.skewness,
        kurtosis: r.kurtosis,
      })),
    }
    downloadJson(qualitySummary, `${baseFileName}_data_quality.json`)
    showFeedback('Downloaded Data Quality JSON!')
  }

  // 7. Data Preview
  const handleExportPreviewCsv = () => {
    const pRows = previewRows || dataset.preview || []
    const pCols = columnsList || dataset.columns_list || (pRows[0] ? Object.keys(pRows[0]) : [])
    if (pRows.length === 0 || pCols.length === 0) {
      showFeedback('No preview data available to export.')
      return
    }
    const { headers, rows } = generateDataPreviewCsv(pRows, pCols)
    downloadCsv(headers, rows, `${baseFileName}_preview_data.csv`)
    showFeedback('Downloaded Data Preview CSV!')
  }

  const handleExportPreviewJson = () => {
    const pRows = previewRows || dataset.preview || []
    downloadJson(pRows, `${baseFileName}_preview_data.json`)
    showFeedback('Downloaded Data Preview JSON!')
  }

  // ==========================================
  // MASTER ENTIRE REPORT HANDLERS
  // ==========================================

  const handleDownloadMasterJson = () => {
    const masterObj = buildMasterReportJson({
      dataset,
      columnRows,
      predictiveResult,
      patternResult,
      timeSeriesResult,
      previewRows: previewRows || dataset.preview,
      columnsList: columnsList || dataset.columns_list,
    })
    downloadJson(masterObj, `${baseFileName}_complete_intelligence_report.json`)
    showFeedback('Downloaded Entire Master JSON Report!')
  }

  const handleDownloadMasterMd = () => {
    const md = buildMasterReportMarkdown({
      dataset,
      columnRows,
      predictiveResult,
      patternResult,
      timeSeriesResult,
    })
    downloadFile(md, `${baseFileName}_executive_master_report.md`, 'text/markdown;charset=utf-8;')
    showFeedback('Downloaded Executive Master Markdown Report!')
  }

  const handleCopyMasterMd = () => {
    const md = buildMasterReportMarkdown({
      dataset,
      columnRows,
      predictiveResult,
      patternResult,
      timeSeriesResult,
    })
    navigator.clipboard.writeText(md)
    showFeedback('Copied Master Markdown Report to clipboard!')
  }

  const handleDownloadAllCsvs = () => {
    let delay = 0
    const step = 200

    // 1. Statistics
    setTimeout(() => handleExportStatisticsCsv(), delay)
    delay += step

    // 2. Distributions
    setTimeout(() => handleExportDistributionsCsv(), delay)
    delay += step

    // 3. Data Quality
    setTimeout(() => handleExportQualityCsv(), delay)
    delay += step

    // 4. ML Predictions & Features (if available)
    if (predictiveResult) {
      setTimeout(() => handleExportMlPredictionsCsv(), delay)
      delay += step
      setTimeout(() => handleExportMlFeatureImportanceCsv(), delay)
      delay += step
    }

    // 5. Pattern Discovery (if available)
    if (patternResult?.clustering) {
      setTimeout(() => handleExportClustersCsv(), delay)
      delay += step
    }
    if (patternResult?.correlation) {
      setTimeout(() => handleExportCorrelationCsv(), delay)
      delay += step
    }
    if (patternResult?.anomaly_detection) {
      setTimeout(() => handleExportAnomaliesCsv(), delay)
      delay += step
    }

    // 6. Time-Series (if available)
    if (timeSeriesResult) {
      setTimeout(() => handleExportForecastTimelineCsv(), delay)
      delay += step
    }

    // 7. Preview
    setTimeout(() => handleExportPreviewCsv(), delay)

    showFeedback('Initiated download of all tabular CSV matrices!')
  }

  const handlePrint = () => {
    window.print()
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-dialog export-modal-expanded" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="modal-header">
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h3 className="modal-title">Export Analytics & Intelligence Reports</h3>
              <span className="brand-badge" style={{ fontSize: '10px' }}>Multi-Tab Engine</span>
            </div>
            <p className="modal-subtitle">
              Download individual tab outcomes, tabular CSV matrices, or the entire unified master report
            </p>
          </div>
          <button type="button" className="close-btn" onClick={onClose}>×</button>
        </div>

        {/* Global Feedback Banner */}
        {feedback && (
          <div className="export-feedback-toast">
            <span>✓ {feedback}</span>
          </div>
        )}

        {/* Navigation Selector for Individual Tabs & Entire Report */}
        <div className="export-tab-bar">
          <button
            type="button"
            className={`export-tab-pill ${activeTab === 'all' ? 'active' : ''}`}
            onClick={() => setActiveTab('all')}
          >
            <span>⭐ Entire Master Report</span>
          </button>

          <button
            type="button"
            className={`export-tab-pill ${activeTab === 'statistics' ? 'active' : ''}`}
            onClick={() => setActiveTab('statistics')}
          >
            <span>📊 Statistics</span>
            <span className="export-badge">{columnRows.length}</span>
          </button>

          <button
            type="button"
            className={`export-tab-pill ${activeTab === 'visuals' ? 'active' : ''}`}
            onClick={() => setActiveTab('visuals')}
          >
            <span>📈 Distributions</span>
          </button>

          <button
            type="button"
            className={`export-tab-pill ${activeTab === 'predictive' ? 'active' : ''}`}
            onClick={() => setActiveTab('predictive')}
          >
            <span>🌲 Predictive ML</span>
            {predictiveResult && <span className="export-badge badge-active">Ready</span>}
          </button>

          <button
            type="button"
            className={`export-tab-pill ${activeTab === 'patterns' ? 'active' : ''}`}
            onClick={() => setActiveTab('patterns')}
          >
            <span>🧩 Patterns</span>
            {patternResult && <span className="export-badge badge-active">Ready</span>}
          </button>

          <button
            type="button"
            className={`export-tab-pill ${activeTab === 'timeseries' ? 'active' : ''}`}
            onClick={() => setActiveTab('timeseries')}
          >
            <span>📉 Forecasting</span>
            {timeSeriesResult && <span className="export-badge badge-active">Ready</span>}
          </button>

          <button
            type="button"
            className={`export-tab-pill ${activeTab === 'quality' ? 'active' : ''}`}
            onClick={() => setActiveTab('quality')}
          >
            <span>🛡️ Data Quality</span>
          </button>

          <button
            type="button"
            className={`export-tab-pill ${activeTab === 'preview' ? 'active' : ''}`}
            onClick={() => setActiveTab('preview')}
          >
            <span>👁️ Data Preview</span>
          </button>
        </div>

        {/* Modal Body */}
        <div className="modal-body export-modal-body">
          {/* =========================================================
              VIEW: ENTIRE MASTER REPORT
          ========================================================= */}
          {activeTab === 'all' && (
            <div className="export-content-pane">
              <div className="master-export-hero">
                <div className="hero-text">
                  <h4>Complete Multi-Module Master Report</h4>
                  <p>
                    Bundles all descriptive statistics, data quality findings, ML model outcomes, pattern discoveries,
                    and time-series trajectories for <strong>{dataset.file_name}</strong> into a single comprehensive package.
                  </p>
                </div>
                <div className="master-hero-actions">
                  <button type="button" className="btn btn-primary" onClick={handleDownloadMasterJson}>
                    ⭐ Download Master JSON
                  </button>
                  <button type="button" className="btn btn-secondary" onClick={handleDownloadMasterMd}>
                    📄 Executive Brief (.md)
                  </button>
                </div>
              </div>

              <div className="export-options-grid">
                <div className="export-option-card" onClick={handleDownloadMasterJson}>
                  <div className="export-icon json-icon">JSON</div>
                  <div className="export-info">
                    <h4>Master Intelligence JSON</h4>
                    <p>
                      Exhaustive JSON document containing dataset metadata, 11 descriptive statistics, distribution metrics,
                      data quality audits, ML evaluation parameters, cluster profiles, and forecast trajectories.
                    </p>
                  </div>
                  <button type="button" className="btn btn-secondary btn-sm">Download .json</button>
                </div>

                <div className="export-option-card" onClick={handleDownloadMasterMd}>
                  <div className="export-icon md-icon">MD</div>
                  <div className="export-info">
                    <h4>Executive Master Brief (Markdown)</h4>
                    <p>
                      Publication-ready multi-section report with executive overview, descriptive statistics table,
                      data quality notice, and ML/forecasting outcomes.
                    </p>
                  </div>
                  <button type="button" className="btn btn-secondary btn-sm">Download .md</button>
                </div>

                <div className="export-option-card" onClick={handleDownloadAllCsvs}>
                  <div className="export-icon csv-icon">CSVs</div>
                  <div className="export-info">
                    <h4>Download All Available CSVs</h4>
                    <p>
                      Sequentially downloads tabular CSV files for all modules: Statistics, Distributions, Data Quality,
                      Predictions (if trained), Clusters (if run), and Forecasting.
                    </p>
                  </div>
                  <button type="button" className="btn btn-secondary btn-sm">Download All .csv</button>
                </div>

                <div className="export-option-card" onClick={handlePrint}>
                  <div className="export-icon print-icon">🖨️</div>
                  <div className="export-info">
                    <h4>Print / Save as PDF</h4>
                    <p>Opens browser print dialog formatted for saving as an executive PDF document.</p>
                  </div>
                  <button type="button" className="btn btn-secondary btn-sm">Print / PDF</button>
                </div>
              </div>
            </div>
          )}

          {/* =========================================================
              VIEW: DESCRIPTIVE STATISTICS
          ========================================================= */}
          {activeTab === 'statistics' && (
            <div className="export-content-pane">
              <div className="tab-export-header">
                <h4>Descriptive Statistics Tab Outcome</h4>
                <p>11 computed statistical indicators across {columnRows.length} numeric columns</p>
              </div>

              <div className="export-options-grid">
                <div className="export-option-card" onClick={handleExportStatisticsCsv}>
                  <div className="export-icon csv-icon">CSV</div>
                  <div className="export-info">
                    <h4>Descriptive Statistics Matrix (.csv)</h4>
                    <p>Column, Mean, Median, Std Dev, Variance, Min, Max, Sum, Count, Missing, Skewness, Kurtosis.</p>
                  </div>
                  <button type="button" className="btn btn-secondary btn-sm">Download .csv</button>
                </div>

                <div className="export-option-card" onClick={handleExportStatisticsJson}>
                  <div className="export-icon json-icon">JSON</div>
                  <div className="export-info">
                    <h4>Statistics Response JSON (.json)</h4>
                    <p>Raw calculated statistical values organized per feature.</p>
                  </div>
                  <button type="button" className="btn btn-secondary btn-sm">Download .json</button>
                </div>

                <div className="export-option-card" onClick={handleCopyStatisticsMd}>
                  <div className="export-icon md-icon">MD</div>
                  <div className="export-info">
                    <h4>Copy Statistics Markdown Table</h4>
                    <p>Clean Markdown table ready to paste into GitHub, Notion, or Slack.</p>
                  </div>
                  <button type="button" className="btn btn-secondary btn-sm">Copy Table</button>
                </div>
              </div>
            </div>
          )}

          {/* =========================================================
              VIEW: DISTRIBUTIONS & VISUALS
          ========================================================= */}
          {activeTab === 'visuals' && (
            <div className="export-content-pane">
              <div className="tab-export-header">
                <h4>Visual Analytics & Distributions Tab Outcome</h4>
                <p>Distribution spreads, spread range (Max-Min), skewness symmetry, and tail classifications</p>
              </div>

              <div className="export-options-grid">
                <div className="export-option-card" onClick={handleExportDistributionsCsv}>
                  <div className="export-icon csv-icon">CSV</div>
                  <div className="export-info">
                    <h4>Distributions Analysis CSV (.csv)</h4>
                    <p>Detailed distribution table with spread ranges, asymmetry classifications, and tail weight indicators.</p>
                  </div>
                  <button type="button" className="btn btn-secondary btn-sm">Download .csv</button>
                </div>

                <div className="export-option-card" onClick={handleExportDistributionsJson}>
                  <div className="export-icon json-icon">JSON</div>
                  <div className="export-info">
                    <h4>Distribution Metrics JSON (.json)</h4>
                    <p>Structured array of distribution parameters for charting or pipelines.</p>
                  </div>
                  <button type="button" className="btn btn-secondary btn-sm">Download .json</button>
                </div>
              </div>
            </div>
          )}

          {/* =========================================================
              VIEW: PREDICTIVE MACHINE LEARNING
          ========================================================= */}
          {activeTab === 'predictive' && (
            <div className="export-content-pane">
              <div className="tab-export-header">
                <h4>Supervised Predictive Modeling Tab Outcome</h4>
                <p>
                  {predictiveResult
                    ? `Trained: ${predictiveResult.model_name} on target '${predictiveResult.target_column}' (${predictiveResult.task_type})`
                    : 'Train a model in the Predictive ML tab to unlock full prediction exports'}
                </p>
              </div>

              {predictiveResult ? (
                <div className="export-options-grid">
                  <div className="export-option-card" onClick={handleExportMlPredictionsCsv}>
                    <div className="export-icon csv-icon">CSV</div>
                    <div className="export-info">
                      <h4>Model Predictions & Residuals (.csv)</h4>
                      <p>
                        {predictiveResult.task_type === 'classification'
                          ? 'Sample predictions, actual vs. predicted labels, and classification accuracy verification.'
                          : 'Sample predictions, actual continuous values, model predictions, and residual errors.'}
                      </p>
                    </div>
                    <button type="button" className="btn btn-secondary btn-sm">Download .csv</button>
                  </div>

                  <div className="export-option-card" onClick={handleExportMlFeatureImportanceCsv}>
                    <div className="export-icon csv-icon">CSV</div>
                    <div className="export-info">
                      <h4>Feature Importance Rankings (.csv)</h4>
                      <p>Ranked table of predictor feature importance scores and normalized percentage weights.</p>
                    </div>
                    <button type="button" className="btn btn-secondary btn-sm">Download .csv</button>
                  </div>

                  <div className="export-option-card" onClick={handleExportMlModelJson}>
                    <div className="export-icon json-icon">JSON</div>
                    <div className="export-info">
                      <h4>Trained Model Evaluation JSON (.json)</h4>
                      <p>Complete model architecture, accuracy, R², RMSE, confusion matrix, and feature mappings.</p>
                    </div>
                    <button type="button" className="btn btn-secondary btn-sm">Download .json</button>
                  </div>

                  <div className="export-option-card" onClick={handleCopyMlMd}>
                    <div className="export-icon md-icon">MD</div>
                    <div className="export-info">
                      <h4>Copy ML Executive Brief (Markdown)</h4>
                      <p>Formatted executive brief summarizing metrics, confusion matrix/R², and top features.</p>
                    </div>
                    <button type="button" className="btn btn-secondary btn-sm">Copy MD</button>
                  </div>
                </div>
              ) : (
                <div className="empty-export-box">
                  <div className="empty-icon">🌲</div>
                  <h4>No Predictive Model Trained Yet</h4>
                  <p>
                    Navigate to the <strong>Predictive ML</strong> tab, select your target variable, and click{' '}
                    <strong>Train & Evaluate Model</strong>. Your trained model outcomes, predictions, and feature
                    importances will automatically appear here for export.
                  </p>
                </div>
              )}
            </div>
          )}

          {/* =========================================================
              VIEW: PATTERN DISCOVERY & CLUSTERS
          ========================================================= */}
          {activeTab === 'patterns' && (
            <div className="export-content-pane">
              <div className="tab-export-header">
                <h4>Pattern Discovery & Clustering Tab Outcome</h4>
                <p>
                  {patternResult
                    ? `Analysis complete across ${patternResult.features_analyzed.length} numeric features`
                    : 'Execute K-Means, Correlation, or Anomaly detection in the Pattern Discovery tab to unlock exports'}
                </p>
              </div>

              {patternResult ? (
                <div className="export-options-grid">
                  {patternResult.clustering && (
                    <div className="export-option-card" onClick={handleExportClustersCsv}>
                      <div className="export-icon csv-icon">CSV</div>
                      <div className="export-info">
                        <h4>Cluster Profiles & Centroids (.csv)</h4>
                        <p>Segment names, sizes, cluster percentages, 2D PCA centroid coordinates, and mean feature values.</p>
                      </div>
                      <button type="button" className="btn btn-secondary btn-sm">Download .csv</button>
                    </div>
                  )}

                  {patternResult.correlation && (
                    <div className="export-option-card" onClick={handleExportCorrelationCsv}>
                      <div className="export-icon csv-icon">CSV</div>
                      <div className="export-info">
                        <h4>Pairwise Correlation Matrix (.csv)</h4>
                        <p>Full N × N Pearson correlation matrix between all profiled features.</p>
                      </div>
                      <button type="button" className="btn btn-secondary btn-sm">Download .csv</button>
                    </div>
                  )}

                  {patternResult.anomaly_detection && (
                    <div className="export-option-card" onClick={handleExportAnomaliesCsv}>
                      <div className="export-icon csv-icon">CSV</div>
                      <div className="export-info">
                        <h4>Detected Anomalies Table (.csv)</h4>
                        <p>List of identified anomalous records with anomaly severity scores and key feature values.</p>
                      </div>
                      <button type="button" className="btn btn-secondary btn-sm">Download .csv</button>
                    </div>
                  )}

                  <div className="export-option-card" onClick={handleExportPatternsJson}>
                    <div className="export-icon json-icon">JSON</div>
                    <div className="export-info">
                      <h4>Full Pattern Discovery JSON (.json)</h4>
                      <p>Complete JSON response including cluster inertia, silhouette scores, 2D PCA points, and correlations.</p>
                    </div>
                    <button type="button" className="btn btn-secondary btn-sm">Download .json</button>
                  </div>
                </div>
              ) : (
                <div className="empty-export-box">
                  <div className="empty-icon">🧩</div>
                  <h4>No Pattern Analysis Run Yet</h4>
                  <p>
                    Switch to the <strong>Pattern Discovery & Clusters</strong> tab and click{' '}
                    <strong>Run Pattern Discovery</strong>. Natural clusters, correlation matrices, and detected anomalies
                    will automatically be available here for instant export.
                  </p>
                </div>
              )}
            </div>
          )}

          {/* =========================================================
              VIEW: TIME-SERIES FORECASTING
          ========================================================= */}
          {activeTab === 'timeseries' && (
            <div className="export-content-pane">
              <div className="tab-export-header">
                <h4>LSTM & Sequential Time-Series Forecasting Tab Outcome</h4>
                <p>
                  {timeSeriesResult
                    ? `Forecasted '${timeSeriesResult.target_column}' using ${timeSeriesResult.model_name} (${timeSeriesResult.horizon} steps ahead)`
                    : 'Generate a forecast in the LSTM Forecasting tab to unlock timeline exports'}
                </p>
              </div>

              {timeSeriesResult ? (
                <div className="export-options-grid">
                  <div className="export-option-card" onClick={handleExportForecastTimelineCsv}>
                    <div className="export-icon csv-icon">CSV</div>
                    <div className="export-info">
                      <h4>Forecast Timeline & 95% Confidence Bounds (.csv)</h4>
                      <p>Historical observed values, fitted model values, future forecast points, and lower/upper confidence bounds.</p>
                    </div>
                    <button type="button" className="btn btn-secondary btn-sm">Download .csv</button>
                  </div>

                  <div className="export-option-card" onClick={handleExportForecastJson}>
                    <div className="export-icon json-icon">JSON</div>
                    <div className="export-info">
                      <h4>Forecast Trajectory JSON (.json)</h4>
                      <p>Complete timeline array with MAE, RMSE, MAPE accuracy metrics, and horizon steps.</p>
                    </div>
                    <button type="button" className="btn btn-secondary btn-sm">Download .json</button>
                  </div>

                  <div className="export-option-card" onClick={handleCopyForecastMd}>
                    <div className="export-icon md-icon">MD</div>
                    <div className="export-info">
                      <h4>Copy Forecasting Brief (Markdown)</h4>
                      <p>Formatted table of forecasted steps and confidence bounds.</p>
                    </div>
                    <button type="button" className="btn btn-secondary btn-sm">Copy MD</button>
                  </div>
                </div>
              ) : (
                <div className="empty-export-box">
                  <div className="empty-icon">📈</div>
                  <h4>No Forecast Generated Yet</h4>
                  <p>
                    Switch to the <strong>LSTM Forecasting</strong> tab, select your sequential target column, and click{' '}
                    <strong>Generate Forecast</strong>. The future projection timeline and 95% confidence bounds will
                    appear here for export.
                  </p>
                </div>
              )}
            </div>
          )}

          {/* =========================================================
              VIEW: DATA QUALITY AUDIT
          ========================================================= */}
          {activeTab === 'quality' && (
            <div className="export-content-pane">
              <div className="tab-export-header">
                <h4>Data Quality & Hygiene Audit Tab Outcome</h4>
                <p>Completeness scores, missing value counts, and outlier/skewness risk assessment</p>
              </div>

              <div className="export-options-grid">
                <div className="export-option-card" onClick={handleExportQualityCsv}>
                  <div className="export-icon csv-icon">CSV</div>
                  <div className="export-info">
                    <h4>Data Quality Audit Matrix (.csv)</h4>
                    <p>Column-by-column missing value counts, missing percentages, skewness risk, and health grades.</p>
                  </div>
                  <button type="button" className="btn btn-secondary btn-sm">Download .csv</button>
                </div>

                <div className="export-option-card" onClick={handleExportQualityJson}>
                  <div className="export-icon json-icon">JSON</div>
                  <div className="export-info">
                    <h4>Data Quality Audit JSON (.json)</h4>
                    <p>Structured audit object for automated validation pipelines or documentation.</p>
                  </div>
                  <button type="button" className="btn btn-secondary btn-sm">Download .json</button>
                </div>
              </div>
            </div>
          )}

          {/* =========================================================
              VIEW: DATA PREVIEW
          ========================================================= */}
          {activeTab === 'preview' && (
            <div className="export-content-pane">
              <div className="tab-export-header">
                <h4>Data Preview Tab Outcome</h4>
                <p>Export preview sample records from {dataset.file_name}</p>
              </div>

              <div className="export-options-grid">
                <div className="export-option-card" onClick={handleExportPreviewCsv}>
                  <div className="export-icon csv-icon">CSV</div>
                  <div className="export-info">
                    <h4>Tabular Data Preview (.csv)</h4>
                    <p>Standard comma-delimited export of the loaded preview records.</p>
                  </div>
                  <button type="button" className="btn btn-secondary btn-sm">Download .csv</button>
                </div>

                <div className="export-option-card" onClick={handleExportPreviewJson}>
                  <div className="export-icon json-icon">JSON</div>
                  <div className="export-info">
                    <h4>Data Records JSON (.json)</h4>
                    <p>JSON array of row objects.</p>
                  </div>
                  <button type="button" className="btn btn-secondary btn-sm">Download .json</button>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="modal-footer" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <button type="button" className="btn btn-secondary btn-sm" onClick={handleCopyMasterMd}>
              📋 Copy Master Report (MD)
            </button>
            <button type="button" className="btn btn-secondary btn-sm" onClick={handlePrint}>
              🖨️ Print / PDF
            </button>
          </div>
          <button type="button" className="btn btn-ghost" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  )
}
