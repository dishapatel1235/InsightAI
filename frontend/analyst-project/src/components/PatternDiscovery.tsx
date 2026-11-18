import React, { useState } from 'react'
import type { DatasetResult } from '../types/analytics'
import type { PatternDiscoveryResult, PatternMethod } from '../types/ml'
import { analyzePatterns, formatNumber } from '../services/api'
import {
  downloadCsv,
  downloadJson,
  generateAnomaliesCsv,
  generateClusterSegmentsCsv,
  generateCorrelationMatrixCsv,
} from '../utils/exportUtils'

interface PatternDiscoveryProps {
  dataset: DatasetResult
  backendUrl: string
  initialResult?: PatternDiscoveryResult | null
  onResultAnalyzed?: (res: PatternDiscoveryResult) => void
}

export const PatternDiscovery: React.FC<PatternDiscoveryProps> = ({
  dataset,
  backendUrl,
  initialResult,
  onResultAnalyzed,
}) => {
  const [method, setMethod] = useState<PatternMethod | 'all'>('clustering')
  const [nClusters, setNClusters] = useState<number>(3)
  const [contamination, setContamination] = useState<number>(0.05)
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<PatternDiscoveryResult | null>(initialResult || null)

  const handleRunAnalysis = async () => {
    setIsLoading(true)
    setError(null)
    try {
      const res = await analyzePatterns({
        fileName: dataset.file_name,
        method,
        nClusters,
        contamination,
        baseUrl: backendUrl,
      })
      setResult(res)
      onResultAnalyzed?.(res)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Pattern discovery failed.')
    } finally {
      setIsLoading(false)
    }
  }

  // Predefined cluster colors
  const clusterColors = ['#6366f1', '#10b981', '#f59e0b', '#ec4899', '#06b6d4', '#8b5cf6', '#ef4444', '#14b8a6']

  return (
    <div className="ml-module-container">
      {/* Pattern Discovery Controller Card */}
      <div className="card ml-config-card">
        <div className="card-header-bar">
          <div>
            <h3 className="card-title">Unsupervised Pattern Discovery & Clustering</h3>
            <p className="card-subtitle">
              Discover natural data clusters, compute 2D PCA embeddings, correlation matrices, and isolate anomalies
            </p>
          </div>

          <div className="task-type-toggle">
            <button
              type="button"
              className={`pill-btn ${method === 'clustering' ? 'pill-active' : ''}`}
              onClick={() => setMethod('clustering')}
            >
              K-Means Clusters
            </button>
            <button
              type="button"
              className={`pill-btn ${method === 'correlation' ? 'pill-active' : ''}`}
              onClick={() => setMethod('correlation')}
            >
              Correlation Matrix
            </button>
            <button
              type="button"
              className={`pill-btn ${method === 'anomaly' ? 'pill-active' : ''}`}
              onClick={() => setMethod('anomaly')}
            >
              Anomaly Detection
            </button>
          </div>
        </div>

        {error && (
          <div className="error-banner" style={{ margin: '16px 20px 0' }}>
            <span>{error}</span>
            <button type="button" className="close-btn" onClick={() => setError(null)}>×</button>
          </div>
        )}

        <div className="pattern-ctrl-grid">
          {method === 'clustering' && (
            <div className="param-item">
              <label>Number of Clusters ($k$): <strong>{nClusters}</strong></label>
              <input
                type="range"
                min={2}
                max={8}
                value={nClusters}
                onChange={(e) => setNClusters(Number(e.target.value))}
              />
            </div>
          )}

          {method === 'anomaly' && (
            <div className="param-item">
              <label>Anomaly Contamination Rate: <strong>{(contamination * 100).toFixed(0)}%</strong></label>
              <input
                type="range"
                min={0.01}
                max={0.2}
                step={0.01}
                value={contamination}
                onChange={(e) => setContamination(Number(e.target.value))}
              />
            </div>
          )}

          <div className="pattern-action-box">
            <button
              type="button"
              className="btn btn-primary"
              disabled={isLoading}
              onClick={handleRunAnalysis}
            >
              {isLoading ? (
                <>
                  <span className="spinner spinner-sm"></span>
                  <span>Discovering Patterns...</span>
                </>
              ) : (
                <>
                  <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" strokeWidth="2" fill="none">
                    <circle cx="12" cy="12" r="10" />
                    <line x1="12" y1="16" x2="12" y2="12" />
                    <line x1="12" y1="8" x2="12.01" y2="8" />
                  </svg>
                  <span>Execute Pattern Discovery</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Results View */}
      {result && (
        <div className="ml-results-wrapper">
          {/* Quick Tab Export Bar */}
          <div className="results-header-row">
            <div>
              <h3 style={{ fontSize: '16px', fontWeight: 700, margin: 0, color: 'var(--text-main)' }}>
                Pattern Discovery & Clustering Outcome
              </h3>
              <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                Profiled {result.features_analyzed.length} numeric features across {result.total_samples} samples
              </span>
            </div>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              {result.clustering && (
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  title="Download cluster segments, centroids, and feature means"
                  onClick={() => {
                    const cRes = generateClusterSegmentsCsv(result)
                    if (cRes) {
                      downloadCsv(cRes.headers, cRes.rows, `${dataset.file_name.replace(/\.[^/.]+$/, '')}_cluster_segments.csv`)
                    }
                  }}
                >
                  <svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" strokeWidth="2" fill="none">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="7 10 12 15 17 10" />
                    <line x1="12" y1="15" x2="12" y2="3" />
                  </svg>
                  <span>Clusters (.csv)</span>
                </button>
              )}

              {result.correlation && (
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  title="Download pairwise correlation matrix"
                  onClick={() => {
                    const mRes = generateCorrelationMatrixCsv(result)
                    if (mRes) {
                      downloadCsv(mRes.headers, mRes.rows, `${dataset.file_name.replace(/\.[^/.]+$/, '')}_correlation_matrix.csv`)
                    }
                  }}
                >
                  <span>Correlation (.csv)</span>
                </button>
              )}

              {result.anomaly_detection && (
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  title="Download detected anomalies"
                  onClick={() => {
                    const aRes = generateAnomaliesCsv(result)
                    if (aRes) {
                      downloadCsv(aRes.headers, aRes.rows, `${dataset.file_name.replace(/\.[^/.]+$/, '')}_anomalies.csv`)
                    }
                  }}
                >
                  <span>Anomalies (.csv)</span>
                </button>
              )}

              <button
                type="button"
                className="btn btn-secondary btn-sm"
                title="Download full pattern discovery JSON"
                onClick={() => {
                  downloadJson(result, `${dataset.file_name.replace(/\.[^/.]+$/, '')}_pattern_discovery.json`)
                }}
              >
                <span>Patterns (.json)</span>
              </button>
            </div>
          </div>

          {/* 1. Clustering Results View */}
          {result.clustering && (
            <>
              {/* KPI Scorecards */}
              <div className="kpi-grid">
                <div className="kpi-card">
                  <div className="kpi-icon-wrapper kpi-icon-purple">🧩</div>
                  <div className="kpi-content">
                    <span className="kpi-label">Discovered Segments</span>
                    <span className="kpi-value">{result.clustering.n_clusters}</span>
                    <span className="kpi-caption">K-Means clusters</span>
                  </div>
                </div>

                <div className="kpi-card">
                  <div className="kpi-icon-wrapper kpi-icon-emerald">✨</div>
                  <div className="kpi-content">
                    <span className="kpi-label">Silhouette Score</span>
                    <span className="kpi-value">{result.clustering.silhouette_score.toFixed(3)}</span>
                    <span className="kpi-caption">
                      {result.clustering.silhouette_score > 0.4 ? 'Well separated' : 'Moderate separation'} (-1 to +1)
                    </span>
                  </div>
                </div>

                <div className="kpi-card">
                  <div className="kpi-icon-wrapper kpi-icon-blue">📐</div>
                  <div className="kpi-content">
                    <span className="kpi-label">PCA Variance Explained</span>
                    <span className="kpi-value">
                      {((result.clustering.explained_variance_ratio.reduce((a, b) => a + b, 0)) * 100).toFixed(1)}%
                    </span>
                    <span className="kpi-caption">Top 2 Principal Components</span>
                  </div>
                </div>

                <div className="kpi-card">
                  <div className="kpi-icon-wrapper kpi-icon-amber">📏</div>
                  <div className="kpi-content">
                    <span className="kpi-label">Cluster Inertia</span>
                    <span className="kpi-value">{formatNumber(result.clustering.inertia, 1)}</span>
                    <span className="kpi-caption">Sum of squared distances</span>
                  </div>
                </div>
              </div>

              {/* 2D PCA Cluster Scatter Plot */}
              <div className="card">
                <div className="card-header-bar">
                  <div>
                    <h4 className="card-title">2D PCA Cluster Projection Space</h4>
                    <p className="card-subtitle">
                      Dimensionality reduced to 2 Principal Components with Centroids marked
                    </p>
                  </div>
                  <div className="cluster-legend">
                    {result.clustering.cluster_profiles.map((c) => (
                      <span key={c.cluster_id} className="legend-item">
                        <span
                          className="legend-dot"
                          style={{ backgroundColor: clusterColors[c.cluster_id % clusterColors.length] }}
                        />
                        {c.name} ({c.size} pts)
                      </span>
                    ))}
                  </div>
                </div>

                <div style={{ padding: '24px' }}>
                  <div className="pca-scatter-container">
                    {result.clustering.scatter_points.map((pt) => {
                      const allX = result.clustering!.scatter_points.map((p) => p.x)
                      const allY = result.clustering!.scatter_points.map((p) => p.y)
                      const minX = Math.min(...allX)
                      const maxX = Math.max(...allX)
                      const minY = Math.min(...allY)
                      const maxY = Math.max(...allY)
                      const rangeX = maxX - minX || 1
                      const rangeY = maxY - minY || 1
                      const xPct = ((pt.x - minX) / rangeX) * 90 + 5
                      const yPct = 95 - (((pt.y - minY) / rangeY) * 90 + 5)
                      const color = clusterColors[pt.cluster % clusterColors.length]

                      return (
                        <div
                          key={pt.id}
                          className="pca-point"
                          style={{ left: `${xPct}%`, top: `${yPct}%`, backgroundColor: color }}
                          title={`Point #${pt.id} (Cluster ${pt.cluster + 1}) | PCA: (${pt.x}, ${pt.y})`}
                        />
                      )
                    })}
                  </div>
                </div>
              </div>

              {/* Cluster Profiles Table */}
              <div className="card">
                <div className="card-header-bar">
                  <h4 className="card-title">Cluster Profile & Feature Archetypes</h4>
                  <span className="text-muted text-xs">Mean feature values per segment</span>
                </div>
                <div className="table-wrapper">
                  <table className="stats-table">
                    <thead>
                      <tr>
                        <th>Cluster</th>
                        <th>Size</th>
                        <th>Share</th>
                        {result.features_analyzed.map((feat) => (
                          <th key={feat} className="text-right">{feat} (Mean)</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {result.clustering.cluster_profiles.map((c) => (
                        <tr key={c.cluster_id}>
                          <td>
                            <span
                              className="badge"
                              style={{
                                backgroundColor: `${clusterColors[c.cluster_id % clusterColors.length]}20`,
                                color: clusterColors[c.cluster_id % clusterColors.length],
                                border: `1px solid ${clusterColors[c.cluster_id % clusterColors.length]}40`,
                              }}
                            >
                              {c.name}
                            </span>
                          </td>
                          <td>{c.size}</td>
                          <td>{c.percentage}%</td>
                          {result.features_analyzed.map((feat) => (
                            <td key={feat} className="text-right num-cell">
                              {formatNumber(c.feature_means[feat])}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}

          {/* 2. Correlation Matrix Heatmap */}
          {result.correlation && (
            <div className="card">
              <div className="card-header-bar">
                <div>
                  <h4 className="card-title">Feature Correlation Matrix Heatmap</h4>
                  <p className="card-subtitle">
                    Pearson correlation coefficients ($r$) between numeric variables (-1.0 to +1.0)
                  </p>
                </div>
              </div>

              <div style={{ padding: '24px' }}>
                <div className="heatmap-scroll-container">
                  <div
                    className="heatmap-grid"
                    style={{
                      gridTemplateColumns: `140px repeat(${result.correlation.columns.length}, 80px)`,
                    }}
                  >
                    <div></div>
                    {result.correlation.columns.map((col) => (
                      <div key={col} className="heatmap-header-cell" title={col}>
                        {col}
                      </div>
                    ))}

                    {result.correlation.columns.map((rowCol) => (
                      <React.Fragment key={rowCol}>
                        <div className="heatmap-row-label">{rowCol}</div>
                        {result.correlation!.columns.map((col) => {
                          const val = result.correlation!.matrix[rowCol]?.[col] ?? 0
                          const isPositive = val >= 0
                          const intensity = Math.abs(val)
                          const bgColor = isPositive
                            ? `rgba(16, 185, 129, ${intensity * 0.85})`
                            : `rgba(244, 63, 94, ${intensity * 0.85})`

                          return (
                            <div
                              key={col}
                              className="heatmap-cell"
                              style={{ backgroundColor: bgColor }}
                              title={`${rowCol} ↔ ${col}: ${val.toFixed(2)}`}
                            >
                              {val.toFixed(2)}
                            </div>
                          )
                        })}
                      </React.Fragment>
                    ))}
                  </div>
                </div>

                {/* Top Correlated Pairs */}
                <div style={{ marginTop: '24px' }}>
                  <h5 style={{ fontSize: '14px', marginBottom: '12px' }}>Strongest Feature Relationships:</h5>
                  <div className="top-pairs-grid">
                    {result.correlation.top_pairs.slice(0, 6).map((pair, idx) => (
                      <div key={idx} className="pair-card">
                        <div className="pair-cols">
                          <strong>{pair.feature_a}</strong>
                          <span className="pair-arrow">↔</span>
                          <strong>{pair.feature_b}</strong>
                        </div>
                        <span className={`badge ${pair.correlation > 0 ? 'badge-success' : 'badge-warning'}`}>
                          r = {pair.correlation > 0 ? `+${pair.correlation.toFixed(2)}` : pair.correlation.toFixed(2)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* 3. Anomaly Detection View */}
          {result.anomaly_detection && (
            <div className="card">
              <div className="card-header-bar">
                <div>
                  <h4 className="card-title">Isolation Forest Anomaly Detection</h4>
                  <p className="card-subtitle">
                    Detected {result.anomaly_detection.detected_anomalies} anomalous records ({result.anomaly_detection.anomaly_rate_pct}% outlier rate)
                  </p>
                </div>
                <span className="badge badge-warning">
                  {result.anomaly_detection.detected_anomalies} Outliers Found
                </span>
              </div>

              <div className="table-wrapper">
                <table className="stats-table">
                  <thead>
                    <tr>
                      <th>Record #</th>
                      <th>Anomaly Score</th>
                      <th>Key Feature Readings</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.anomaly_detection.top_anomalies.map((item) => (
                      <tr key={item.row_index}>
                        <td><strong>#{item.row_index}</strong></td>
                        <td>
                          <span className="badge badge-danger">
                            {item.anomaly_score.toFixed(3)}
                          </span>
                        </td>
                        <td>
                          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                            {Object.entries(item.features).map(([k, v]) => (
                              <span key={k} className="tag tag-neutral">
                                {k}: <strong>{formatNumber(v)}</strong>
                              </span>
                            ))}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
