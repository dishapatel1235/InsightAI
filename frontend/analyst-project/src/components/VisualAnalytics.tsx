import React, { useState } from 'react'
import type { ColumnStatRow } from '../types/analytics'
import { formatNumber } from '../services/api'

interface VisualAnalyticsProps {
  rows: ColumnStatRow[]
  selectedColumn: string | null
  onSelectColumn: (col: string) => void
}

export const VisualAnalytics: React.FC<VisualAnalyticsProps> = ({
  rows,
  selectedColumn,
  onSelectColumn,
}) => {
  const [activeChartTab, setActiveChartTab] = useState<'ranges' | 'stddev' | 'shapes'>('ranges')

  if (rows.length === 0) {
    return null
  }

  // Active column for deep-dive
  const activeColRow =
    rows.find((r) => r.column === selectedColumn) || rows[0]

  // Calculate maximum values for relative bar charts
  const maxStdDev = Math.max(...rows.map((r) => r.std ?? 0), 1)

  const getSkewInterpretation = (skew: number | null) => {
    if (skew === null) return { label: 'Insufficient Data', desc: 'Cannot compute skewness' }
    if (skew > 1) return { label: 'Highly Right-Skewed', desc: 'Substantial tail on the right side; large values pull the mean above the median' }
    if (skew > 0.5) return { label: 'Moderately Right-Skewed', desc: 'Positive asymmetry; distribution leans slightly towards lower values' }
    if (skew >= -0.5) return { label: 'Approximately Symmetric', desc: 'Balanced distribution; mean and median are closely aligned' }
    if (skew >= -1) return { label: 'Moderately Left-Skewed', desc: 'Negative asymmetry; tail extends towards smaller numbers' }
    return { label: 'Highly Left-Skewed', desc: 'Substantial tail on the left side; smaller outliers drag down the mean' }
  }

  const getKurtosisInterpretation = (kurt: number | null) => {
    if (kurt === null) return { label: 'Unknown', desc: 'Cannot compute kurtosis' }
    if (kurt > 1) return { label: 'Leptokurtic (Heavy Tails)', desc: 'Sharp peak with thicker tails; higher probability of extreme outliers' }
    if (kurt < -1) return { label: 'Platykurtic (Light Tails)', desc: 'Flatter peak with thinner tails; values spread evenly with fewer outliers' }
    return { label: 'Mesokurtic (Normal-like)', desc: 'Similar peak and tail behavior to a standard Gaussian normal distribution' }
  }

  const activeSkew = getSkewInterpretation(activeColRow.skewness)
  const activeKurt = getKurtosisInterpretation(activeColRow.kurtosis)

  return (
    <div className="card visual-analytics-card">
      <div className="card-header-bar">
        <div>
          <h3 className="card-title">Visual Analytics & Distribution Shapes</h3>
          <p className="card-subtitle">
            Exploratory visual representations of range spreads, volatility, and distribution symmetry
          </p>
        </div>

        {/* Chart View Switcher */}
        <div className="chart-tabs">
          <button
            type="button"
            className={`tab-btn ${activeChartTab === 'ranges' ? 'tab-btn-active' : ''}`}
            onClick={() => setActiveChartTab('ranges')}
          >
            Range & Centroids
          </button>
          <button
            type="button"
            className={`tab-btn ${activeChartTab === 'stddev' ? 'tab-btn-active' : ''}`}
            onClick={() => setActiveChartTab('stddev')}
          >
            Volatility & Spread
          </button>
          <button
            type="button"
            className={`tab-btn ${activeChartTab === 'shapes' ? 'tab-btn-active' : ''}`}
            onClick={() => setActiveChartTab('shapes')}
          >
            Skewness & Kurtosis
          </button>
        </div>
      </div>

      {/* Column Pills Selector */}
      <div className="column-selector-bar">
        <span className="selector-label">Active Feature:</span>
        <div className="column-pills-list">
          {rows.map((r) => (
            <button
              key={r.column}
              type="button"
              className={`column-pill ${r.column === activeColRow.column ? 'column-pill-active' : ''}`}
              onClick={() => onSelectColumn(r.column)}
            >
              {r.column}
            </button>
          ))}
        </div>
      </div>

      <div className="visual-grid">
        {/* Main Chart Section */}
        <div className="chart-main-view">
          {activeChartTab === 'ranges' && (
            <div className="chart-container">
              <div className="chart-header">
                <h4>Min-to-Max Range & Central Measures</h4>
                <div className="legend-row">
                  <span className="legend-item"><span className="legend-dot dot-min"></span> Min</span>
                  <span className="legend-item"><span className="legend-dot dot-median"></span> Median</span>
                  <span className="legend-item"><span className="legend-dot dot-mean"></span> Mean</span>
                  <span className="legend-item"><span className="legend-dot dot-max"></span> Max</span>
                </div>
              </div>

              <div className="ranges-list">
                {rows.map((row) => {
                  const min = row.min ?? 0
                  const max = row.max ?? 1
                  const range = max - min === 0 ? 1 : max - min
                  const medianPos = row.median !== null ? Math.max(0, Math.min(100, ((row.median - min) / range) * 100)) : 50
                  const meanPos = row.mean !== null ? Math.max(0, Math.min(100, ((row.mean - min) / range) * 100)) : 50
                  const isCurrent = row.column === activeColRow.column

                  return (
                    <div
                      key={row.column}
                      className={`range-row ${isCurrent ? 'range-row-active' : ''}`}
                      onClick={() => onSelectColumn(row.column)}
                    >
                      <div className="range-col-label">
                        <strong>{row.column}</strong>
                        <span className="range-subtext">Range: [{formatNumber(min)} → {formatNumber(max)}]</span>
                      </div>

                      <div className="range-track-container">
                        <div className="range-track">
                          <div className="range-track-bar"></div>
                          {/* Min point */}
                          <div className="range-point point-min" style={{ left: '0%' }} title={`Min: ${min}`} />
                          {/* Median point */}
                          <div
                            className="range-point point-median"
                            style={{ left: `${medianPos}%` }}
                            title={`Median: ${row.median}`}
                          />
                          {/* Mean point */}
                          <div
                            className="range-point point-mean"
                            style={{ left: `${meanPos}%` }}
                            title={`Mean: ${row.mean}`}
                          />
                          {/* Max point */}
                          <div className="range-point point-max" style={{ left: '100%' }} title={`Max: ${max}`} />
                        </div>
                      </div>

                      <div className="range-values">
                        <span className="val-badge val-mean">μ {formatNumber(row.mean)}</span>
                        <span className="val-badge val-median">M {formatNumber(row.median)}</span>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {activeChartTab === 'stddev' && (
            <div className="chart-container">
              <div className="chart-header">
                <h4>Dispersion & Standard Deviation (σ)</h4>
                <span className="text-muted text-xs">Relative variance indicator</span>
              </div>
              <div className="bar-chart-container">
                {rows.map((row) => {
                  const val = row.std ?? 0
                  const pct = Math.max(4, Math.min(100, (val / maxStdDev) * 100))
                  const isCurrent = row.column === activeColRow.column

                  return (
                    <div
                      key={row.column}
                      className={`bar-chart-row ${isCurrent ? 'bar-row-active' : ''}`}
                      onClick={() => onSelectColumn(row.column)}
                    >
                      <span className="bar-label">{row.column}</span>
                      <div className="bar-track">
                        <div className="bar-fill" style={{ width: `${pct}%` }}>
                          <span className="bar-inner-text">{formatNumber(val)}</span>
                        </div>
                      </div>
                      <span className="bar-variance-label">Var: {formatNumber(row.variance)}</span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {activeChartTab === 'shapes' && (
            <div className="chart-container">
              <div className="chart-header">
                <h4>Distribution Asymmetry (Skewness Index)</h4>
                <span className="text-muted text-xs">Zero = Symmetric Gaussian</span>
              </div>
              <div className="skew-chart-container">
                {rows.map((row) => {
                  const skew = row.skewness ?? 0
                  // Clamp between -2 and +2 for display
                  const clampedSkew = Math.max(-2, Math.min(2, skew))
                  // 0 is at 50%
                  const isPositive = clampedSkew >= 0
                  const widthPct = (Math.abs(clampedSkew) / 2) * 50

                  return (
                    <div key={row.column} className="skew-row" onClick={() => onSelectColumn(row.column)}>
                      <span className="skew-col-name">{row.column}</span>
                      <div className="skew-meter">
                        <div className="skew-center-line"></div>
                        <div
                          className={`skew-bar ${isPositive ? 'skew-pos' : 'skew-neg'}`}
                          style={{
                            left: isPositive ? '50%' : `${50 - widthPct}%`,
                            width: `${widthPct}%`,
                          }}
                        />
                      </div>
                      <span className="skew-value-badge">
                        {skew > 0 ? `+${skew.toFixed(2)}` : skew.toFixed(2)}
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>

        {/* Feature Deep-Dive Card */}
        <div className="feature-spotlight-card">
          <div className="spotlight-header">
            <span className="spotlight-badge">Feature Profile</span>
            <h4 className="spotlight-title">{activeColRow.column}</h4>
          </div>

          <div className="spotlight-metrics-grid">
            <div className="spotlight-metric">
              <span className="metric-tag">Mean (Average)</span>
              <span className="metric-val">{formatNumber(activeColRow.mean)}</span>
            </div>
            <div className="spotlight-metric">
              <span className="metric-tag">Median (50th %ile)</span>
              <span className="metric-val">{formatNumber(activeColRow.median)}</span>
            </div>
            <div className="spotlight-metric">
              <span className="metric-tag">Std Deviation (σ)</span>
              <span className="metric-val">{formatNumber(activeColRow.std)}</span>
            </div>
            <div className="spotlight-metric">
              <span className="metric-tag">Variance (σ²)</span>
              <span className="metric-val">{formatNumber(activeColRow.variance)}</span>
            </div>
            <div className="spotlight-metric">
              <span className="metric-tag">Minimum Value</span>
              <span className="metric-val">{formatNumber(activeColRow.min)}</span>
            </div>
            <div className="spotlight-metric">
              <span className="metric-tag">Maximum Value</span>
              <span className="metric-val">{formatNumber(activeColRow.max)}</span>
            </div>
            <div className="spotlight-metric">
              <span className="metric-tag">Cumulative Sum</span>
              <span className="metric-val">{formatNumber(activeColRow.sum)}</span>
            </div>
            <div className="spotlight-metric">
              <span className="metric-tag">Missing Data</span>
              <span className="metric-val">
                {activeColRow.missing_values === 0 ? '0 (Clean)' : `${activeColRow.missing_values} missing`}
              </span>
            </div>
          </div>

          <div className="spotlight-insights">
            <div className="insight-item">
              <span className="insight-badge badge-purple">{activeSkew.label}</span>
              <p className="insight-desc">{activeSkew.desc}</p>
            </div>
            <div className="insight-item">
              <span className="insight-badge badge-blue">{activeKurt.label}</span>
              <p className="insight-desc">{activeKurt.desc}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
