import React from 'react'
import type { ColumnStatRow, DatasetResult } from '../types/analytics'
import { formatNumber } from '../services/api'

interface SummaryCardsProps {
  dataset: DatasetResult
  columnRows: ColumnStatRow[]
}

export const SummaryCards: React.FC<SummaryCardsProps> = ({ dataset, columnRows }) => {
  const numericCount = columnRows.length
  const totalCells = dataset.rows * numericCount
  const totalMissing = columnRows.reduce((acc, r) => acc + r.missing_values, 0)
  const completenessPct =
    totalCells > 0 ? Math.max(0, Math.min(100, ((totalCells - totalMissing) / totalCells) * 100)) : 100

  const getHealthBadge = (pct: number) => {
    if (pct >= 99) return { label: 'Excellent', colorClass: 'badge-success' }
    if (pct >= 90) return { label: 'Good', colorClass: 'badge-info' }
    if (pct >= 75) return { label: 'Moderate', colorClass: 'badge-warning' }
    return { label: 'Action Needed', colorClass: 'badge-danger' }
  }

  const health = getHealthBadge(completenessPct)

  return (
    <div className="summary-cards-container">
      {/* File Info Bar */}
      <div className="dataset-meta-bar">
        <div className="meta-left">
          <span className="file-pill">
            <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" strokeWidth="2" fill="none">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
            </svg>
            <strong>{dataset.file_name}</strong>
          </span>
          <span className="meta-timestamp">Processed via FastAPI Backend</span>
        </div>
      </div>

      {/* KPI Cards Grid */}
      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-icon-wrapper kpi-icon-blue">
            <svg viewBox="0 0 24 24" width="22" height="22" stroke="currentColor" strokeWidth="2" fill="none">
              <line x1="8" y1="6" x2="21" y2="6" />
              <line x1="8" y1="12" x2="21" y2="12" />
              <line x1="8" y1="18" x2="21" y2="18" />
              <line x1="3" y1="6" x2="3.01" y2="6" />
              <line x1="3" y1="12" x2="3.01" y2="12" />
              <line x1="3" y1="18" x2="3.01" y2="18" />
            </svg>
          </div>
          <div className="kpi-content">
            <span className="kpi-label">Dataset Records</span>
            <div className="kpi-val-row">
              <span className="kpi-value">{formatNumber(dataset.rows, 0)}</span>
              <span className="kpi-sub">rows</span>
            </div>
            <span className="kpi-caption">Sample size / observations</span>
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-icon-wrapper kpi-icon-purple">
            <svg viewBox="0 0 24 24" width="22" height="22" stroke="currentColor" strokeWidth="2" fill="none">
              <rect x="3" y="3" width="18" height="18" rx="2" />
              <line x1="9" y1="3" x2="9" y2="21" />
              <line x1="15" y1="3" x2="15" y2="21" />
            </svg>
          </div>
          <div className="kpi-content">
            <span className="kpi-label">Total Attributes</span>
            <div className="kpi-val-row">
              <span className="kpi-value">{dataset.columns}</span>
              <span className="kpi-sub">columns</span>
            </div>
            <span className="kpi-caption">{numericCount} numeric, {Math.max(0, dataset.columns - numericCount)} categorical</span>
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-icon-wrapper kpi-icon-emerald">
            <svg viewBox="0 0 24 24" width="22" height="22" stroke="currentColor" strokeWidth="2" fill="none">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
            </svg>
          </div>
          <div className="kpi-content">
            <span className="kpi-label">Statistical Coverage</span>
            <div className="kpi-val-row">
              <span className="kpi-value">11</span>
              <span className="kpi-sub">metrics / col</span>
            </div>
            <span className="kpi-caption">Central, dispersion, shape</span>
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-icon-wrapper kpi-icon-amber">
            <svg viewBox="0 0 24 24" width="22" height="22" stroke="currentColor" strokeWidth="2" fill="none">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
            </svg>
          </div>
          <div className="kpi-content">
            <div className="kpi-header-row">
              <span className="kpi-label">Data Completeness</span>
              <span className={`badge ${health.colorClass}`}>{health.label}</span>
            </div>
            <div className="kpi-val-row">
              <span className="kpi-value">{completenessPct.toFixed(1)}%</span>
            </div>
            <span className="kpi-caption">
              {totalMissing === 0 ? 'No missing values detected' : `${totalMissing} missing numeric entries`}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
