import React from 'react'
import type { ColumnStatRow } from '../types/analytics'
import { downloadCsv, generateDataQualityCsv } from '../utils/exportUtils'

interface DataQualityAuditProps {
  rows: ColumnStatRow[]
  totalRows: number
}

export const DataQualityAudit: React.FC<DataQualityAuditProps> = ({ rows, totalRows }) => {
  const missingCols = rows.filter((r) => r.missing_values > 0)
  const skewedCols = rows.filter((r) => r.skewness !== null && Math.abs(r.skewness) > 1)
  const highKurtosisCols = rows.filter((r) => r.kurtosis !== null && r.kurtosis > 2)

  const handleDownloadQualityCsv = () => {
    const { headers, rows: csvRows } = generateDataQualityCsv(rows, totalRows)
    downloadCsv(headers, csvRows, 'data_quality_audit.csv')
  }

  return (
    <div className="card audit-card">
      <div className="card-header-bar">
        <div>
          <h3 className="card-title">Data Quality & Hygiene Audit</h3>
          <p className="card-subtitle">
            Automated screening for missing values, extreme asymmetry, and distribution anomalies
          </p>
        </div>
        <div>
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={handleDownloadQualityCsv}
            title="Download data quality screening CSV matrix"
          >
            <svg viewBox="0 0 24 24" width="13" height="13" stroke="currentColor" strokeWidth="2" fill="none">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7 10 12 15 17 10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
            <span>Download Quality CSV</span>
          </button>
        </div>
      </div>

      <div className="audit-sections-grid">
        {/* Missing Value Section */}
        <div className="audit-section-box">
          <div className="audit-section-header">
            <span className="audit-icon audit-icon-warning">⚠️</span>
            <div>
              <h4 className="audit-section-title">Missing Value Analysis</h4>
              <p className="audit-section-sub">
                {missingCols.length === 0
                  ? 'Zero missing values detected in numeric columns.'
                  : `${missingCols.length} column(s) have missing entries.`}
              </p>
            </div>
          </div>

          {missingCols.length === 0 ? (
            <div className="audit-clean-state">
              <span className="clean-check">✓</span>
              <span>All numeric columns have 100% complete data integrity.</span>
            </div>
          ) : (
            <div className="missing-bars-list">
              {missingCols.map((col) => {
                const missingPct = (col.missing_values / (col.count ? col.count + col.missing_values : totalRows)) * 100
                return (
                  <div key={col.column} className="missing-bar-item">
                    <div className="missing-bar-labels">
                      <span className="missing-col-name">{col.column}</span>
                      <span className="missing-count">
                        {col.missing_values} missing ({missingPct.toFixed(1)}%)
                      </span>
                    </div>
                    <div className="missing-track">
                      <div
                        className="missing-fill"
                        style={{ width: `${Math.min(100, Math.max(5, missingPct))}%` }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>

        {/* Anomaly & Outlier Watch */}
        <div className="audit-section-box">
          <div className="audit-section-header">
            <span className="audit-icon audit-icon-purple">🎯</span>
            <div>
              <h4 className="audit-section-title">Distribution Skew & Outlier Susceptibility</h4>
              <p className="audit-section-sub">
                Features with heavy tails or high skewness may benefit from log transformations
              </p>
            </div>
          </div>

          <div className="audit-findings-list">
            {skewedCols.length > 0 && (
              <div className="finding-row">
                <span className="finding-tag tag-purple">High Skewness</span>
                <span className="finding-text">
                  Columns <strong>{skewedCols.map((c) => c.column).join(', ')}</strong> exhibit |skewness| &gt; 1.
                  Consider checking for extreme values or applying logarithmic transformation.
                </span>
              </div>
            )}

            {highKurtosisCols.length > 0 && (
              <div className="finding-row">
                <span className="finding-tag tag-amber">Leptokurtic Tails</span>
                <span className="finding-text">
                  Columns <strong>{highKurtosisCols.map((c) => c.column).join(', ')}</strong> have high kurtosis (&gt; 2), indicating potential heavy outlier presence.
                </span>
              </div>
            )}

            {skewedCols.length === 0 && highKurtosisCols.length === 0 && (
              <div className="audit-clean-state">
                <span className="clean-check">✓</span>
                <span>No severe skewness or heavy-tail anomalies detected across numeric features.</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
