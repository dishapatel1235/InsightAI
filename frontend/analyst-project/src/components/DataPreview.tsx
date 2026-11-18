import React, { useState } from 'react'

interface DataPreviewProps {
  previewRows?: Record<string, unknown>[]
  columnsList?: string[]
  totalRows: number
  totalColumns: number
}

export const DataPreview: React.FC<DataPreviewProps> = ({
  previewRows = [],
  columnsList = [],
  totalRows,
  totalColumns,
}) => {
  const [searchTerm, setSearchTerm] = useState('')

  if (!previewRows || previewRows.length === 0) {
    return (
      <div className="card empty-preview-card">
        <p>No preview rows available for this dataset.</p>
      </div>
    )
  }

  // Derive columns if not explicitly supplied
  const columns = columnsList.length > 0 ? columnsList : Object.keys(previewRows[0] || {})

  const filteredRows = previewRows.filter((row) => {
    if (!searchTerm) return true
    return Object.values(row).some((val) =>
      String(val ?? '').toLowerCase().includes(searchTerm.toLowerCase())
    )
  })

  const inferType = (val: unknown) => {
    if (val === null || val === undefined) return 'null'
    if (typeof val === 'number') return 'numeric'
    return 'text'
  }

  return (
    <div className="card preview-card">
      <div className="card-header-bar">
        <div>
          <h3 className="card-title">Dataset Raw Preview</h3>
          <p className="card-subtitle">
            Showing first {previewRows.length} sample observations out of {totalRows} total rows ({totalColumns} columns)
          </p>
        </div>

        <div className="search-box">
          <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" strokeWidth="2" fill="none">
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <input
            type="text"
            placeholder="Search preview rows..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="table-search-input"
          />
          {searchTerm && (
            <button type="button" className="clear-search-btn" onClick={() => setSearchTerm('')}>×</button>
          )}
        </div>
      </div>

      <div className="table-wrapper preview-table-wrapper">
        <table className="preview-table">
          <thead>
            <tr>
              <th className="index-th">#</th>
              {columns.map((col) => {
                const sampleVal = previewRows[0]?.[col]
                const type = inferType(sampleVal)
                return (
                  <th key={col}>
                    <div className="preview-th-content">
                      <span className="preview-col-name">{col}</span>
                      <span className={`type-badge type-${type}`}>
                        {type === 'numeric' ? '#' : 'Aa'}
                      </span>
                    </div>
                  </th>
                )
              })}
            </tr>
          </thead>
          <tbody>
            {filteredRows.map((row, idx) => (
              <tr key={idx}>
                <td className="index-td">{idx + 1}</td>
                {columns.map((col) => {
                  const val = row[col]
                  return (
                    <td key={col} className={typeof val === 'number' ? 'text-right' : ''}>
                      {val === null || val === undefined ? (
                        <span className="null-indicator">null</span>
                      ) : (
                        String(val)
                      )}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="table-footer">
        <span className="table-footer-note">
          Displaying head preview. All statistical calculations are computed across the full dataset ({totalRows} records).
        </span>
      </div>
    </div>
  )
}
