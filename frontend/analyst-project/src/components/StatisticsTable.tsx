import React, { useMemo, useState } from 'react'
import type { ColumnStatRow } from '../types/analytics'
import { formatNumber } from '../services/api'
import { downloadCsv, generateStatisticsCsv } from '../utils/exportUtils'

interface StatisticsTableProps {
  rows: ColumnStatRow[]
  onSelectColumn?: (col: string) => void
  selectedColumn?: string | null
}

type SortKey = keyof ColumnStatRow
type SortDirection = 'asc' | 'desc'

export const StatisticsTable: React.FC<StatisticsTableProps> = ({
  rows,
  onSelectColumn,
  selectedColumn,
}) => {
  const [searchTerm, setSearchTerm] = useState('')
  const [sortKey, setSortKey] = useState<SortKey>('column')
  const [sortDir, setSortDir] = useState<SortDirection>('asc')
  const [copiedNotification, setCopiedNotification] = useState<string | null>(null)

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((prev) => (prev === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir('asc')
    }
  }

  const filteredAndSortedRows = useMemo(() => {
    return rows
      .filter((row) => row.column.toLowerCase().includes(searchTerm.toLowerCase()))
      .sort((a, b) => {
        const valA = a[sortKey]
        const valB = b[sortKey]

        if (valA === null || valA === undefined) return 1
        if (valB === null || valB === undefined) return -1

        if (typeof valA === 'string' && typeof valB === 'string') {
          return sortDir === 'asc' ? valA.localeCompare(valB) : valB.localeCompare(valA)
        }

        const numA = Number(valA)
        const numB = Number(valB)
        return sortDir === 'asc' ? numA - numB : numB - numA
      })
  }, [rows, searchTerm, sortKey, sortDir])

  const copyAsCsv = () => {
    const headers = [
      'Column',
      'Mean',
      'Median',
      'Std Dev',
      'Variance',
      'Min',
      'Max',
      'Sum',
      'Count',
      'Missing',
      'Skewness',
      'Kurtosis',
    ]
    const csvLines = [
      headers.join(','),
      ...rows.map((r) =>
        [
          r.column,
          r.mean ?? '',
          r.median ?? '',
          r.std ?? '',
          r.variance ?? '',
          r.min ?? '',
          r.max ?? '',
          r.sum ?? '',
          r.count ?? '',
          r.missing_values,
          r.skewness ?? '',
          r.kurtosis ?? '',
        ].join(',')
      ),
    ]

    navigator.clipboard.writeText(csvLines.join('\n'))
    setCopiedNotification('Statistics copied as CSV!')
    setTimeout(() => setCopiedNotification(null), 3000)
  }

  const copyAsMarkdown = () => {
    const headers = '| Column | Mean | Median | Std Dev | Min | Max | Missing | Skewness |'
    const divider = '|---|---|---|---|---|---|---|---|'
    const mdLines = [
      headers,
      divider,
      ...rows.map(
        (r) =>
          `| **${r.column}** | ${formatNumber(r.mean)} | ${formatNumber(r.median)} | ${formatNumber(r.std)} | ${formatNumber(r.min)} | ${formatNumber(r.max)} | ${r.missing_values} | ${formatNumber(r.skewness)} |`
      ),
    ]

    navigator.clipboard.writeText(mdLines.join('\n'))
    setCopiedNotification('Markdown table copied to clipboard!')
    setTimeout(() => setCopiedNotification(null), 3000)
  }

  const getSkewnessBadge = (skew: number | null) => {
    if (skew === null) return null
    if (Math.abs(skew) <= 0.5) return <span className="tag tag-neutral" title="Approx. Symmetric">Symmetric</span>
    if (skew > 0.5) return <span className="tag tag-purple" title="Right-skewed tail">Right Tail</span>
    return <span className="tag tag-amber" title="Left-skewed tail">Left Tail</span>
  }

  return (
    <div className="card table-card">
      <div className="card-header-bar">
        <div>
          <h3 className="card-title">Descriptive Statistics Matrix</h3>
          <p className="card-subtitle">
            Computed summary statistics across {rows.length} numeric columns
          </p>
        </div>

        <div className="table-controls">
          <div className="search-box">
            <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" strokeWidth="2" fill="none">
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
            <input
              type="text"
              placeholder="Filter columns..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="table-search-input"
            />
            {searchTerm && (
              <button type="button" className="clear-search-btn" onClick={() => setSearchTerm('')}>×</button>
            )}
          </div>

          <div className="table-actions">
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => {
                const { headers, rows: csvRows } = generateStatisticsCsv(rows)
                downloadCsv(headers, csvRows, 'descriptive_statistics.csv')
                setCopiedNotification('Downloaded Statistics CSV!')
                setTimeout(() => setCopiedNotification(null), 3000)
              }}
              title="Download full descriptive statistics CSV file"
            >
              <svg viewBox="0 0 24 24" width="13" height="13" stroke="currentColor" strokeWidth="2" fill="none">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
              </svg>
              <span>Download CSV</span>
            </button>
            <button type="button" className="btn btn-secondary btn-sm" onClick={copyAsCsv} title="Copy table as CSV">
              Copy CSV
            </button>
            <button type="button" className="btn btn-secondary btn-sm" onClick={copyAsMarkdown} title="Copy table formatted for markdown reports">
              Copy MD
            </button>
          </div>
        </div>
      </div>

      {copiedNotification && <div className="toast-banner">{copiedNotification}</div>}

      <div className="table-wrapper">
        <table className="stats-table">
          <thead>
            <tr>
              <th onClick={() => handleSort('column')} className="sortable-th">
                Column Name {sortKey === 'column' ? (sortDir === 'asc' ? '↑' : '↓') : ''}
              </th>
              <th onClick={() => handleSort('mean')} className="sortable-th text-right">
                Mean {sortKey === 'mean' ? (sortDir === 'asc' ? '↑' : '↓') : ''}
              </th>
              <th onClick={() => handleSort('median')} className="sortable-th text-right">
                Median {sortKey === 'median' ? (sortDir === 'asc' ? '↑' : '↓') : ''}
              </th>
              <th onClick={() => handleSort('std')} className="sortable-th text-right">
                Std Dev {sortKey === 'std' ? (sortDir === 'asc' ? '↑' : '↓') : ''}
              </th>
              <th onClick={() => handleSort('variance')} className="sortable-th text-right">
                Variance {sortKey === 'variance' ? (sortDir === 'asc' ? '↑' : '↓') : ''}
              </th>
              <th onClick={() => handleSort('min')} className="sortable-th text-right">
                Min {sortKey === 'min' ? (sortDir === 'asc' ? '↑' : '↓') : ''}
              </th>
              <th onClick={() => handleSort('max')} className="sortable-th text-right">
                Max {sortKey === 'max' ? (sortDir === 'asc' ? '↑' : '↓') : ''}
              </th>
              <th onClick={() => handleSort('sum')} className="sortable-th text-right">
                Sum {sortKey === 'sum' ? (sortDir === 'asc' ? '↑' : '↓') : ''}
              </th>
              <th onClick={() => handleSort('count')} className="sortable-th text-right">
                Count {sortKey === 'count' ? (sortDir === 'asc' ? '↑' : '↓') : ''}
              </th>
              <th onClick={() => handleSort('missing_values')} className="sortable-th text-right">
                Missing {sortKey === 'missing_values' ? (sortDir === 'asc' ? '↑' : '↓') : ''}
              </th>
              <th onClick={() => handleSort('skewness')} className="sortable-th text-right">
                Skewness {sortKey === 'skewness' ? (sortDir === 'asc' ? '↑' : '↓') : ''}
              </th>
              <th onClick={() => handleSort('kurtosis')} className="sortable-th text-right">
                Kurtosis {sortKey === 'kurtosis' ? (sortDir === 'asc' ? '↑' : '↓') : ''}
              </th>
              <th>Shape</th>
            </tr>
          </thead>
          <tbody>
            {filteredAndSortedRows.length === 0 ? (
              <tr>
                <td colSpan={13} className="empty-table-cell">
                  No matching columns found.
                </td>
              </tr>
            ) : (
              filteredAndSortedRows.map((row) => {
                const isSelected = selectedColumn === row.column
                return (
                  <tr
                    key={row.column}
                    className={`stats-row ${isSelected ? 'row-selected' : ''}`}
                    onClick={() => onSelectColumn?.(row.column)}
                    title="Click row to inspect visual distribution charts"
                  >
                    <td className="col-name-cell">
                      <span className="col-icon">#</span>
                      <strong>{row.column}</strong>
                    </td>
                    <td className="text-right num-cell">{formatNumber(row.mean)}</td>
                    <td className="text-right num-cell">{formatNumber(row.median)}</td>
                    <td className="text-right num-cell">{formatNumber(row.std)}</td>
                    <td className="text-right num-cell">{formatNumber(row.variance)}</td>
                    <td className="text-right num-cell">{formatNumber(row.min)}</td>
                    <td className="text-right num-cell">{formatNumber(row.max)}</td>
                    <td className="text-right num-cell">{formatNumber(row.sum)}</td>
                    <td className="text-right num-cell">{row.count ?? '—'}</td>
                    <td className="text-right num-cell">
                      {row.missing_values > 0 ? (
                        <span className="badge badge-warning">{row.missing_values}</span>
                      ) : (
                        <span className="text-muted">0</span>
                      )}
                    </td>
                    <td className="text-right num-cell">{formatNumber(row.skewness)}</td>
                    <td className="text-right num-cell">{formatNumber(row.kurtosis)}</td>
                    <td className="text-center">{getSkewnessBadge(row.skewness)}</td>
                  </tr>
                )
              })
            )}
          </tbody>
        </table>
      </div>
      <div className="table-footer">
        <span className="table-footer-note">
          Showing {filteredAndSortedRows.length} of {rows.length} columns. Click any row to view visual charts.
        </span>
      </div>
    </div>
  )
}
