import type { ColumnStatRow, DatasetResult } from '../types/analytics'
import type {
  ClassificationMetrics,
  PatternDiscoveryResult,
  PredictiveModelResult,
  RegressionMetrics,
  TimeSeriesForecastResult,
} from '../types/ml'
import { formatNumber } from '../services/api'

/**
 * Initiates an automatic file download in the browser.
 */
export const downloadFile = (content: string, filename: string, mimeType: string) => {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1500)
}

/**
 * Downloads arbitrary data as formatted JSON.
 */
export const downloadJson = (data: unknown, filename: string) => {
  const jsonStr = JSON.stringify(data, null, 2)
  downloadFile(jsonStr, filename, 'application/json;charset=utf-8;')
}

/**
 * Escapes a CSV cell value to handle commas, newlines, and quotes safely.
 */
const escapeCsvCell = (val: unknown): string => {
  if (val === null || val === undefined) return ''
  const str = String(val)
  if (str.includes(',') || str.includes('\n') || str.includes('"') || str.includes('\r')) {
    return `"${str.replace(/"/g, '""')}"`
  }
  return str
}

/**
 * Downloads tabular data as a clean CSV file.
 */
export const downloadCsv = (
  headers: string[],
  rows: (string | number | boolean | null | undefined)[][],
  filename: string
) => {
  const headerLine = headers.map(escapeCsvCell).join(',')
  const rowLines = rows.map((r) => r.map(escapeCsvCell).join(','))
  const csvContent = [headerLine, ...rowLines].join('\r\n')
  downloadFile(csvContent, filename, 'text/csv;charset=utf-8;')
}

// =========================================================================
// TAB 1: DESCRIPTIVE STATISTICS
// =========================================================================

export const generateStatisticsCsv = (columnRows: ColumnStatRow[]): { headers: string[]; rows: (string | number | null)[][] } => {
  const headers = [
    'Column Name',
    'Mean',
    'Median',
    'Std Dev',
    'Variance',
    'Min',
    'Max',
    'Sum',
    'Sample Count',
    'Missing Values',
    'Skewness',
    'Kurtosis',
  ]

  const rows = columnRows.map((r) => [
    r.column,
    r.mean ?? null,
    r.median ?? null,
    r.std ?? null,
    r.variance ?? null,
    r.min ?? null,
    r.max ?? null,
    r.sum ?? null,
    r.count ?? null,
    r.missing_values,
    r.skewness ?? null,
    r.kurtosis ?? null,
  ])

  return { headers, rows }
}

export const generateStatisticsMarkdown = (dataset: DatasetResult, columnRows: ColumnStatRow[]): string => {
  let md = `# Descriptive Statistics Report for ${dataset.file_name}\n\n`
  md += `**Observations (Rows):** ${dataset.rows} | **Attributes (Columns):** ${dataset.columns}\n\n`
  md += `| Column | Mean | Median | Std Dev | Min | Max | Missing | Skewness | Kurtosis |\n`
  md += `| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n`

  columnRows.forEach((r) => {
    md += `| **${r.column}** | ${formatNumber(r.mean)} | ${formatNumber(r.median)} | ${formatNumber(r.std)} | ${formatNumber(r.min)} | ${formatNumber(r.max)} | ${r.missing_values} | ${formatNumber(r.skewness)} | ${formatNumber(r.kurtosis)} |\n`
  })

  return md
}

// =========================================================================
// TAB 2: DISTRIBUTIONS & VISUAL ANALYTICS
// =========================================================================

export const generateDistributionsCsv = (columnRows: ColumnStatRow[]): { headers: string[]; rows: (string | number | null)[][] } => {
  const headers = [
    'Column Name',
    'Mean',
    'Median',
    'Std Dev',
    'Min',
    'Max',
    'Spread Range (Max-Min)',
    'Skewness',
    'Kurtosis',
    'Distribution Symmetry',
    'Tail Weight',
  ]

  const rows = columnRows.map((r) => {
    const range = r.max !== null && r.min !== null ? r.max - r.min : null
    let symmetry = 'Symmetric'
    if (r.skewness !== null) {
      if (r.skewness > 1) symmetry = 'Highly Right-Skewed'
      else if (r.skewness > 0.5) symmetry = 'Moderately Right-Skewed'
      else if (r.skewness < -1) symmetry = 'Highly Left-Skewed'
      else if (r.skewness < -0.5) symmetry = 'Moderately Left-Skewed'
    }

    let tail = 'Mesokurtic (Normal-like)'
    if (r.kurtosis !== null) {
      if (r.kurtosis > 2) tail = 'Leptokurtic (Heavy Tails / Outliers)'
      else if (r.kurtosis < -1) tail = 'Platykurtic (Light Tails / Flat)'
    }

    return [
      r.column,
      r.mean ?? null,
      r.median ?? null,
      r.std ?? null,
      r.min ?? null,
      r.max ?? null,
      range,
      r.skewness ?? null,
      r.kurtosis ?? null,
      symmetry,
      tail,
    ]
  })

  return { headers, rows }
}

// =========================================================================
// TAB 3: PREDICTIVE MACHINE LEARNING
// =========================================================================

export const generateMlPredictionsCsv = (
  result: PredictiveModelResult
): { headers: string[]; rows: (string | number | boolean)[][] } | null => {
  if (!result || !result.metrics || !result.metrics.sample_predictions) return null

  if (result.task_type === 'classification') {
    const classMetrics = result.metrics as ClassificationMetrics
    const headers = ['Sample #', 'Actual Label', 'Predicted Label', 'Correct Classification']
    const rows = classMetrics.sample_predictions.map((p) => [
      p.index,
      p.actual,
      p.predicted,
      p.correct ? 'YES' : 'NO',
    ])
    return { headers, rows }
  } else {
    const regMetrics = result.metrics as RegressionMetrics
    const headers = ['Sample #', 'Actual Value', 'Predicted Value', 'Residual Error', 'Absolute Error']
    const rows = regMetrics.sample_predictions.map((p) => [
      p.index,
      p.actual,
      p.predicted,
      p.residual,
      Math.abs(p.residual),
    ])
    return { headers, rows }
  }
}

export const generateMlFeatureImportanceCsv = (
  result: PredictiveModelResult
): { headers: string[]; rows: (string | number)[][] } | null => {
  if (!result || !result.feature_importances) return null

  const headers = ['Rank', 'Feature Name', 'Relative Importance Score', 'Importance Percentage (%)']
  const rows = result.feature_importances.map((item, idx) => [
    idx + 1,
    item.feature,
    item.importance,
    round(item.importance * 100, 2),
  ])
  return { headers, rows }
}

export const generateMlSummaryMarkdown = (result: PredictiveModelResult): string => {
  let md = `# Predictive Machine Learning Report\n\n`
  md += `- **Model Architecture:** ${result.model_name}\n`
  md += `- **Task Type:** ${result.task_type.toUpperCase()}\n`
  md += `- **Target Variable:** ${result.target_column}\n`
  md += `- **Predictor Features Count:** ${result.feature_columns.length}\n`
  md += `- **Training Observations:** ${result.train_sample_count} | **Test Observations:** ${result.test_sample_count}\n\n`

  md += `## Evaluation Performance Metrics\n\n`
  if (result.task_type === 'classification') {
    const m = result.metrics as ClassificationMetrics
    md += `| Metric | Score |\n| :--- | :--- |\n`
    md += `| Test Accuracy | ${(m.accuracy * 100).toFixed(2)}% |\n`
    md += `| Train Accuracy | ${(m.train_accuracy * 100).toFixed(2)}% |\n`
    md += `| Weighted Precision | ${(m.precision * 100).toFixed(2)}% |\n`
    md += `| Weighted Recall | ${(m.recall * 100).toFixed(2)}% |\n`
    md += `| Weighted F1-Score | ${(m.f1_score * 100).toFixed(2)}% |\n\n`
  } else {
    const m = result.metrics as RegressionMetrics
    md += `| Metric | Value |\n| :--- | :--- |\n`
    md += `| Test R² Score | ${(m.r2_score * 100).toFixed(2)}% |\n`
    md += `| Train R² Score | ${(m.train_r2_score * 100).toFixed(2)}% |\n`
    md += `| Root Mean Squared Error (RMSE) | ${m.rmse} |\n`
    md += `| Mean Absolute Error (MAE) | ${m.mae} |\n`
    md += `| Mean Squared Error (MSE) | ${m.mse} |\n`
    md += `| Mean Absolute Percentage Error (MAPE) | ${m.mape_pct}% |\n\n`
  }

  md += `## Top Predictor Feature Importances\n\n`
  md += `| Rank | Feature | Importance |\n| :--- | :--- | :--- |\n`
  result.feature_importances.forEach((fi, idx) => {
    md += `| ${idx + 1} | **${fi.feature}** | ${(fi.importance * 100).toFixed(2)}% |\n`
  })

  return md
}

// =========================================================================
// TAB 4: PATTERN DISCOVERY & CLUSTERS
// =========================================================================

export const generateClusterSegmentsCsv = (
  result: PatternDiscoveryResult
): { headers: string[]; rows: (string | number)[][] } | null => {
  if (!result || !result.clustering || !result.clustering.cluster_profiles) return null

  const profiles = result.clustering.cluster_profiles
  const featureKeys = Object.keys(profiles[0]?.feature_means || {})

  const headers = [
    'Cluster ID',
    'Segment Name',
    'Cluster Size (Count)',
    'Percentage (%)',
    'PCA Centroid X',
    'PCA Centroid Y',
    ...featureKeys.map((k) => `Mean: ${k}`),
  ]

  const rows = profiles.map((p) => [
    p.cluster_id,
    p.name,
    p.size,
    p.percentage,
    p.centroid_pca.x,
    p.centroid_pca.y,
    ...featureKeys.map((k) => p.feature_means[k] ?? ''),
  ])

  return { headers, rows }
}

export const generateCorrelationMatrixCsv = (
  result: PatternDiscoveryResult
): { headers: string[]; rows: (string | number)[][] } | null => {
  if (!result || !result.correlation || !result.correlation.matrix) return null

  const cols = result.correlation.columns
  const headers = ['Feature', ...cols]
  const rows = cols.map((colA) => [
    colA,
    ...cols.map((colB) => result.correlation!.matrix[colA]?.[colB] ?? 0),
  ])

  return { headers, rows }
}

export const generateAnomaliesCsv = (
  result: PatternDiscoveryResult
): { headers: string[]; rows: (string | number)[][] } | null => {
  if (!result || !result.anomaly_detection || !result.anomaly_detection.top_anomalies) return null

  const anomalies = result.anomaly_detection.top_anomalies
  if (anomalies.length === 0) return null

  const featureKeys = Object.keys(anomalies[0].features)
  const headers = ['Row Index', 'Anomaly Score', ...featureKeys]
  const rows = anomalies.map((item) => [
    item.row_index,
    item.anomaly_score,
    ...featureKeys.map((k) => item.features[k] ?? ''),
  ])

  return { headers, rows }
}

// =========================================================================
// TAB 5: LSTM & TIME-SERIES FORECASTING
// =========================================================================

export const generateForecastTimelineCsv = (
  result: TimeSeriesForecastResult
): { headers: string[]; rows: (string | number | null)[][] } | null => {
  if (!result || !result.timeline) return null

  const headers = [
    'Time / Step Label',
    'Observed Actual',
    'Model Fitted Value',
    'Future Forecast',
    '95% Lower Confidence Bound',
    '95% Upper Confidence Bound',
  ]

  const rows = result.timeline.map((pt) => [
    pt.label,
    pt.actual ?? null,
    pt.fitted ?? null,
    pt.forecast ?? null,
    pt.lower_bound ?? null,
    pt.upper_bound ?? null,
  ])

  return { headers, rows }
}

export const generateForecastMarkdown = (result: TimeSeriesForecastResult): string => {
  let md = `# Time-Series Forecasting Trajectory Report\n\n`
  md += `- **Forecasting Model:** ${result.model_name}\n`
  md += `- **Target Variable:** ${result.target_column}\n`
  md += `- **Lookback Window:** ${result.lookback_window} steps | **Future Horizon:** ${result.horizon} steps\n`
  md += `- **Last Observed Value:** ${result.last_observed_value}\n\n`

  md += `## Forecast Accuracy Metrics\n\n`
  md += `| Metric | Value |\n| :--- | :--- |\n`
  md += `| Mean Absolute Error (MAE) | ${result.metrics.mae} |\n`
  md += `| Root Mean Squared Error (RMSE) | ${result.metrics.rmse} |\n`
  md += `| Mean Absolute Percentage Error (MAPE) | ${result.metrics.mape_pct}% |\n\n`

  md += `## Future Forecast Projections (95% Confidence Bounds)\n\n`
  md += `| Step | Forecasted Value | 95% Lower Bound | 95% Upper Bound |\n| :--- | :--- | :--- | :--- |\n`
  result.future_forecast.forEach((f) => {
    md += `| Step +${f.step} | **${f.forecast}** | ${f.lower_bound} | ${f.upper_bound} |\n`
  })

  return md
}

// =========================================================================
// TAB 6: DATA QUALITY AUDIT
// =========================================================================

export const generateDataQualityCsv = (
  columnRows: ColumnStatRow[],
  totalRows: number
): { headers: string[]; rows: (string | number)[][] } => {
  const headers = [
    'Column Name',
    'Total Dataset Rows',
    'Valid Entries',
    'Missing Count',
    'Missing Percentage (%)',
    'Skewness Asymmetry',
    'Kurtosis Heavy Tail Indicator',
    'Integrity Rating',
  ]

  const rows = columnRows.map((r) => {
    const missingPct = totalRows > 0 ? (r.missing_values / totalRows) * 100 : 0
    let rating = 'Excellent (100% Clean)'
    if (missingPct > 20) rating = 'Critical Missing (>20%)'
    else if (missingPct > 5) rating = 'Moderate Missing (>5%)'
    else if (missingPct > 0) rating = 'Minor Missing (<5%)'

    return [
      r.column,
      totalRows,
      r.count ?? totalRows - r.missing_values,
      r.missing_values,
      round(missingPct, 2),
      r.skewness !== null ? round(r.skewness, 3) : 'N/A',
      r.kurtosis !== null ? round(r.kurtosis, 3) : 'N/A',
      rating,
    ]
  })

  return { headers, rows }
}

// =========================================================================
// TAB 7: DATA PREVIEW
// =========================================================================

export const generateDataPreviewCsv = (
  previewRows: Record<string, unknown>[],
  columns: string[]
): { headers: string[]; rows: (string | number | boolean | null)[][] } => {
  const headers = columns
  const rows = previewRows.map((row) =>
    columns.map((col) => {
      const val = row[col]
      if (val === null || val === undefined) return ''
      if (typeof val === 'object') return JSON.stringify(val)
      return val as string | number | boolean
    })
  )
  return { headers, rows }
}

// =========================================================================
// ENTIRE COMPREHENSIVE MASTER REPORT GENERATOR
// =========================================================================

export interface MasterReportBundleOptions {
  dataset: DatasetResult
  columnRows: ColumnStatRow[]
  predictiveResult?: PredictiveModelResult | null
  patternResult?: PatternDiscoveryResult | null
  timeSeriesResult?: TimeSeriesForecastResult | null
  previewRows?: Record<string, unknown>[]
  columnsList?: string[]
}

/**
 * Builds the complete multi-module JSON master document containing outcomes of every tab.
 */
export const buildMasterReportJson = (opts: MasterReportBundleOptions) => {
  const totalRows = opts.dataset.rows
  const qualityRows = opts.columnRows.map((r) => {
    const missingPct = totalRows > 0 ? (r.missing_values / totalRows) * 100 : 0
    return {
      column: r.column,
      total_rows: totalRows,
      present_count: r.count,
      missing_count: r.missing_values,
      missing_pct: round(missingPct, 2),
      skewness: r.skewness,
      kurtosis: r.kurtosis,
    }
  })

  return {
    report_title: 'InsightAI Master Analytics & Intelligence Report',
    generated_at: new Date().toISOString(),
    dataset_metadata: {
      file_name: opts.dataset.file_name,
      total_rows: opts.dataset.rows,
      total_columns: opts.dataset.columns,
      numeric_columns_count: opts.columnRows.length,
      all_columns: opts.dataset.columns_list || Object.keys(opts.dataset.preview?.[0] || {}),
    },
    module_1_descriptive_statistics: {
      metrics_by_column: opts.dataset.statistics,
      summary_table: opts.columnRows,
    },
    module_2_distributions: {
      columns_analyzed: opts.columnRows.map((r) => ({
        column: r.column,
        mean: r.mean,
        median: r.median,
        std: r.std,
        min: r.min,
        max: r.max,
        skewness: r.skewness,
        kurtosis: r.kurtosis,
      })),
    },
    module_3_data_quality_audit: {
      overall_completeness_pct: calculateCompleteness(opts.columnRows, totalRows),
      missing_columns_count: opts.columnRows.filter((r) => r.missing_values > 0).length,
      column_audit_matrix: qualityRows,
    },
    module_4_predictive_machine_learning: opts.predictiveResult || {
      status: 'No model trained in this session',
    },
    module_5_pattern_discovery_and_clusters: opts.patternResult || {
      status: 'Pattern discovery not run in this session',
    },
    module_6_time_series_forecasting: opts.timeSeriesResult || {
      status: 'Time-series forecasting not run in this session',
    },
    module_7_data_preview_sample: {
      sample_records_count: opts.previewRows?.length || opts.dataset.preview?.length || 0,
      preview_data: opts.previewRows || opts.dataset.preview || [],
    },
  }
}

/**
 * Builds the Executive Master Markdown Brief covering every single tab.
 */
export const buildMasterReportMarkdown = (opts: MasterReportBundleOptions): string => {
  const { dataset, columnRows, predictiveResult, patternResult, timeSeriesResult } = opts
  const completeness = calculateCompleteness(columnRows, dataset.rows)

  let md = `# InsightAI Executive Analytics & Intelligence Brief\n`
  md += `*Generated automatically on ${new Date().toLocaleDateString()} at ${new Date().toLocaleTimeString()}*\n\n`

  md += `## 1. Executive Summary & Dataset Overview\n\n`
  md += `- **Target File:** \`${dataset.file_name}\`\n`
  md += `- **Total Observations (Rows):** ${dataset.rows.toLocaleString()}\n`
  md += `- **Attributes (Columns):** ${dataset.columns.toLocaleString()}\n`
  md += `- **Numeric Features Profiled:** ${columnRows.length}\n`
  md += `- **Data Completeness Health Score:** **${completeness}%**\n\n`

  md += `## 2. Descriptive Statistics Summary\n\n`
  md += `| Column | Mean | Median | Std Dev | Min | Max | Missing | Skewness |\n`
  md += `| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n`
  columnRows.forEach((r) => {
    md += `| **${r.column}** | ${formatNumber(r.mean)} | ${formatNumber(r.median)} | ${formatNumber(r.std)} | ${formatNumber(r.min)} | ${formatNumber(r.max)} | ${r.missing_values} | ${formatNumber(r.skewness)} |\n`
  })
  md += `\n`

  md += `## 3. Data Quality & Hygiene Screening\n\n`
  const missingCols = columnRows.filter((r) => r.missing_values > 0)
  if (missingCols.length === 0) {
    md += `> **100% Data Integrity:** All numeric columns contain zero missing entries.\n\n`
  } else {
    md += `> **Missing Data Notice:** Detected missing values across ${missingCols.length} column(s):\n`
    missingCols.forEach((c) => {
      const pct = dataset.rows > 0 ? ((c.missing_values / dataset.rows) * 100).toFixed(1) : '0'
      md += `- **${c.column}:** ${c.missing_values} missing (${pct}% of records)\n`
    })
    md += `\n`
  }

  // Predictive ML
  md += `## 4. Supervised Machine Learning Outcome\n\n`
  if (predictiveResult) {
    md += `- **Model Architecture:** ${predictiveResult.model_name}\n`
    md += `- **Task Type:** ${predictiveResult.task_type.toUpperCase()}\n`
    md += `- **Target Column Predicted:** \`${predictiveResult.target_column}\`\n`
    if (predictiveResult.task_type === 'classification') {
      const m = predictiveResult.metrics as ClassificationMetrics
      md += `- **Accuracy:** ${(m.accuracy * 100).toFixed(2)}% | **Weighted F1:** ${(m.f1_score * 100).toFixed(2)}%\n\n`
    } else {
      const m = predictiveResult.metrics as RegressionMetrics
      md += `- **R² Score:** ${(m.r2_score * 100).toFixed(2)}% | **RMSE:** ${m.rmse} | **MAE:** ${m.mae}\n\n`
    }
    md += `### Top Predictor Features\n`
    predictiveResult.feature_importances.slice(0, 5).forEach((fi, idx) => {
      md += `${idx + 1}. **${fi.feature}**: ${(fi.importance * 100).toFixed(2)}%\n`
    })
    md += `\n`
  } else {
    md += `*Predictive modeling not executed in this session.*\n\n`
  }

  // Pattern Discovery
  md += `## 5. Unsupervised Pattern Discovery & Clustering\n\n`
  if (patternResult?.clustering) {
    const cl = patternResult.clustering
    md += `- **Clusters Identified ($K$):** ${cl.n_clusters}\n`
    md += `- **Silhouette Score:** ${cl.silhouette_score} | **Inertia:** ${cl.inertia}\n\n`
    md += `| Segment | Size (Records) | Share (%) | Centroid (PCA X, Y) |\n| :--- | :--- | :--- | :--- |\n`
    cl.cluster_profiles.forEach((p) => {
      md += `| **${p.name}** | ${p.size.toLocaleString()} | ${p.percentage}% | (${p.centroid_pca.x}, ${p.centroid_pca.y}) |\n`
    })
    md += `\n`
  } else {
    md += `*Clustering analysis not executed in this session.*\n\n`
  }

  if (patternResult?.anomaly_detection) {
    const ad = patternResult.anomaly_detection
    md += `### Anomaly Screening\n`
    md += `- **Detected Anomalies:** ${ad.detected_anomalies} (${ad.anomaly_rate_pct}% anomaly rate)\n\n`
  }

  // Time-Series
  md += `## 6. Sequential Time-Series Forecasting\n\n`
  if (timeSeriesResult) {
    md += `- **Forecasting Model:** ${timeSeriesResult.model_name}\n`
    md += `- **Target Variable:** \`${timeSeriesResult.target_column}\`\n`
    md += `- **Forecast Horizon:** ${timeSeriesResult.horizon} steps ahead\n`
    md += `- **Validation MAPE:** ${timeSeriesResult.metrics.mape_pct}% | **RMSE:** ${timeSeriesResult.metrics.rmse}\n\n`
    md += `| Horizon Step | Forecasted Point | 95% Range Envelope |\n| :--- | :--- | :--- |\n`
    timeSeriesResult.future_forecast.slice(0, 7).forEach((f) => {
      md += `| Step +${f.step} | **${f.forecast}** | [${f.lower_bound} to ${f.upper_bound}] |\n`
    })
    md += `\n`
  } else {
    md += `*Time-series forecasting not executed in this session.*\n\n`
  }

  md += `---\n*Generated by InsightAI Analytics Platform*\n`
  return md
}

const calculateCompleteness = (columnRows: ColumnStatRow[], totalRows: number): number => {
  if (columnRows.length === 0 || totalRows === 0) return 100
  const totalEntries = columnRows.length * totalRows
  const missingEntries = columnRows.reduce((sum, r) => sum + r.missing_values, 0)
  return round(Math.max(0, ((totalEntries - missingEntries) / totalEntries) * 100), 1)
}

const round = (val: number, decimals: number): number => {
  const factor = Math.pow(10, decimals)
  return Math.round(val * factor) / factor
}
