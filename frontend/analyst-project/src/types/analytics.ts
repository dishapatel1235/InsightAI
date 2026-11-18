export interface MetricMap {
  [columnName: string]: number | null
}

export interface DescriptiveStatisticsData {
  mean?: MetricMap
  median?: MetricMap
  std?: MetricMap
  variance?: MetricMap
  min?: MetricMap
  max?: MetricMap
  sum?: MetricMap
  count?: { [columnName: string]: number }
  missing_values?: { [columnName: string]: number }
  skewness?: MetricMap
  kurtosis?: MetricMap
  message?: string
}

export interface DatasetResult {
  file_name: string
  rows: number
  columns: number
  statistics: DescriptiveStatisticsData
  columns_list?: string[]
  preview?: Record<string, unknown>[]
}

export interface ApiResponse {
  success: boolean
  data?: DatasetResult
  detail?: string
  message?: string
}

export interface ColumnStatRow {
  column: string
  mean: number | null
  median: number | null
  std: number | null
  variance: number | null
  min: number | null
  max: number | null
  sum: number | null
  count: number | null
  missing_values: number
  skewness: number | null
  kurtosis: number | null
}
