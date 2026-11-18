import type { ApiResponse, ColumnStatRow, DatasetResult, DescriptiveStatisticsData } from '../types/analytics'
import type {
  LivePredictionResult,
  ModelType,
  PatternDiscoveryResult,
  PatternMethod,
  PredictiveModelResult,
  TaskType,
  TimeSeriesForecastResult,
  TimeSeriesModel,
} from '../types/ml'

export const DEFAULT_BACKEND_URL =
  import.meta.env.VITE_BACKEND_URL || 'http://127.0.0.1:8000'

export const STORAGE_KEY_API_URL = 'analyst_backend_url'

export function getBackendBaseUrl(): string {
  if (typeof window !== 'undefined') {
    const saved = localStorage.getItem(STORAGE_KEY_API_URL)
    if (saved) return saved
  }
  return DEFAULT_BACKEND_URL
}

export function setBackendBaseUrl(url: string) {
  if (typeof window !== 'undefined') {
    localStorage.setItem(STORAGE_KEY_API_URL, url)
  }
}

/**
 * Pings backend to check health status
 */
export async function checkBackendHealth(baseUrl?: string): Promise<{ isOnline: boolean; message: string }> {
  const base = baseUrl || getBackendBaseUrl()
  try {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 3000)

    const response = await fetch(`${base}/`, {
      method: 'GET',
      signal: controller.signal,
    })
    clearTimeout(timeoutId)

    if (response.ok) {
      const data = await response.json().catch(() => ({}))
      return { isOnline: true, message: data.message || 'FastAPI service active' }
    }
    return { isOnline: false, message: `HTTP ${response.status}: ${response.statusText}` }
  } catch (err: unknown) {
    try {
      const proxyRes = await fetch('/api/', { method: 'GET' })
      if (proxyRes.ok) {
        const data = await proxyRes.json().catch(() => ({}))
        return { isOnline: true, message: data.message || 'FastAPI service active (via Proxy)' }
      }
    } catch {
      // ignore
    }
    const isAbort = err instanceof Error && err.name === 'AbortError'
    return {
      isOnline: false,
      message: isAbort ? 'Connection timed out' : 'Backend offline or unreachable',
    }
  }
}

/**
 * Uploads a CSV or Excel dataset to POST /datasets/upload
 */
export async function uploadDataset(file: File, baseUrl?: string): Promise<DatasetResult> {
  const base = baseUrl || getBackendBaseUrl()
  const formData = new FormData()
  formData.append('file', file)

  const endpointsToTry = [
    `${base}/datasets/upload`,
    '/datasets/upload',
  ]

  let lastError: Error | null = null

  for (const endpoint of endpointsToTry) {
    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        const detailMsg = errorData.detail || errorData.message || `Upload failed with status: ${response.status}`
        throw new Error(detailMsg)
      }

      const json: ApiResponse = await response.json()
      if (json.success && json.data) {
        return json.data
      }

      throw new Error(json.message || 'Malformed response from backend')
    } catch (err: unknown) {
      lastError = err instanceof Error ? err : new Error(String(err))
      if (endpoint === endpointsToTry[endpointsToTry.length - 1]) {
        break
      }
    }
  }

  throw lastError || new Error('Failed to connect to backend server. Make sure uvicorn is running.')
}

export interface ServerFile {
  file_name: string
  size_bytes: number
  size_display: string
}

/**
 * Retrieves list of files already present on the backend in uploads/
 */
export async function fetchServerDatasets(baseUrl?: string): Promise<ServerFile[]> {
  const base = baseUrl || getBackendBaseUrl()
  const endpoints = [`${base}/datasets/list`, '/datasets/list']
  for (const ep of endpoints) {
    try {
      const res = await fetch(ep)
      if (res.ok) {
        const json = await res.json()
        if (json.success && Array.isArray(json.files)) {
          return json.files
        }
      }
    } catch {
      // try next
    }
  }
  return []
}

/**
 * Loads an existing dataset from uploads/ without needing to re-upload
 */
export async function selectExistingDataset(fileName: string, baseUrl?: string): Promise<DatasetResult> {
  const base = baseUrl || getBackendBaseUrl()
  const endpoints = [`${base}/datasets/select-existing`, '/datasets/select-existing']
  let lastErr: Error | null = null

  for (const ep of endpoints) {
    try {
      const res = await fetch(ep, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_name: fileName }),
      })
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}))
        throw new Error(errJson.detail || `Failed with status ${res.status}`)
      }
      const json: ApiResponse = await res.json()
      if (json.success && json.data) {
        return json.data
      }
      throw new Error('Invalid response structure')
    } catch (err: unknown) {
      lastErr = err instanceof Error ? err : new Error(String(err))
      if (ep === endpoints[endpoints.length - 1]) break
    }
  }
  throw lastErr || new Error('Failed to load existing dataset from server.')
}


/* =========================================================================
   MACHINE LEARNING & DEEP LEARNING API METHODS
========================================================================= */

/**
 * Trains a predictive ML or DL model (Random Forest, XGBoost, Neural Network)
 */
export async function trainPredictiveModel(params: {
  fileName: string
  targetColumn: string
  taskType: TaskType
  modelName: ModelType
  featureColumns?: string[]
  hyperparameters?: Record<string, unknown>
  baseUrl?: string
}): Promise<PredictiveModelResult> {
  const base = params.baseUrl || getBackendBaseUrl()
  const payload = {
    file_name: params.fileName,
    target_column: params.targetColumn,
    task_type: params.taskType,
    model_name: params.modelName,
    feature_columns: params.featureColumns,
    hyperparameters: params.hyperparameters,
  }

  const endpoints = [`${base}/ml/train-predictive`, '/ml/train-predictive']
  let lastErr: Error | null = null

  for (const url of endpoints) {
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}))
        throw new Error(errJson.detail || `Training failed with status ${res.status}`)
      }
      const data = await res.json()
      if (data.success && data.data) {
        return data.data
      }
      throw new Error('Invalid response structure from ML backend.')
    } catch (err: unknown) {
      lastErr = err instanceof Error ? err : new Error(String(err))
      if (url === endpoints[endpoints.length - 1]) break
    }
  }
  throw lastErr || new Error('Failed to connect to ML training service.')
}

/**
 * Executes pattern discovery: K-Means Clustering, Correlation Heatmap, Anomaly Detection
 */
export async function analyzePatterns(params: {
  fileName: string
  method: PatternMethod | 'all'
  featureColumns?: string[]
  nClusters?: number
  contamination?: number
  baseUrl?: string
}): Promise<PatternDiscoveryResult> {
  const base = params.baseUrl || getBackendBaseUrl()
  const payload = {
    file_name: params.fileName,
    method: params.method,
    feature_columns: params.featureColumns,
    n_clusters: params.nClusters ?? 3,
    contamination: params.contamination ?? 0.05,
  }

  const endpoints = [`${base}/ml/pattern-discovery`, '/ml/pattern-discovery']
  let lastErr: Error | null = null

  for (const url of endpoints) {
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}))
        throw new Error(errJson.detail || `Pattern analysis failed with status ${res.status}`)
      }
      const data = await res.json()
      if (data.success && data.data) {
        return data.data
      }
      throw new Error('Invalid response from pattern discovery service.')
    } catch (err: unknown) {
      lastErr = err instanceof Error ? err : new Error(String(err))
      if (url === endpoints[endpoints.length - 1]) break
    }
  }
  throw lastErr || new Error('Failed to connect to pattern discovery service.')
}

/**
 * Generates continuous time-series forecasts using LSTM or sequential models
 */
export async function forecastTimeSeries(params: {
  fileName: string
  targetColumn: string
  timeColumn?: string
  modelName: TimeSeriesModel
  lookback?: number
  horizon?: number
  epochs?: number
  baseUrl?: string
}): Promise<TimeSeriesForecastResult> {
  const base = params.baseUrl || getBackendBaseUrl()
  const payload = {
    file_name: params.fileName,
    target_column: params.targetColumn,
    time_column: params.timeColumn,
    model_name: params.modelName,
    lookback: params.lookback ?? 5,
    horizon: params.horizon ?? 10,
    epochs: params.epochs ?? 50,
  }

  const endpoints = [`${base}/ml/forecast-timeseries`, '/ml/forecast-timeseries']
  let lastErr: Error | null = null

  for (const url of endpoints) {
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}))
        throw new Error(errJson.detail || `Forecasting failed with status ${res.status}`)
      }
      const data = await res.json()
      if (data.success && data.data) {
        return data.data
      }
      throw new Error('Invalid response from time-series forecast service.')
    } catch (err: unknown) {
      lastErr = err instanceof Error ? err : new Error(String(err))
      if (url === endpoints[endpoints.length - 1]) break
    }
  }
  throw lastErr || new Error('Failed to connect to time-series service.')
}

/**
 * Performs instantaneous "What-If" inference on an active model
 */
export async function predictLive(params: {
  modelId: string
  inputFeatures: Record<string, unknown>
  baseUrl?: string
}): Promise<LivePredictionResult> {
  const base = params.baseUrl || getBackendBaseUrl()
  const payload = {
    model_id: params.modelId,
    input_features: params.inputFeatures,
  }

  const endpoints = [`${base}/ml/predict`, '/ml/predict']
  let lastErr: Error | null = null

  for (const url of endpoints) {
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}))
        throw new Error(errJson.detail || `Inference failed with status ${res.status}`)
      }
      const data = await res.json()
      if (data.success && data.data) {
        return data.data
      }
      throw new Error('Invalid response from prediction endpoint.')
    } catch (err: unknown) {
      lastErr = err instanceof Error ? err : new Error(String(err))
      if (url === endpoints[endpoints.length - 1]) break
    }
  }
  throw lastErr || new Error('Failed to compute live prediction.')
}

/* =========================================================================
   STATISTICS HELPERS
========================================================================= */

export function formatColumnStatistics(stats: DescriptiveStatisticsData): ColumnStatRow[] {
  if (!stats || stats.message) {
    return []
  }

  const columnNames = new Set<string>()
  const checkMap = (map?: Record<string, unknown>) => {
    if (map) {
      Object.keys(map).forEach((col) => columnNames.add(col))
    }
  }

  checkMap(stats.mean)
  checkMap(stats.median)
  checkMap(stats.std)
  checkMap(stats.min)
  checkMap(stats.max)
  checkMap(stats.count)
  checkMap(stats.missing_values)

  const rows: ColumnStatRow[] = []

  columnNames.forEach((column) => {
    rows.push({
      column,
      mean: stats.mean?.[column] ?? null,
      median: stats.median?.[column] ?? null,
      std: stats.std?.[column] ?? null,
      variance: stats.variance?.[column] ?? null,
      min: stats.min?.[column] ?? null,
      max: stats.max?.[column] ?? null,
      sum: stats.sum?.[column] ?? null,
      count: stats.count?.[column] ?? null,
      missing_values: stats.missing_values?.[column] ?? 0,
      skewness: stats.skewness?.[column] ?? null,
      kurtosis: stats.kurtosis?.[column] ?? null,
    })
  })

  return rows.sort((a, b) => a.column.localeCompare(b.column))
}

export function formatNumber(val: number | null | undefined, precision = 2): string {
  if (val === null || val === undefined || isNaN(val)) return '—'
  if (Math.abs(val) >= 1_000_000 || (Math.abs(val) > 0 && Math.abs(val) < 0.001)) {
    return val.toExponential(3)
  }
  return Number.isInteger(val)
    ? val.toLocaleString()
    : val.toLocaleString(undefined, {
        minimumFractionDigits: 0,
        maximumFractionDigits: precision,
      })
}
