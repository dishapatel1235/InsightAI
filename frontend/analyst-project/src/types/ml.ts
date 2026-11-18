export type TaskType = 'classification' | 'regression'
export type ModelType = 'random_forest' | 'xgboost' | 'neural_network'
export type PatternMethod = 'clustering' | 'correlation' | 'anomaly'
export type TimeSeriesModel = 'lstm' | 'neural_net' | 'random_forest'

export interface ClassificationMetrics {
  accuracy: number
  train_accuracy: number
  precision: number
  recall: number
  f1_score: number
  confusion_matrix: number[][]
  classes: string[]
  sample_predictions: {
    index: number
    actual: string
    predicted: string
    correct: boolean
  }[]
}

export interface RegressionMetrics {
  r2_score: number
  train_r2_score: number
  rmse: number
  mae: number
  mse: number
  mape_pct: number
  sample_predictions: {
    index: number
    actual: number
    predicted: number
    residual: number
  }[]
}

export interface FeatureImportance {
  feature: string
  importance: number
}

export interface PredictiveModelResult {
  model_id: string
  task_type: TaskType
  model_name: string
  target_column: string
  feature_columns: string[]
  metrics: ClassificationMetrics | RegressionMetrics
  feature_importances: FeatureImportance[]
  test_sample_count: number
  train_sample_count: number
}

export interface ClusterProfile {
  cluster_id: number
  name: string
  size: number
  percentage: number
  centroid_pca: { x: number; y: number }
  feature_means: Record<string, number>
}

export interface ScatterPoint {
  id: number
  x: number
  y: number
  cluster: number
}

export interface ClusteringResult {
  n_clusters: number
  silhouette_score: number
  inertia: number
  explained_variance_ratio: number[]
  cluster_profiles: ClusterProfile[]
  scatter_points: ScatterPoint[]
}

export interface CorrelationResult {
  columns: string[]
  matrix: Record<string, Record<string, number>>
  top_pairs: {
    feature_a: string
    feature_b: string
    correlation: number
    abs_corr: number
  }[]
}

export interface AnomalyItem {
  row_index: number
  anomaly_score: number
  features: Record<string, number>
}

export interface AnomalyResult {
  detected_anomalies: number
  anomaly_rate_pct: number
  top_anomalies: AnomalyItem[]
}

export interface PatternDiscoveryResult {
  method: PatternMethod | 'all'
  features_analyzed: string[]
  total_samples: number
  clustering?: ClusteringResult
  correlation?: CorrelationResult
  anomaly_detection?: AnomalyResult
}

export interface TimeSeriesPoint {
  label: string
  actual: number | null
  fitted: number | null
  forecast: number | null
  lower_bound: number | null
  upper_bound: number | null
}

export interface TimeSeriesForecastResult {
  model_name: string
  target_column: string
  lookback_window: number
  horizon: number
  metrics: {
    mae: number
    rmse: number
    mape_pct: number
  }
  timeline: TimeSeriesPoint[]
  future_forecast: {
    step: number
    forecast: number
    lower_bound: number
    upper_bound: number
  }[]
  last_observed_value: number
}

export interface LivePredictionResult {
  prediction: string | number
  probabilities?: Record<string, number> | null
  task_type: TaskType
  model_name: string
}
