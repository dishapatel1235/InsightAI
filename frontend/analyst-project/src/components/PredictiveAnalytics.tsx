import React, { useState } from 'react'
import type { DatasetResult } from '../types/analytics'
import type {
  ClassificationMetrics,
  LivePredictionResult,
  ModelType,
  PredictiveModelResult,
  RegressionMetrics,
  TaskType,
} from '../types/ml'
import { formatNumber, predictLive, trainPredictiveModel } from '../services/api'
import {
  downloadCsv,
  downloadJson,
  generateMlFeatureImportanceCsv,
  generateMlPredictionsCsv,
} from '../utils/exportUtils'

interface PredictiveAnalyticsProps {
  dataset: DatasetResult
  backendUrl: string
  initialResult?: PredictiveModelResult | null
  onResultTrained?: (res: PredictiveModelResult) => void
}

export const PredictiveAnalytics: React.FC<PredictiveAnalyticsProps> = ({
  dataset,
  backendUrl,
  initialResult,
  onResultTrained,
}) => {
  const columns = dataset.columns_list || Object.keys(dataset.preview?.[0] || {})

  // Compute smart initial defaults
  const initialTarget = columns.length > 1 ? columns[columns.length - 1] : ''
  const initialFeatures = columns.length > 1 ? columns.slice(0, columns.length - 1) : []
  const initialTaskType: TaskType = (() => {
    if (dataset.preview && dataset.preview.length > 0 && initialTarget) {
      const sampleVal = dataset.preview[0][initialTarget]
      if (typeof sampleVal === 'number' && sampleVal % 1 !== 0) {
        return 'regression'
      }
      if (typeof sampleVal === 'string') {
        return 'classification'
      }
    }
    return 'classification'
  })()

  // State for configuration
  const [taskType, setTaskType] = useState<TaskType>(initialTaskType)
  const [targetColumn, setTargetColumn] = useState<string>(initialTarget)
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>(initialFeatures)
  const [modelName, setModelName] = useState<ModelType>('random_forest')

  // Hyperparameters
  const [nEstimators, setNEstimators] = useState<number>(100)
  const [maxDepth, setMaxDepth] = useState<number>(6)
  const [learningRate, setLearningRate] = useState<number>(0.1)
  const [epochs, setEpochs] = useState<number>(100)

  // Execution & Results
  const [isTraining, setIsTraining] = useState<boolean>(false)
  const [trainError, setTrainError] = useState<string | null>(null)
  const [result, setResult] = useState<PredictiveModelResult | null>(initialResult || null)

  // Live "What-If" Inference State
  const [liveInputs, setLiveInputs] = useState<Record<string, string>>(() => {
    if (initialResult) {
      const initialMap: Record<string, string> = {}
      initialResult.feature_columns.forEach((feat) => {
        const sampleVal = dataset.preview?.[0]?.[feat]
        initialMap[feat] = sampleVal !== undefined && sampleVal !== null ? String(sampleVal) : '10'
      })
      return initialMap
    }
    return {}
  })
  const [livePrediction, setLivePrediction] = useState<LivePredictionResult | null>(null)
  const [isPredicting, setIsPredicting] = useState<boolean>(false)

  const handleToggleFeature = (col: string) => {
    if (selectedFeatures.includes(col)) {
      if (selectedFeatures.length > 1) {
        setSelectedFeatures(selectedFeatures.filter((f) => f !== col))
      }
    } else {
      setSelectedFeatures([...selectedFeatures, col])
    }
  }

  const handleTrain = async () => {
    if (!targetColumn) {
      setTrainError('Please select a target column to predict.')
      return
    }
    if (selectedFeatures.length === 0) {
      setTrainError('Please select at least one predictor feature.')
      return
    }

    setIsTraining(true)
    setTrainError(null)

    try {
      const res = await trainPredictiveModel({
        fileName: dataset.file_name,
        targetColumn,
        taskType,
        modelName,
        featureColumns: selectedFeatures,
        hyperparameters: {
          n_estimators: nEstimators,
          max_depth: maxDepth,
          learning_rate: learningRate,
          epochs,
        },
        baseUrl: backendUrl,
      })
      setResult(res)
      onResultTrained?.(res)

      // Initialize live inputs
      const initialMap: Record<string, string> = {}
      res.feature_columns.forEach((feat) => {
        const sampleVal = dataset.preview?.[0]?.[feat]
        initialMap[feat] = sampleVal !== undefined && sampleVal !== null ? String(sampleVal) : '10'
      })
      setLiveInputs(initialMap)
      setLivePrediction(null)
    } catch (err: unknown) {
      setTrainError(err instanceof Error ? err.message : 'Training failed.')
    } finally {
      setIsTraining(false)
    }
  }

  const handleLivePredict = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!result?.model_id) return

    setIsPredicting(true)
    try {
      const res = await predictLive({
        modelId: result.model_id,
        inputFeatures: liveInputs,
        baseUrl: backendUrl,
      })
      setLivePrediction(res)
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : 'Prediction failed')
    } finally {
      setIsPredicting(false)
    }
  }

  const classMetrics = result?.task_type === 'classification' ? (result.metrics as ClassificationMetrics) : null
  const regMetrics = result?.task_type === 'regression' ? (result.metrics as RegressionMetrics) : null

  return (
    <div className="ml-module-container">
      {/* Configuration & Training Control Card */}
      <div className="card ml-config-card">
        <div className="card-header-bar">
          <div>
            <h3 className="card-title">Supervised Predictive Modeling</h3>
            <p className="card-subtitle">
              Train Random Forest, XGBoost, and Deep Neural Networks for classification and regression
            </p>
          </div>
          <div className="task-type-toggle">
            <button
              type="button"
              className={`pill-btn ${taskType === 'classification' ? 'pill-active' : ''}`}
              onClick={() => setTaskType('classification')}
            >
              Classification
            </button>
            <button
              type="button"
              className={`pill-btn ${taskType === 'regression' ? 'pill-active' : ''}`}
              onClick={() => setTaskType('regression')}
            >
              Regression
            </button>
          </div>
        </div>

        {trainError && (
          <div className="error-banner" style={{ margin: '16px 20px 0' }}>
            <span>{trainError}</span>
            <button type="button" className="close-btn" onClick={() => setTrainError(null)}>×</button>
          </div>
        )}

        <div className="config-body-grid">
          {/* Column Target & Feature Selectors */}
          <div className="config-column-section">
            <div className="form-group">
              <label className="form-label">
                Target Variable ($y$):
                <span className="label-sub">What you want the model to predict</span>
              </label>
              <select
                value={targetColumn}
                onChange={(e) => {
                  setTargetColumn(e.target.value)
                  setSelectedFeatures(columns.filter((c) => c !== e.target.value))
                }}
                className="select-input"
              >
                {columns.map((c) => (
                  <option key={c} value={c}>
                    {c} {c === columns[columns.length - 1] ? '(Suggested)' : ''}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">
                Predictor Features ($X$):
                <span className="label-sub">{selectedFeatures.length} of {columns.length - 1} selected</span>
              </label>
              <div className="features-checklist">
                {columns
                  .filter((c) => c !== targetColumn)
                  .map((col) => {
                    const isChecked = selectedFeatures.includes(col)
                    return (
                      <button
                        key={col}
                        type="button"
                        className={`feature-chip ${isChecked ? 'feature-chip-active' : ''}`}
                        onClick={() => handleToggleFeature(col)}
                      >
                        <span className="chip-icon">{isChecked ? '✓' : '+'}</span>
                        <span className="chip-text">{col}</span>
                      </button>
                    )
                  })}
              </div>
            </div>
          </div>

          {/* Model Architecture & Hyperparameters */}
          <div className="config-model-section">
            <div className="form-group">
              <label className="form-label">Algorithm & Model Architecture:</label>
              <div className="model-selector-grid">
                <div
                  className={`model-card-option ${modelName === 'random_forest' ? 'model-card-active' : ''}`}
                  onClick={() => setModelName('random_forest')}
                >
                  <span className="model-icon">🌲</span>
                  <div>
                    <strong>Random Forest</strong>
                    <p>Ensemble of bagged decision trees</p>
                  </div>
                </div>

                <div
                  className={`model-card-option ${modelName === 'xgboost' ? 'model-card-active' : ''}`}
                  onClick={() => setModelName('xgboost')}
                >
                  <span className="model-icon">⚡</span>
                  <div>
                    <strong>XGBoost / Gradient Boost</strong>
                    <p>High-accuracy sequential boosting</p>
                  </div>
                </div>

                <div
                  className={`model-card-option ${modelName === 'neural_network' ? 'model-card-active' : ''}`}
                  onClick={() => setModelName('neural_network')}
                >
                  <span className="model-icon">🧠</span>
                  <div>
                    <strong>Deep Neural Network</strong>
                    <p>Multi-Layer Perceptron (MLP)</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Hyperparameter Inputs */}
            <div className="hyperparam-grid">
              {modelName !== 'neural_network' ? (
                <>
                  <div className="param-item">
                    <label>Estimators (Trees): {nEstimators}</label>
                    <input
                      type="range"
                      min={10}
                      max={300}
                      step={10}
                      value={nEstimators}
                      onChange={(e) => setNEstimators(Number(e.target.value))}
                    />
                  </div>
                  <div className="param-item">
                    <label>Max Tree Depth: {maxDepth}</label>
                    <input
                      type="range"
                      min={2}
                      max={20}
                      value={maxDepth}
                      onChange={(e) => setMaxDepth(Number(e.target.value))}
                    />
                  </div>
                </>
              ) : (
                <>
                  <div className="param-item">
                    <label>Training Epochs: {epochs}</label>
                    <input
                      type="range"
                      min={20}
                      max={400}
                      step={20}
                      value={epochs}
                      onChange={(e) => setEpochs(Number(e.target.value))}
                    />
                  </div>
                  <div className="param-item">
                    <label>Learning Rate: {learningRate}</label>
                    <select
                      value={learningRate}
                      onChange={(e) => setLearningRate(Number(e.target.value))}
                      className="select-input select-sm"
                    >
                      <option value={0.001}>0.001 (Fine)</option>
                      <option value={0.01}>0.01 (Standard)</option>
                      <option value={0.1}>0.1 (Fast)</option>
                    </select>
                  </div>
                </>
              )}
            </div>

            <button
              type="button"
              className="btn btn-primary btn-block train-btn"
              disabled={isTraining}
              onClick={handleTrain}
            >
              {isTraining ? (
                <>
                  <span className="spinner spinner-sm"></span>
                  <span>Training {modelName}...</span>
                </>
              ) : (
                <>
                  <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" strokeWidth="2" fill="none">
                    <polygon points="5 3 19 12 5 21 5 3" />
                  </svg>
                  <span>Train & Evaluate Model</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Model Evaluation Dashboard */}
      {result && (
        <div className="ml-results-wrapper">
          {/* Quick Tab Export Bar */}
          <div className="results-header-row">
            <div>
              <h3 style={{ fontSize: '16px', fontWeight: 700, margin: 0, color: 'var(--text-main)' }}>
                {result.model_name} — {result.task_type.toUpperCase()} Outcome
              </h3>
              <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                Target: <strong>{result.target_column}</strong> | Predictor Features: {result.feature_columns.length}
              </span>
            </div>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                title="Download sample predictions with actual vs predicted values"
                onClick={() => {
                  const pRes = generateMlPredictionsCsv(result)
                  if (pRes) {
                    downloadCsv(pRes.headers, pRes.rows, `${dataset.file_name.replace(/\.[^/.]+$/, '')}_ml_predictions.csv`)
                  }
                }}
              >
                <svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" strokeWidth="2" fill="none">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="7 10 12 15 17 10" />
                  <line x1="12" y1="15" x2="12" y2="3" />
                </svg>
                <span>Predictions (.csv)</span>
              </button>

              <button
                type="button"
                className="btn btn-secondary btn-sm"
                title="Download ranked feature importance matrix"
                onClick={() => {
                  const fiRes = generateMlFeatureImportanceCsv(result)
                  if (fiRes) {
                    downloadCsv(fiRes.headers, fiRes.rows, `${dataset.file_name.replace(/\.[^/.]+$/, '')}_feature_importances.csv`)
                  }
                }}
              >
                <span>Features (.csv)</span>
              </button>

              <button
                type="button"
                className="btn btn-secondary btn-sm"
                title="Download complete trained model evaluation JSON"
                onClick={() => {
                  downloadJson(result, `${dataset.file_name.replace(/\.[^/.]+$/, '')}_predictive_model.json`)
                }}
              >
                <span>Model (.json)</span>
              </button>
            </div>
          </div>

          {/* Performance Metrics Cards */}
          <div className="kpi-grid">
            {classMetrics ? (
              <>
                <div className="kpi-card">
                  <div className="kpi-icon-wrapper kpi-icon-emerald">🎯</div>
                  <div className="kpi-content">
                    <span className="kpi-label">Test Accuracy</span>
                    <span className="kpi-value">{(classMetrics.accuracy * 100).toFixed(1)}%</span>
                    <span className="kpi-caption">Train: {(classMetrics.train_accuracy * 100).toFixed(1)}%</span>
                  </div>
                </div>

                <div className="kpi-card">
                  <div className="kpi-icon-wrapper kpi-icon-blue">📊</div>
                  <div className="kpi-content">
                    <span className="kpi-label">F1-Score (Weighted)</span>
                    <span className="kpi-value">{classMetrics.f1_score.toFixed(3)}</span>
                    <span className="kpi-caption">Harmonic precision/recall</span>
                  </div>
                </div>

                <div className="kpi-card">
                  <div className="kpi-icon-wrapper kpi-icon-purple">⚖️</div>
                  <div className="kpi-content">
                    <span className="kpi-label">Precision / Recall</span>
                    <span className="kpi-value">{classMetrics.precision.toFixed(2)}</span>
                    <span className="kpi-caption">Recall: {classMetrics.recall.toFixed(2)}</span>
                  </div>
                </div>

                <div className="kpi-card">
                  <div className="kpi-icon-wrapper kpi-icon-amber">📦</div>
                  <div className="kpi-content">
                    <span className="kpi-label">Data Partition</span>
                    <span className="kpi-value">{result.test_sample_count}</span>
                    <span className="kpi-caption">Test samples ({result.train_sample_count} Train)</span>
                  </div>
                </div>
              </>
            ) : regMetrics ? (
              <>
                <div className="kpi-card">
                  <div className="kpi-icon-wrapper kpi-icon-emerald">📈</div>
                  <div className="kpi-content">
                    <span className="kpi-label">R² Goodness of Fit</span>
                    <span className="kpi-value">{regMetrics.r2_score.toFixed(3)}</span>
                    <span className="kpi-caption">Variance explained</span>
                  </div>
                </div>

                <div className="kpi-card">
                  <div className="kpi-icon-wrapper kpi-icon-blue">📉</div>
                  <div className="kpi-content">
                    <span className="kpi-label">RMSE (Root MSE)</span>
                    <span className="kpi-value">{formatNumber(regMetrics.rmse)}</span>
                    <span className="kpi-caption">Std error of predictions</span>
                  </div>
                </div>

                <div className="kpi-card">
                  <div className="kpi-icon-wrapper kpi-icon-purple">📏</div>
                  <div className="kpi-content">
                    <span className="kpi-label">Mean Absolute Error</span>
                    <span className="kpi-value">{formatNumber(regMetrics.mae)}</span>
                    <span className="kpi-caption">Average absolute residual</span>
                  </div>
                </div>

                <div className="kpi-card">
                  <div className="kpi-icon-wrapper kpi-icon-amber">🎯</div>
                  <div className="kpi-content">
                    <span className="kpi-label">Mean Abs Percentage (MAPE)</span>
                    <span className="kpi-value">{regMetrics.mape_pct}%</span>
                    <span className="kpi-caption">Relative error percentage</span>
                  </div>
                </div>
              </>
            ) : null}
          </div>

          <div className="results-grid-two">
            {/* Visual 1: Confusion Matrix OR Actual vs Predicted Scatter */}
            <div className="card">
              <div className="card-header-bar">
                <h4 className="card-title">
                  {classMetrics ? 'Confusion Matrix Heatmap' : 'Actual vs Predicted Scatter Plot'}
                </h4>
                <span className="badge badge-info">{result.model_name}</span>
              </div>
              <div style={{ padding: '20px' }}>
                {classMetrics ? (
                  <div className="confusion-matrix-wrapper">
                    <div className="cm-grid">
                      {classMetrics.confusion_matrix.map((row, rIdx) => (
                        <div key={rIdx} className="cm-row">
                          <span className="cm-label">
                            {classMetrics.classes?.[rIdx] || `Class ${rIdx}`}
                          </span>
                          <div className="cm-cells">
                            {row.map((val, cIdx) => {
                              const isDiagonal = rIdx === cIdx
                              return (
                                <div
                                  key={cIdx}
                                  className={`cm-cell ${isDiagonal ? 'cm-diagonal' : 'cm-off-diagonal'}`}
                                  title={`Actual: ${classMetrics.classes?.[rIdx]} | Predicted: ${classMetrics.classes?.[cIdx]}`}
                                >
                                  <span className="cm-val">{val}</span>
                                </div>
                              )
                            })}
                          </div>
                        </div>
                      ))}
                    </div>
                    <div className="cm-legend">
                      <span>Row = Actual Ground Truth</span>
                      <span>Column = Model Prediction</span>
                    </div>
                  </div>
                ) : regMetrics ? (
                  <div className="regression-scatter-wrapper">
                    <div className="scatter-axis-container">
                      <div className="scatter-canvas">
                        {regMetrics.sample_predictions.map((p, idx) => {
                          const allVals = regMetrics.sample_predictions.flatMap((i) => [i.actual, i.predicted])
                          const minV = Math.min(...allVals)
                          const maxV = Math.max(...allVals)
                          const range = maxV - minV || 1
                          const x = ((p.actual - minV) / range) * 90 + 5
                          const y = 95 - (((p.predicted - minV) / range) * 90 + 5)
                          return (
                            <div
                              key={idx}
                              className="scatter-point"
                              style={{ left: `${x}%`, top: `${y}%` }}
                              title={`Actual: ${p.actual} | Pred: ${p.predicted} (Res: ${p.residual})`}
                            />
                          )
                        })}
                        {/* 45 degree perfect prediction line */}
                        <div className="ideal-line" />
                      </div>
                      <div className="scatter-labels">
                        <span>Min: {formatNumber(Math.min(...regMetrics.sample_predictions.map((p) => p.actual)))}</span>
                        <span>Actual vs Predicted (Dashed line = Ideal 1:1 prediction)</span>
                        <span>Max: {formatNumber(Math.max(...regMetrics.sample_predictions.map((p) => p.actual)))}</span>
                      </div>
                    </div>
                  </div>
                ) : null}
              </div>
            </div>

            {/* Visual 2: Feature Importance Ranking */}
            <div className="card">
              <div className="card-header-bar">
                <h4 className="card-title">Feature Importance Ranking</h4>
                <span className="text-muted text-xs">Relative contribution to predictions</span>
              </div>
              <div style={{ padding: '20px' }}>
                <div className="feature-importance-list">
                  {result.feature_importances.map((item, idx) => {
                    const maxImp = result.feature_importances[0]?.importance || 1
                    const pct = Math.max(8, (item.importance / maxImp) * 100)
                    return (
                      <div key={item.feature} className="importance-row">
                        <div className="importance-labels">
                          <span className="feat-rank">#{idx + 1}</span>
                          <span className="feat-name">{item.feature}</span>
                          <span className="feat-val">{(item.importance * 100).toFixed(1)}%</span>
                        </div>
                        <div className="importance-track">
                          <div className="importance-fill" style={{ width: `${pct}%` }} />
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            </div>
          </div>

          {/* Interactive "What-If" Live Predictor Card */}
          <div className="card live-predict-card">
            <div className="card-header-bar">
              <div>
                <h4 className="card-title">Interactive "What-If" Real-Time Predictor</h4>
                <p className="card-subtitle">
                  Adjust feature inputs below to simulate instant real-time inference with the trained {result.model_name}
                </p>
              </div>
              <span className="status-pill status-online">Model Active</span>
            </div>

            <form onSubmit={handleLivePredict} className="live-predict-form">
              <div className="live-inputs-grid">
                {result.feature_columns.map((feat) => (
                  <div key={feat} className="live-input-item">
                    <label>{feat}:</label>
                    <input
                      type="text"
                      value={liveInputs[feat] || ''}
                      onChange={(e) => setLiveInputs({ ...liveInputs, [feat]: e.target.value })}
                      placeholder="e.g. 25"
                      className="config-input"
                    />
                  </div>
                ))}
              </div>

              <div className="live-predict-actions">
                <button type="submit" className="btn btn-primary" disabled={isPredicting}>
                  {isPredicting ? 'Inferring...' : 'Compute Instant Prediction'}
                </button>

                {livePrediction && (
                  <div className="prediction-outcome-pill">
                    <span className="outcome-label">Model Output:</span>
                    <strong className="outcome-val">{String(livePrediction.prediction)}</strong>
                    {livePrediction.probabilities && (
                      <span className="outcome-confidence">
                        (Confidence: {(Math.max(...Object.values(livePrediction.probabilities)) * 100).toFixed(1)}%)
                      </span>
                    )}
                  </div>
                )}
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
