import React, { useState } from 'react'
import type { DatasetResult } from '../types/analytics'
import type { TimeSeriesForecastResult, TimeSeriesModel } from '../types/ml'
import { forecastTimeSeries, formatNumber } from '../services/api'
import {
  downloadCsv,
  downloadJson,
  generateForecastTimelineCsv,
} from '../utils/exportUtils'

interface TimeSeriesForecastingProps {
  dataset: DatasetResult
  backendUrl: string
  initialResult?: TimeSeriesForecastResult | null
  onResultForecasted?: (res: TimeSeriesForecastResult) => void
}

export const TimeSeriesForecasting: React.FC<TimeSeriesForecastingProps> = ({
  dataset,
  backendUrl,
  initialResult,
  onResultForecasted,
}) => {
  const columns = dataset.columns_list || Object.keys(dataset.preview?.[0] || {})

  // Compute smart initial date and target columns
  const initialTimeCol = (() => {
    const dateCandidate = columns.find((c) =>
      ['date', 'time', 'timestamp', 'day', 'month', 'year'].some((term) =>
        c.toLowerCase().includes(term)
      )
    )
    return dateCandidate || ''
  })()

  const initialTarget = (() => {
    const numericCandidate = columns.find(
      (c) =>
        c !== initialTimeCol &&
        ['sales', 'revenue', 'price', 'demand', 'value', 'total'].some((term) =>
          c.toLowerCase().includes(term)
        )
    )
    return numericCandidate || (columns.length > 0 ? columns[columns.length - 1] : '')
  })()

  const [timeColumn, setTimeColumn] = useState<string>(initialTimeCol)
  const [targetColumn, setTargetColumn] = useState<string>(initialTarget)
  const [modelName, setModelName] = useState<TimeSeriesModel>('lstm')
  const [lookback, setLookback] = useState<number>(5)
  const [horizon, setHorizon] = useState<number>(10)
  const [epochs, setEpochs] = useState<number>(50)

  const [isForecasting, setIsForecasting] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<TimeSeriesForecastResult | null>(initialResult || null)

  const handleRunForecast = async () => {
    if (!targetColumn) {
      setError('Please select a numeric target metric to forecast.')
      return
    }

    setIsForecasting(true)
    setError(null)

    try {
      const res = await forecastTimeSeries({
        fileName: dataset.file_name,
        targetColumn,
        timeColumn: timeColumn || undefined,
        modelName,
        lookback,
        horizon,
        epochs,
        baseUrl: backendUrl,
      })
      setResult(res)
      onResultForecasted?.(res)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Forecasting failed.')
    } finally {
      setIsForecasting(false)
    }
  }

  let minVal = 0
  let maxVal = 1
  if (result && result.timeline.length > 0) {
    const allNums = result.timeline
      .flatMap((p) => [p.actual, p.fitted, p.forecast, p.lower_bound, p.upper_bound])
      .filter((v): v is number => v !== null && !isNaN(v))
    minVal = Math.min(...allNums)
    maxVal = Math.max(...allNums)
    if (minVal === maxVal) maxVal += 1
  }
  const valRange = maxVal - minVal

  return (
    <div className="ml-module-container">
      {/* Configuration Card */}
      <div className="card ml-config-card">
        <div className="card-header-bar">
          <div>
            <h3 className="card-title">Deep Learning Time-Series Forecasting</h3>
            <p className="card-subtitle">
              Train Recurrent LSTM Neural Networks and autoregressive models to project future trends and confidence intervals
            </p>
          </div>
          <span className="badge badge-purple">Sequential DL</span>
        </div>

        {error && (
          <div className="error-banner" style={{ margin: '16px 20px 0' }}>
            <span>{error}</span>
            <button type="button" className="close-btn" onClick={() => setError(null)}>×</button>
          </div>
        )}

        <div className="config-body-grid">
          {/* Target & Time Column selectors */}
          <div className="config-column-section">
            <div className="form-group">
              <label className="form-label">
                Target Metric to Forecast:
                <span className="label-sub">Continuous numeric variable over time</span>
              </label>
              <select
                value={targetColumn}
                onChange={(e) => setTargetColumn(e.target.value)}
                className="select-input"
              >
                {columns.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">
                Date / Sequence Column:
                <span className="label-sub">Optional: sorted automatically if provided</span>
              </label>
              <select
                value={timeColumn}
                onChange={(e) => setTimeColumn(e.target.value)}
                className="select-input"
              >
                <option value="">Index / Observation Order (Auto)</option>
                {columns.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Model Architecture & Forecast Parameters */}
          <div className="config-model-section">
            <div className="form-group">
              <label className="form-label">Forecasting Architecture:</label>
              <div className="model-selector-grid">
                <div
                  className={`model-card-option ${modelName === 'lstm' ? 'model-card-active' : ''}`}
                  onClick={() => setModelName('lstm')}
                >
                  <span className="model-icon">🧠</span>
                  <div>
                    <strong>LSTM Neural Network</strong>
                    <p>Long Short-Term Memory cells</p>
                  </div>
                </div>

                <div
                  className={`model-card-option ${modelName === 'neural_net' ? 'model-card-active' : ''}`}
                  onClick={() => setModelName('neural_net')}
                >
                  <span className="model-icon">⚡</span>
                  <div>
                    <strong>Deep Neural Net (MLP)</strong>
                    <p>Dense multi-layer lag network</p>
                  </div>
                </div>

                <div
                  className={`model-card-option ${modelName === 'random_forest' ? 'model-card-active' : ''}`}
                  onClick={() => setModelName('random_forest')}
                >
                  <span className="model-icon">🌲</span>
                  <div>
                    <strong>Sequential RF Forest</strong>
                    <p>Lagged ensemble regression</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="hyperparam-grid">
              <div className="param-item">
                <label>Lookback Window: <strong>{lookback} steps</strong></label>
                <input
                  type="range"
                  min={2}
                  max={15}
                  value={lookback}
                  onChange={(e) => setLookback(Number(e.target.value))}
                />
              </div>

              <div className="param-item">
                <label>Future Horizon: <strong>{horizon} periods ahead</strong></label>
                <input
                  type="range"
                  min={3}
                  max={30}
                  value={horizon}
                  onChange={(e) => setHorizon(Number(e.target.value))}
                />
              </div>

              <div className="param-item">
                <label>Training Epochs: <strong>{epochs}</strong></label>
                <input
                  type="range"
                  min={20}
                  max={120}
                  step={10}
                  value={epochs}
                  onChange={(e) => setEpochs(Number(e.target.value))}
                />
              </div>
            </div>

            <button
              type="button"
              className="btn btn-primary btn-block train-btn"
              disabled={isForecasting}
              onClick={handleRunForecast}
            >
              {isForecasting ? (
                <>
                  <span className="spinner spinner-sm"></span>
                  <span>Training {modelName.toUpperCase()} & Projecting...</span>
                </>
              ) : (
                <>
                  <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" strokeWidth="2" fill="none">
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
                  </svg>
                  <span>Train & Project Horizon</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Results View */}
      {result && (
        <div className="ml-results-wrapper">
          {/* Quick Tab Export Bar */}
          <div className="results-header-row">
            <div>
              <h3 style={{ fontSize: '16px', fontWeight: 700, margin: 0, color: 'var(--text-main)' }}>
                {result.model_name} — Forecasting Outcome
              </h3>
              <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                Target: <strong>{result.target_column}</strong> | Lookback Window: {result.lookback_window} | Horizon: {result.horizon} steps
              </span>
            </div>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                title="Download historical, fitted, and future forecast points with confidence intervals"
                onClick={() => {
                  const tRes = generateForecastTimelineCsv(result)
                  if (tRes) {
                    downloadCsv(tRes.headers, tRes.rows, `${dataset.file_name.replace(/\.[^/.]+$/, '')}_forecast_timeline.csv`)
                  }
                }}
              >
                <svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" strokeWidth="2" fill="none">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="7 10 12 15 17 10" />
                  <line x1="12" y1="15" x2="12" y2="3" />
                </svg>
                <span>Timeline (.csv)</span>
              </button>

              <button
                type="button"
                className="btn btn-secondary btn-sm"
                title="Download forecast results JSON"
                onClick={() => {
                  downloadJson(result, `${dataset.file_name.replace(/\.[^/.]+$/, '')}_time_series_forecast.json`)
                }}
              >
                <span>Forecast (.json)</span>
              </button>
            </div>
          </div>

          {/* KPI Scorecards */}
          <div className="kpi-grid">
            <div className="kpi-card">
              <div className="kpi-icon-wrapper kpi-icon-emerald">🎯</div>
              <div className="kpi-content">
                <span className="kpi-label">Mean Absolute Error (MAE)</span>
                <span className="kpi-value">{formatNumber(result.metrics.mae)}</span>
                <span className="kpi-caption">Validation test residual</span>
              </div>
            </div>

            <div className="kpi-card">
              <div className="kpi-icon-wrapper kpi-icon-blue">📉</div>
              <div className="kpi-content">
                <span className="kpi-label">RMSE (Root MSE)</span>
                <span className="kpi-value">{formatNumber(result.metrics.rmse)}</span>
                <span className="kpi-caption">Standard deviation of errors</span>
              </div>
            </div>

            <div className="kpi-card">
              <div className="kpi-icon-wrapper kpi-icon-purple">📏</div>
              <div className="kpi-content">
                <span className="kpi-label">Percentage Error (MAPE)</span>
                <span className="kpi-value">{result.metrics.mape_pct}%</span>
                <span className="kpi-caption">Relative accuracy rate</span>
              </div>
            </div>

            <div className="kpi-card">
              <div className="kpi-icon-wrapper kpi-icon-amber">🔮</div>
              <div className="kpi-content">
                <span className="kpi-label">Forecast Horizon</span>
                <span className="kpi-value">+{result.horizon}</span>
                <span className="kpi-caption">Periods into future</span>
              </div>
            </div>
          </div>

          {/* Time Series Interactive Visual Canvas */}
          <div className="card">
            <div className="card-header-bar">
              <div>
                <h4 className="card-title">Trajectory & Confidence Interval Projection</h4>
                <p className="card-subtitle">
                  Historical actuals, model fitted line, and out-of-sample forecast with 95% confidence envelope
                </p>
              </div>

              <div className="chart-legend-row">
                <span className="legend-item"><span className="legend-dot" style={{ background: '#38bdf8' }}></span> Historical Actual</span>
                <span className="legend-item"><span className="legend-dot" style={{ background: '#10b981' }}></span> In-Sample Fit</span>
                <span className="legend-item"><span className="legend-dot" style={{ background: '#a855f7' }}></span> Horizon Forecast</span>
                <span className="legend-item"><span className="legend-dot" style={{ background: 'rgba(168, 85, 247, 0.35)' }}></span> 95% Confidence Band</span>
              </div>
            </div>

            <div style={{ padding: '24px' }}>
              <div className="ts-chart-container">
                <svg viewBox="0 0 800 320" className="ts-chart-svg" preserveAspectRatio="none">
                  <line x1="0" y1="40" x2="800" y2="40" stroke="rgba(255,255,255,0.06)" strokeDasharray="4 4" />
                  <line x1="0" y1="120" x2="800" y2="120" stroke="rgba(255,255,255,0.06)" strokeDasharray="4 4" />
                  <line x1="0" y1="200" x2="800" y2="200" stroke="rgba(255,255,255,0.06)" strokeDasharray="4 4" />
                  <line x1="0" y1="280" x2="800" y2="280" stroke="rgba(255,255,255,0.06)" strokeDasharray="4 4" />

                  {/* Confidence Interval Polygon */}
                  {(() => {
                    const forecastPoints = result.timeline
                      .map((p, idx) => ({ ...p, idx }))
                      .filter((p) => p.forecast !== null && p.lower_bound !== null && p.upper_bound !== null)
                    if (forecastPoints.length === 0) return null

                    const totalPoints = result.timeline.length
                    const getX = (idx: number) => (idx / (totalPoints - 1)) * 800
                    const getY = (val: number) => 300 - ((val - minVal) / valRange) * 260

                    const upperPath = forecastPoints.map((p) => `${getX(p.idx)},${getY(p.upper_bound!)}`).join(' L ')
                    const lowerPath = [...forecastPoints].reverse().map((p) => `${getX(p.idx)},${getY(p.lower_bound!)}`).join(' L ')
                    const polyD = `M ${upperPath} L ${lowerPath} Z`

                    return <path d={polyD} fill="rgba(168, 85, 247, 0.15)" stroke="none" />
                  })()}

                  {/* Historical Line (Actuals) */}
                  {(() => {
                    const actualPoints = result.timeline
                      .map((p, idx) => ({ ...p, idx }))
                      .filter((p) => p.actual !== null)
                    if (actualPoints.length === 0) return null
                    const totalPoints = result.timeline.length
                    const getX = (idx: number) => (idx / (totalPoints - 1)) * 800
                    const getY = (val: number) => 300 - ((val - minVal) / valRange) * 260

                    const d = actualPoints.map((p, i) => `${i === 0 ? 'M' : 'L'} ${getX(p.idx)} ${getY(p.actual!)}`).join(' ')
                    return <path d={d} fill="none" stroke="#38bdf8" strokeWidth="2.5" />
                  })()}

                  {/* Fitted Line */}
                  {(() => {
                    const fittedPoints = result.timeline
                      .map((p, idx) => ({ ...p, idx }))
                      .filter((p) => p.fitted !== null)
                    if (fittedPoints.length === 0) return null
                    const totalPoints = result.timeline.length
                    const getX = (idx: number) => (idx / (totalPoints - 1)) * 800
                    const getY = (val: number) => 300 - ((val - minVal) / valRange) * 260

                    const d = fittedPoints.map((p, i) => `${i === 0 ? 'M' : 'L'} ${getX(p.idx)} ${getY(p.fitted!)}`).join(' ')
                    return <path d={d} fill="none" stroke="#10b981" strokeWidth="2" strokeDasharray="3 3" />
                  })()}

                  {/* Forecast Line */}
                  {(() => {
                    const forecastPoints = result.timeline
                      .map((p, idx) => ({ ...p, idx }))
                      .filter((p) => p.forecast !== null)
                    if (forecastPoints.length === 0) return null
                    const totalPoints = result.timeline.length
                    const getX = (idx: number) => (idx / (totalPoints - 1)) * 800
                    const getY = (val: number) => 300 - ((val - minVal) / valRange) * 260

                    const lastActualIdx = result.timeline.length - result.horizon - 1
                    const lastActual = result.timeline[lastActualIdx]
                    let startPrefix = ''
                    if (lastActual && lastActual.actual !== null) {
                      startPrefix = `M ${getX(lastActualIdx)} ${getY(lastActual.actual)} L `
                    }

                    const d = startPrefix + forecastPoints.map((p, i) => `${!startPrefix && i === 0 ? 'M' : 'L'} ${getX(p.idx)} ${getY(p.forecast!)}`).join(' ')
                    return <path d={d} fill="none" stroke="#a855f7" strokeWidth="3" />
                  })()}
                </svg>

                <div className="chart-x-axis">
                  <span>{result.timeline[0]?.label}</span>
                  <span>{result.timeline[result.timeline.length - result.horizon - 1]?.label} (Forecast Start)</span>
                  <span>{result.timeline[result.timeline.length - 1]?.label} (Horizon End)</span>
                </div>
              </div>
            </div>
          </div>

          {/* Future Projection Table */}
          <div className="card">
            <div className="card-header-bar">
              <h4 className="card-title">Forecast Table & 95% Confidence Bounds</h4>
              <span className="badge badge-purple">+{result.horizon} Periods Projected</span>
            </div>
            <div className="table-wrapper">
              <table className="stats-table">
                <thead>
                  <tr>
                    <th>Future Period</th>
                    <th className="text-right">Projected Value</th>
                    <th className="text-right">Lower Bound (95%)</th>
                    <th className="text-right">Upper Bound (95%)</th>
                    <th className="text-right">Change from Current</th>
                  </tr>
                </thead>
                <tbody>
                  {result.future_forecast.map((step) => {
                    const diff = step.forecast - result.last_observed_value
                    const isPositive = diff >= 0
                    return (
                      <tr key={step.step}>
                        <td><strong>Step +{step.step}</strong></td>
                        <td className="text-right num-cell">
                          <strong style={{ color: '#c084fc' }}>{formatNumber(step.forecast)}</strong>
                        </td>
                        <td className="text-right num-cell text-muted">{formatNumber(step.lower_bound)}</td>
                        <td className="text-right num-cell text-muted">{formatNumber(step.upper_bound)}</td>
                        <td className="text-right num-cell">
                          <span className={`badge ${isPositive ? 'badge-success' : 'badge-danger'}`}>
                            {isPositive ? `+${formatNumber(diff)}` : formatNumber(diff)}
                          </span>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
