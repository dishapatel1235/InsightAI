export interface ParsedCsv {
  headers: string[]
  rows: Record<string, string | number | null>[]
  totalRows: number
}

/**
 * Fast, memory-safe client-side CSV text parser
 * Limits line parsing to maxRows to prevent any main-thread freezes on large datasets
 */
export function parseCsvText(text: string, maxRows = 25): ParsedCsv {
  // Split on newlines, limiting parsing
  const rawLines = text.split(/\r?\n/)
  // Filter non-empty lines, keeping up to maxRows + 2 lines
  const lines: string[] = []
  for (let i = 0; i < rawLines.length && lines.length < maxRows + 2; i++) {
    const trimmed = rawLines[i].trim()
    if (trimmed.length > 0) {
      lines.push(trimmed)
    }
  }

  if (lines.length === 0) {
    return { headers: [], rows: [], totalRows: 0 }
  }

  const parseLine = (line: string): string[] => {
    const result: string[] = []
    let current = ''
    let inQuotes = false

    for (let i = 0; i < line.length; i++) {
      const char = line[i]
      if (char === '"') {
        inQuotes = !inQuotes
      } else if (char === ',' && !inQuotes) {
        result.push(current.trim().replace(/^"|"$/g, ''))
        current = ''
      } else {
        current += char
      }
    }
    result.push(current.trim().replace(/^"|"$/g, ''))
    return result
  }

  const headers = parseLine(lines[0])
  const dataLines = lines.slice(1, maxRows + 1)
  const rows: Record<string, string | number | null>[] = []

  for (let i = 0; i < dataLines.length; i++) {
    const values = parseLine(dataLines[i])
    const rowObj: Record<string, string | number | null> = {}
    headers.forEach((header, idx) => {
      const val = values[idx] ?? ''
      if (val !== '' && !isNaN(Number(val))) {
        rowObj[header] = Number(val)
      } else {
        rowObj[header] = val === '' ? null : val
      }
    })
    rows.push(rowObj)
  }

  return {
    headers,
    rows,
    totalRows: dataLines.length,
  }
}

/**
 * Predefined sample datasets for 1-click testing
 */
export const SAMPLE_DATASETS: Record<
  string,
  { filename: string; description: string; recommendedMode: string; content: string }
> = {
  churn: {
    filename: 'telecom_customer_churn.csv',
    description: 'Predictive Classification: Predict Customer Churn (Random Forest, XGBoost, NN)',
    recommendedMode: 'predictive',
    content: `customer_id,tenure_months,monthly_charges,total_charges,contract_type,support_tickets,churn
C101,1,29.85,29.85,Month-to-month,3,Yes
C102,34,56.95,1889.50,One year,0,No
C103,2,53.85,108.15,Month-to-month,2,Yes
C104,45,42.30,1840.75,One year,0,No
C105,2,70.70,151.65,Month-to-month,4,Yes
C106,8,99.65,820.50,Month-to-month,3,Yes
C107,22,89.10,1949.40,Month-to-month,1,No
C108,10,29.75,301.90,Month-to-month,0,No
C109,28,104.80,3046.05,Month-to-month,2,Yes
C110,62,56.15,3487.95,Two year,0,No
C111,13,55.15,742.90,Month-to-month,1,No
C112,16,18.95,326.80,Two year,0,No
C113,58,100.35,5681.10,One year,0,No
C114,49,103.70,5036.30,Month-to-month,1,Yes
C115,25,20.65,512.45,Month-to-month,0,No
C116,69,113.25,7895.15,Two year,0,No
C117,52,20.65,1022.95,Two year,0,No
C118,71,106.70,7382.25,Two year,0,No
C119,10,55.20,528.35,Month-to-month,2,Yes
C120,21,90.05,1862.90,Month-to-month,1,No
C121,12,19.80,202.25,One year,0,No
C122,61,25.00,1498.35,Two year,0,No
C123,3,74.40,229.55,Month-to-month,3,Yes
C124,72,115.55,8127.60,Two year,0,No
C125,17,70.15,1148.80,Month-to-month,1,No`,
  },
  housing: {
    filename: 'housing_prices_regression.csv',
    description: 'Predictive Regression: Predict Real Estate Market Value (RF, XGBoost, Neural Net)',
    recommendedMode: 'predictive',
    content: `property_id,area_sqft,bedrooms,bathrooms,age_years,distance_city_km,has_garden,price_k
H01,1450,3,2,12,6.5,1,385.5
H02,2100,4,3,5,11.2,1,540.0
H03,980,2,1,24,3.1,0,245.0
H04,3200,5,4,2,14.8,1,890.0
H05,1850,3,2.5,8,8.4,1,465.0
H06,1200,2,2,18,4.7,0,310.0
H07,2600,4,3.5,6,12.0,1,670.0
H08,1650,3,2,15,7.9,0,420.0
H09,2900,4,4,3,13.5,1,780.0
H10,850,1,1,30,2.0,0,215.0
H11,1750,3,2,10,9.0,1,450.0
H12,2300,4,3,4,10.5,1,595.0
H13,1350,2,2,14,5.2,0,340.0
H14,3500,5,4,1,16.0,1,960.0
H15,1950,3,2.5,7,7.8,1,490.0
H16,1100,2,1,22,3.8,0,280.0
H17,2750,4,3.5,5,11.5,1,710.0
H18,1550,3,2,16,8.2,0,395.0
H19,3100,5,4,2,14.0,1,840.0
H20,920,2,1,28,2.5,0,230.0`,
  },
  sales_ts: {
    filename: 'daily_retail_forecast_timeseries.csv',
    description: 'Time-Series Forecasting: 40 Days of Demand & Revenue (LSTM & Neural Networks)',
    recommendedMode: 'timeseries',
    content: `date,daily_sales,store_traffic,temperature,marketing_spend
2026-01-01,1240,480,18.5,120
2026-01-02,1310,510,19.0,140
2026-01-03,1420,560,20.2,160
2026-01-04,1580,620,21.0,200
2026-01-05,1750,710,22.4,250
2026-01-06,1920,790,21.8,280
2026-01-07,1840,750,20.5,230
2026-01-08,1350,520,19.1,130
2026-01-09,1390,540,18.8,145
2026-01-10,1480,580,19.5,160
2026-01-11,1620,640,21.3,210
2026-01-12,1810,730,22.0,260
2026-01-13,1980,810,23.1,300
2026-01-14,1910,770,22.4,240
2026-01-15,1420,530,20.1,140
2026-01-16,1460,560,19.6,150
2026-01-17,1550,600,20.4,180
2026-01-18,1710,670,21.9,220
2026-01-19,1890,750,22.8,270
2026-01-20,2050,830,23.5,310
2026-01-21,1960,790,22.0,260
2026-01-22,1490,550,20.2,150
2026-01-23,1530,570,19.8,160
2026-01-24,1630,620,20.6,190
2026-01-25,1790,690,21.7,230
2026-01-26,1970,780,22.9,280
2026-01-27,2140,860,24.0,330
2026-01-28,2020,810,23.2,270
2026-01-29,1560,580,20.8,160
2026-01-30,1610,600,20.2,170
2026-01-31,1720,650,21.1,200
2026-02-01,1880,720,22.2,240
2026-02-02,2060,810,23.4,290
2026-02-03,2230,890,24.5,340
2026-02-04,2110,840,23.6,280
2026-02-05,1640,610,21.2,170
2026-02-06,1690,630,20.7,180
2026-02-07,1800,680,21.5,210
2026-02-08,1960,750,22.6,250
2026-02-09,2140,840,23.8,300`,
  },
  customers: {
    filename: 'mall_customers_segmentation.csv',
    description: 'Pattern Discovery: Customer Segmentation (K-Means Clustering, PCA & Anomaly Detection)',
    recommendedMode: 'patterns',
    content: `customer_id,gender,age,annual_income_k,spending_score_1_100
1,Male,19,15,39
2,Male,21,15,81
3,Female,20,16,6
4,Female,23,16,77
5,Female,31,17,40
6,Female,22,17,76
7,Female,35,18,6
8,Female,23,18,94
9,Male,64,19,3
10,Female,30,19,72
11,Male,67,19,14
12,Female,35,19,99
13,Female,58,20,15
14,Female,24,20,77
15,Male,37,20,13
16,Male,22,20,79
17,Female,35,21,35
18,Male,20,21,66
19,Male,52,23,29
20,Female,35,23,98
21,Male,35,24,35
22,Male,25,24,73
23,Female,46,25,5
24,Male,31,25,73
25,Female,54,28,14`,
  },
}

export function createSampleFile(key: keyof typeof SAMPLE_DATASETS): File {
  const sample = SAMPLE_DATASETS[key]
  const blob = new Blob([sample.content], { type: 'text/csv' })
  return new File([blob], sample.filename, { type: 'text/csv' })
}
