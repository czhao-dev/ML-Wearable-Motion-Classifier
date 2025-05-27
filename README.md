# Climate Modeling using System Identification and Machine Learning

This project presents a study on modeling local climate patterns in Boston, Massachusetts, using system identification techniques. The project utilizes historical weather data and builds predictive models for precipitation (PRCP), snowfall (SNOW), and observed average temperature (TOBS) using several approaches, including linear models and advanced matrix decomposition methods.

## Overview

The aim of the project is to:
- Evaluate how well system identification methods can model and predict local climate data.
- Compare the performance of various identification methods.
- Understand the underlying seasonal climate trends and day-to-day variability.

## Data

- **Location**: Reading, MA (a Boston suburb)
- **Source**: NOAA (National Oceanic and Atmospheric Administration)
- **Period**: Daily data from 2013 to 2016
  - **Training set**: 2013–2015
  - **Testing set**: 2016
- **Variables**:
  - PRCP: Daily precipitation
  - SNOW: Daily snowfall
  - TOBS: Daily average observed temperature

## Methods

### 1. **ident Software**
- Lag selection: Optimal lag determined to be 4 (uses the past 4 days' data to predict future conditions).
- Models constructed for PRCP, SNOW, and TOBS.
- Best performance in terms of prediction error (misfit).

### 2. **Hankel Matrix + Singular Value Decomposition (SVD)**
- Construction of a Hankel matrix for system identification.
- Matrix decomposition and truncation applied to derive A, B, C, D state-space matrices.
- Moderate performance; better than Veronese but not as good as ident.

### 3. **Veronese Embedding + ident**
- Degree-2 embedding applied for non-linear transformation of the input data.
- Performed poorly relative to the other two methods.

## Results

| Metric | ident | SVD | Veronese + ident |
|--------|-------|-----|------------------|
| PRCP Misfit (mm) | 0.0193 | 0.0257 | 0.1407 |
| SNOW Misfit (mm) | 0.1988 | 0.2538 | 0.6998 |
| TOBS Misfit (°F) | 5.861  | 12.119 | 23.741 |

- **Conclusion**: Linear models built using the ident tool provided the best predictive accuracy. The study also highlights the difficulty in capturing fine-grained, day-to-day weather variations due to inherent randomness and missing atmospheric variables.

## Climate Insight

The modeled data reflect a **humid continental climate**, as per Köppen classification:
- **Summer**: Warm to hot, rainy, and humid.
- **Winter**: Cold, with rain and snow.
- **Spring and Fall**: Mild and transitional.

## Repository Contents

- `README.md`: Summary of the project and methodology.

## License

This project is intended for academic and research purposes.
