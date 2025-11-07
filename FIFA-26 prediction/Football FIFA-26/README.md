# Overview

This is a FIFA World Cup 2026 Finalist Prediction application that uses machine learning to predict tournament outcomes based on real football data. The application is built with Streamlit for the web interface and integrates with the Football API (API-Sports) to fetch live team statistics. It employs a Random Forest Classifier to analyze team performance metrics and predict which teams are likely to reach the finals.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture

**Framework**: Streamlit
- **Rationale**: Streamlit provides a rapid development framework for data science applications with minimal boilerplate code, making it ideal for ML-based prediction dashboards
- **UI Components**: Uses Streamlit's native widgets for interactive controls and data display
- **Visualization**: Plotly (Express and Graph Objects) for interactive charts and data visualization
- **Layout**: Wide layout configuration for better data presentation

## Backend Architecture

**Machine Learning Pipeline**:
- **Model**: Random Forest Classifier from scikit-learn (200 estimators, max_depth=10)
- **Training Approach**: Supervised learning using performance-based labeling system
- **Label Assignment**: Combines historical World Cup winners recognition with performance score quantiles (top 33%, middle 33%, bottom 33%)
- **Features**: 8 key metrics - FIFA ranking, goals scored/conceded, average team age, win rate, attack rating, defense rating, overall rating
- **Data Processing**: Pandas and NumPy for data manipulation and feature engineering
- **Train/Test Split**: Stratified train_test_split (80/20) to maintain class balance
- **Fallback Logic**: Automatic binning by performance score if single-class scenario occurs

**Data Flow**:
1. Fetch real-time team data from Football API
2. Process and normalize team statistics
3. Train/update ML model with current data
4. Generate predictions for World Cup 2026 finalists
5. Visualize results through interactive dashboard

## Data Management

**Caching Strategy**: Streamlit's `@st.cache_data` decorator with 1-hour TTL (3600 seconds)
- **Purpose**: Reduce API calls and improve application performance
- **Trade-off**: Balances data freshness with API rate limits

**Data Sources**:
- Primary: Live API data from major European leagues (Premier League, La Liga, Bundesliga, Serie A, Ligue 1)
- Fallback: Simulated data for teams not yet qualified (addressing the 28/48 qualification constraint)
- Team limit: 48 teams to match World Cup 2026 format

**Data Structure**:
- Team performance metrics stored in Pandas DataFrames
- Key features: FIFA_Ranking, Goals_Scored, Goals_Conceded, Avg_Age, Win_Rate, Qualified status

## Simulation Logic

**Hybrid Approach**: Combines real qualified teams (28) with simulated teams (20) until all slots are filled
- **Problem**: Only 28 of 48 World Cup 2026 teams have qualified at development time
- **Solution**: Simulate remaining teams from top 100 FIFA rankings with randomized but realistic statistics
- **Alternatives Considered**: 
  - Wait for all qualifications (delays project)
  - Use only qualified teams (insufficient data)
- **Chosen Approach**: Dynamic simulation allows immediate development and testing while maintaining realistic prediction scope

# External Dependencies

## Third-Party APIs

**Football API (API-Sports)**:
- **Endpoint**: `https://v3.football.api-sports.io/standings`
- **Authentication**: API key-based (x-apisports-key header)
- **Configuration**: API key stored in environment variable `FOOTBALL_API_KEY`
- **Data Retrieved**: Team standings from major leagues (2024 season)
- **Rate Limiting**: Managed through caching strategy (1-hour TTL)

**Leagues Monitored**:
- Premier League (League ID: 39)
- La Liga (League ID: 140)
- Bundesliga (League ID: 78)
- Serie A (League ID: 135)
- Ligue 1 (League ID: 61)

## Python Libraries

**Core Dependencies**:
- `streamlit`: Web application framework
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `scikit-learn`: Machine learning models and utilities
- `plotly`: Interactive visualization (express and graph_objects)
- `requests`: HTTP client for API calls

**System Libraries**:
- `os`: Environment variable management
- `datetime`: Timestamp handling

## Environment Configuration

**Required Environment Variables**:
- `FOOTBALL_API_KEY`: Authentication token for Football API access