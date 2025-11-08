# Overview

This is a FIFA World Cup 2026 finalist prediction application that uses machine learning (RandomForestClassifier) to analyze and predict tournament outcomes. The application is built with Streamlit for the web interface and integrates with the API-Football service to fetch real-time football statistics including team rankings, statistics, and national team data. The system combines historical data with AI-powered predictions to forecast which teams are likely to reach the World Cup finals.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture

**Technology**: Streamlit web framework

The application uses Streamlit as its primary UI framework, providing an interactive dashboard for users to view predictions and team analytics. The frontend is configured with a wide layout and custom page settings (soccer ball emoji, custom title). The interface displays:

- AI-powered tournament analysis
- Team statistics and visualizations using Plotly (both Express and Graph Objects)
- Real-time data from football APIs
- Interactive charts and predictions

**Rationale**: Streamlit was chosen for rapid development of data science applications with minimal frontend code, allowing focus on the ML models and data analysis rather than complex UI implementation.

## Backend Architecture

**ML Pipeline**: scikit-learn RandomForestClassifier

The prediction engine uses a Random Forest classifier trained on team statistics including:
- FIFA rankings
- Goals scored/conceded
- Average team age
- Win rates
- Qualification status

The system employs train-test split methodology for model validation and provides accuracy scores and classification reports for transparency.

**Design Pattern**: The application follows a data pipeline pattern:
1. Data collection (API integration)
2. Feature engineering (team statistics processing)
3. Model training and evaluation
4. Prediction generation
5. Visualization output

**Rationale**: Random Forest was selected for its ability to handle multiple features, resistance to overfitting, and interpretability - crucial for sports predictions where users want to understand the reasoning behind predictions.

## Data Collection Strategy

**API-First with Intelligent Fallback**:
1. **Real-time API integration**: The application attempts to fetch live team statistics from API-Football for all 28 qualified teams
   - Fetches team information using the `/teams` endpoint
   - Retrieves detailed statistics via `/teams/statistics` endpoint
   - Extracts goals scored/conceded, fixtures played, wins, and calculates win rates
2. **Graceful degradation**: If API calls fail or timeout, the system falls back to enhanced calculated data based on FIFA rankings
3. **Hybrid data model**: Combines API-sourced statistics with historical World Cup titles for comprehensive analysis

**Caching Strategy**: 
- Implements Streamlit's `@st.cache_data` decorator with 1-hour TTL (3600 seconds)
- Reduces API calls and improves performance
- Balances data freshness with API rate limits

**Error Handling**: 
- Per-team try-catch blocks ensure partial failures don't crash the application
- User-friendly warnings display when API connectivity issues occur
- Automatic fallback ensures the app remains functional even with API unavailability

**Rationale**: This robust approach prioritizes real data while ensuring the application always provides meaningful predictions, even when external services are unavailable.

## Data Storage

**In-Memory Processing**: The application currently uses pandas DataFrames for in-memory data manipulation without persistent storage.

**Consideration**: The system is designed as a stateless web application where data is fetched fresh from APIs on each session or cached temporarily.

**Future Expansion**: The architecture could accommodate a database layer (likely PostgreSQL) if persistent storage of predictions, historical data, or user preferences becomes necessary.

# External Dependencies

## Third-Party APIs

**API-Football (v3.football.api-sports.io)**
- Primary data source for real football statistics
- Endpoints actively used:
  - `/teams` - Fetch national team information by country name
  - `/teams/statistics` - Detailed team performance metrics (goals, fixtures, wins)
- Authentication: API key via environment variable `API_FOOTBALL_KEY`
- Rate limiting: Implemented via caching strategy with 1-hour TTL

**Configuration**:
- Base URL: `https://v3.football.api-sports.io`
- Headers: `x-rapidapi-host` and `x-rapidapi-key`
- Timeout: 5 seconds per request (reduced for better UX)
- Error handling: Per-request try-catch blocks with graceful degradation
- Fallback strategy: Enhanced calculated data ensures continuous operation

**Implementation Details**:
- Sequential API calls for each of the 28 qualified teams
- Extracts: goals scored/conceded, matches played, wins, and calculates win rates
- Combines API data with static historical World Cup titles
- FIFA rankings maintained locally for consistency and performance

## Python Libraries

**Data Science Stack**:
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computations
- `scikit-learn` - Machine learning models and evaluation metrics

**Visualization**:
- `plotly.express` - High-level plotting interface
- `plotly.graph_objects` - Detailed interactive visualizations

**Web Framework**:
- `streamlit` - Interactive web application framework

**HTTP Client**:
- `requests` - API communication with timeout and error handling

## Environment Configuration

**Required Environment Variables**:
- `API_FOOTBALL_KEY` - Authentication for API-Football service (mandatory)

**Error Handling**: Application displays error and stops execution if API key is not configured, preventing runtime failures.