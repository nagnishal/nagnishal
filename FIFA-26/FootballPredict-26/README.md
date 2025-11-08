# ⚽ FIFA World Cup 2026 Finalist Prediction

An AI-powered web application that predicts FIFA World Cup 2026 finalists using machine learning and real-time football data from the API-Football API.

## 🚀 Features

- **Real-time Data Integration**: Fetches live team statistics from API-Football
- **AI Predictions**: Uses Random Forest ML model to predict tournament finalists
- **Interactive Dashboard**: View team statistics, rankings, and performance metrics
- **Advanced Analytics**: Feature importance, correlation analysis, and team comparisons
- **Smart Fallback**: Works even when API is unavailable with calculated data

## 📋 Prerequisites

- Python 3.11 or higher
- API-Football API key (get one at [api-football.com](https://www.api-football.com/))

## 🔧 Installation & Setup

### Step 1: Download the Project

Download all project files to a folder on your computer:
- `app.py`
- `api_utils.py`
- `local_requirements.txt`
- `README.md`

### Step 2: Install Python (if not already installed)

Download and install Python from [python.org](https://www.python.org/downloads/)

Make sure to check "Add Python to PATH" during installation.

### Step 3: Create a Virtual Environment (Recommended)

Open your terminal/command prompt and navigate to the project folder:

```bash
# On Windows
cd path\to\fifa-world-cup-prediction
python -m venv venv
venv\Scripts\activate

# On Mac/Linux
cd path/to/fifa-world-cup-prediction
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r local_requirements.txt
```

### Step 5: Set Up Your API Key

You have two options:

**Option A: Environment Variable (Recommended)**

```bash
# On Windows (Command Prompt)
set API_FOOTBALL_KEY=bd430de3d62c4946b3b26fa3f2528c51

# On Windows (PowerShell)
$env:API_FOOTBALL_KEY="bd430de3d62c4946b3b26fa3f2528c51"

# On Mac/Linux
export API_FOOTBALL_KEY=bd430de3d62c4946b3b26fa3f2528c51
```

**Option B: Create a .env file (if you install python-dotenv)**

Create a file named `.env` in the project folder:

```
API_FOOTBALL_KEY=bd430de3d62c4946b3b26fa3f2528c51
```

Then install python-dotenv:
```bash
pip install python-dotenv
```

And add this to the top of `app.py`:
```python
from dotenv import load_dotenv
load_dotenv()
```

### Step 6: Run the Application

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

## 🎯 Usage

Once the app is running:

1. **📊 Dashboard Tab**: View current qualification status and team statistics
2. **🎯 Predictions Tab**: See AI-powered finalist predictions with probabilities
3. **📈 Analytics Tab**: Explore feature importance, team comparisons, and correlations
4. **ℹ️ About Tab**: Learn about the methodology and data sources

## 📊 How It Works

### Data Collection
- Fetches real-time team statistics from API-Football API
- Combines with historical World Cup performance data
- Uses smart caching (1-hour) to optimize API usage
- Falls back to calculated data if API is unavailable

### Machine Learning Model
- **Algorithm**: Random Forest Classifier
- **Features**: FIFA ranking, goals scored/conceded, win rate, average age, World Cup titles
- **Training**: Uses historical data to identify patterns of successful teams
- **Output**: Probability scores for each team reaching the finals

### Predictions
The model considers:
- Current FIFA rankings
- Recent performance (goals, wins)
- Historical World Cup success
- Team composition (average age)
- Statistical patterns from past tournaments

## 🔑 API Information

This application uses the **API-Football** service:
- Base URL: `https://v3.football.api-sports.io`
- Free tier available with limited requests
- Get your API key at: [api-football.com](https://www.api-football.com/)

The provided API key in the code is for demonstration. For production use, please obtain your own key.

## 📦 Project Structure

```
fifa-world-cup-prediction/
│
├── app.py                      # Main Streamlit application
├── api_utils.py                # API integration utilities
├── local_requirements.txt      # Python dependencies
└── README.md                   # This file
```

## 🛠️ Troubleshooting

### Port Already in Use

If you see "Port 8501 is already in use", either:
- Close the existing Streamlit app
- Run on a different port: `streamlit run app.py --server.port 8502`

### API Key Error

If you see "API key not found":
- Make sure you've set the `API_FOOTBALL_KEY` environment variable
- Check that there are no typos in the key
- Restart your terminal after setting the environment variable

### Module Not Found

If you see "ModuleNotFoundError":
- Make sure your virtual environment is activated
- Run `pip install -r local_requirements.txt` again

### Slow Loading

The first time you open the app:
- It may take 20-30 seconds to fetch data from the API
- Data is cached for 1 hour to improve subsequent loads
- If API is slow, the app will use fallback data automatically

## 🌟 Features in Detail

### Dashboard
- Current qualification status (28 of 48 teams qualified)
- Complete team statistics table
- Top 10 teams by win rate visualization

### Predictions
- Top 10 predicted finalists with probability scores
- Predicted final match with team comparison
- Model accuracy metrics

### Analytics
- Feature importance analysis
- Goals scored vs conceded scatter plot
- Team comparison radar chart (select up to 4 teams)
- Correlation heatmap of all features

## 📝 Notes

- The application uses data from the 28 currently qualified teams for FIFA World Cup 2026
- Predictions are based on current statistics and historical patterns
- Actual tournament results may vary based on team form, injuries, and match dynamics
- The app works offline with calculated data if the API is unavailable

## 🙏 Credits

- Data source: API-Football
- Machine Learning: scikit-learn
- Web Framework: Streamlit
- Visualizations: Plotly

## 📄 License

This project is for educational and demonstration purposes.

---

**Enjoy exploring FIFA World Cup 2026 predictions! ⚽🏆**
