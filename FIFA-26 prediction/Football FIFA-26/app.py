import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="FIFA World Cup 2026 Finalist Prediction", page_icon="⚽", layout="wide")

API_KEY = os.getenv("089d508592314ba4929e624d675056c2")

st.title("⚽ FIFA World Cup 2026 Finalist Prediction")
st.markdown("### AI-Powered Prediction Using Real Football Data")

@st.cache_data(ttl=3600)
def fetch_real_team_data():
    if not API_KEY:
        return None
    
    headers = {
        'x-apisports-key': API_KEY
    }
    
    try:
        st.info("🔄 Fetching real-time football data from API...")
        
        teams_data = []
        
        leagues_to_fetch = [
            {'league': 39, 'season': 2024, 'name': 'Premier League'},
            {'league': 140, 'season': 2024, 'name': 'La Liga'},
            {'league': 78, 'season': 2024, 'name': 'Bundesliga'},
            {'league': 135, 'season': 2024, 'name': 'Serie A'},
            {'league': 61, 'season': 2024, 'name': 'Ligue 1'},
        ]
        
        for league_info in leagues_to_fetch:
            if len(teams_data) >= 48:
                break
                
            try:
                response = requests.get(
                    'https://v3.football.api-sports.io/standings',
                    headers=headers,
                    params={'season': league_info['season'], 'league': league_info['league']},
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'response' in data and len(data['response']) > 0:
                        league_standings = data['response'][0].get('league', {}).get('standings', [])
                        
                        for group in league_standings:
                            for team_info in group:
                                if len(teams_data) >= 48:
                                    break
                                    
                                team = team_info.get('team', {})
                                stats = team_info.get('all', {})
                                
                                team_name = team.get('name', 'Unknown')
                                
                                if not any(t['name'] == team_name for t in teams_data):
                                    teams_data.append({
                                        'name': team_name,
                                        'ranking': len(teams_data) + 1,
                                        'goals_scored': stats.get('goals', {}).get('for', 0),
                                        'goals_conceded': stats.get('goals', {}).get('against', 0),
                                        'matches_played': stats.get('played', 10),
                                        'wins': stats.get('win', 0),
                                        'draws': stats.get('draw', 0),
                                        'losses': stats.get('lose', 0)
                                    })
            except Exception as league_error:
                continue
        
        if len(teams_data) >= 20:
            st.success(f"✅ Successfully fetched data for {len(teams_data)} teams from API")
            return teams_data
        
        st.warning("⚠️ API returned insufficient data, using simulated data")
        return None
            
    except Exception as e:
        st.warning(f"⚠️ API Error: {str(e)}. Using simulated data as fallback.")
        return None

def map_team_to_confederation(team_name):
    confederation_mapping = {
        'Argentina': 'CONMEBOL', 'Brazil': 'CONMEBOL', 'Uruguay': 'CONMEBOL', 
        'Colombia': 'CONMEBOL', 'Ecuador': 'CONMEBOL', 'Chile': 'CONMEBOL', 
        'Peru': 'CONMEBOL', 'Paraguay': 'CONMEBOL',
        
        'France': 'UEFA', 'England': 'UEFA', 'Belgium': 'UEFA', 'Netherlands': 'UEFA',
        'Portugal': 'UEFA', 'Spain': 'UEFA', 'Italy': 'UEFA', 'Croatia': 'UEFA',
        'Germany': 'UEFA', 'Switzerland': 'UEFA', 'Denmark': 'UEFA', 'Poland': 'UEFA',
        'Sweden': 'UEFA', 'Wales': 'UEFA', 'Ukraine': 'UEFA', 'Austria': 'UEFA',
        'Czech Republic': 'UEFA', 'Serbia': 'UEFA', 'Turkey': 'UEFA', 'Scotland': 'UEFA',
        'Norway': 'UEFA',
        
        'USA': 'CONCACAF', 'Mexico': 'CONCACAF', 'Canada': 'CONCACAF', 
        'Costa Rica': 'CONCACAF',
        
        'Japan': 'AFC', 'South Korea': 'AFC', 'Australia': 'AFC', 'Iran': 'AFC',
        'Qatar': 'AFC', 'Saudi Arabia': 'AFC', 'Iraq': 'AFC',
        
        'Morocco': 'CAF', 'Senegal': 'CAF', 'Nigeria': 'CAF', 'Tunisia': 'CAF',
        'Algeria': 'CAF', 'Egypt': 'CAF', 'Cameroon': 'CAF', 'Mali': 'CAF'
    }
    
    for key in confederation_mapping:
        if key.lower() in team_name.lower():
            return confederation_mapping[key]
    return 'UEFA'

def create_team_dataset():
    api_data = fetch_real_team_data()
    
    if api_data and len(api_data) > 0:
        data = []
        for idx, team_info in enumerate(api_data[:48]):
            matches_played = max(1, team_info.get('matches_played', 10))
            wins = team_info.get('wins', 0)
            goals_scored = team_info.get('goals_scored', 0)
            goals_conceded = team_info.get('goals_conceded', 0)
            
            win_rate = round(wins / matches_played, 2) if matches_played > 0 else 0.5
            goals_per_match = round(goals_scored / matches_played, 2) if matches_played > 0 else 1.5
            defense_rating = max(0, round(100 - (goals_conceded / matches_played) * 10, 1)) if matches_played > 0 else 50
            attack_rating = round(min(100, goals_per_match * 30), 1)
            
            team_name = team_info.get('name', f'Team {idx+1}')
            confederation = map_team_to_confederation(team_name)
            
            confederation_strength = {
                'UEFA': 1.2,
                'CONMEBOL': 1.15,
                'CONCACAF': 1.0,
                'AFC': 0.95,
                'CAF': 0.9
            }
            
            base_score = max(50, 100 - idx * 1.5)
            overall_rating = round(
                base_score * confederation_strength.get(confederation, 1.0),
                1
            )
            
            data.append({
                'Team': team_name,
                'FIFA_Ranking': team_info.get('ranking', idx + 1),
                'Confederation': confederation,
                'Goals_Scored': goals_scored,
                'Goals_Conceded': goals_conceded,
                'Avg_Age': round(np.random.uniform(25.5, 28.5), 1),
                'Win_Rate': win_rate,
                'Matches_Played': matches_played,
                'Attack_Rating': attack_rating,
                'Defense_Rating': defense_rating,
                'Overall_Rating': overall_rating
            })
        
        return pd.DataFrame(data)
    
    st.info("📊 Using simulated World Cup qualifying data (48 teams)")
    
    qualified_teams = [
        {'name': 'Argentina', 'ranking': 1, 'confederation': 'CONMEBOL'},
        {'name': 'France', 'ranking': 2, 'confederation': 'UEFA'},
        {'name': 'Brazil', 'ranking': 3, 'confederation': 'CONMEBOL'},
        {'name': 'England', 'ranking': 4, 'confederation': 'UEFA'},
        {'name': 'Belgium', 'ranking': 5, 'confederation': 'UEFA'},
        {'name': 'Netherlands', 'ranking': 6, 'confederation': 'UEFA'},
        {'name': 'Portugal', 'ranking': 7, 'confederation': 'UEFA'},
        {'name': 'Spain', 'ranking': 8, 'confederation': 'UEFA'},
        {'name': 'Italy', 'ranking': 9, 'confederation': 'UEFA'},
        {'name': 'Croatia', 'ranking': 10, 'confederation': 'UEFA'},
        {'name': 'USA', 'ranking': 11, 'confederation': 'CONCACAF'},
        {'name': 'Mexico', 'ranking': 12, 'confederation': 'CONCACAF'},
        {'name': 'Uruguay', 'ranking': 13, 'confederation': 'CONMEBOL'},
        {'name': 'Germany', 'ranking': 14, 'confederation': 'UEFA'},
        {'name': 'Morocco', 'ranking': 15, 'confederation': 'CAF'},
        {'name': 'Colombia', 'ranking': 16, 'confederation': 'CONMEBOL'},
        {'name': 'Japan', 'ranking': 17, 'confederation': 'AFC'},
        {'name': 'Senegal', 'ranking': 18, 'confederation': 'CAF'},
        {'name': 'Switzerland', 'ranking': 19, 'confederation': 'UEFA'},
        {'name': 'Denmark', 'ranking': 20, 'confederation': 'UEFA'},
        {'name': 'South Korea', 'ranking': 21, 'confederation': 'AFC'},
        {'name': 'Australia', 'ranking': 22, 'confederation': 'AFC'},
        {'name': 'Iran', 'ranking': 23, 'confederation': 'AFC'},
        {'name': 'Canada', 'ranking': 24, 'confederation': 'CONCACAF'},
        {'name': 'Ecuador', 'ranking': 25, 'confederation': 'CONMEBOL'},
        {'name': 'Nigeria', 'ranking': 26, 'confederation': 'CAF'},
        {'name': 'Poland', 'ranking': 27, 'confederation': 'UEFA'},
        {'name': 'Sweden', 'ranking': 28, 'confederation': 'UEFA'},
        {'name': 'Wales', 'ranking': 29, 'confederation': 'UEFA'},
        {'name': 'Ukraine', 'ranking': 30, 'confederation': 'UEFA'},
        {'name': 'Tunisia', 'ranking': 31, 'confederation': 'CAF'},
        {'name': 'Austria', 'ranking': 32, 'confederation': 'UEFA'},
        {'name': 'Czech Republic', 'ranking': 33, 'confederation': 'UEFA'},
        {'name': 'Serbia', 'ranking': 34, 'confederation': 'UEFA'},
        {'name': 'Chile', 'ranking': 35, 'confederation': 'CONMEBOL'},
        {'name': 'Algeria', 'ranking': 36, 'confederation': 'CAF'},
        {'name': 'Peru', 'ranking': 37, 'confederation': 'CONMEBOL'},
        {'name': 'Egypt', 'ranking': 38, 'confederation': 'CAF'},
        {'name': 'Costa Rica', 'ranking': 39, 'confederation': 'CONCACAF'},
        {'name': 'Cameroon', 'ranking': 40, 'confederation': 'CAF'},
        {'name': 'Mali', 'ranking': 41, 'confederation': 'CAF'},
        {'name': 'Qatar', 'ranking': 42, 'confederation': 'AFC'},
        {'name': 'Turkey', 'ranking': 43, 'confederation': 'UEFA'},
        {'name': 'Scotland', 'ranking': 44, 'confederation': 'UEFA'},
        {'name': 'Norway', 'ranking': 45, 'confederation': 'UEFA'},
        {'name': 'Paraguay', 'ranking': 46, 'confederation': 'CONMEBOL'},
        {'name': 'Saudi Arabia', 'ranking': 47, 'confederation': 'AFC'},
        {'name': 'Iraq', 'ranking': 48, 'confederation': 'AFC'}
    ]
    
    np.random.seed(42)
    
    data = []
    for team in qualified_teams:
        base_score = 100 - (team['ranking'] - 1) * 1.5
        variation = np.random.uniform(-5, 5)
        
        goals_scored = max(15, int(base_score / 2.5 + np.random.uniform(-8, 15)))
        goals_conceded = max(5, int(50 - base_score / 2 + np.random.uniform(-5, 10)))
        avg_age = round(np.random.uniform(25.5, 28.5), 1)
        matches_played = np.random.randint(8, 16)
        wins = max(1, int((base_score / 100) * matches_played + np.random.uniform(-2, 3)))
        win_rate = round(wins / matches_played, 2)
        
        goals_per_match = round(goals_scored / matches_played, 2)
        defense_rating = max(0, round(100 - (goals_conceded / matches_played) * 10, 1))
        attack_rating = round(min(100, goals_per_match * 30), 1)
        
        confederation_strength = {
            'UEFA': 1.2,
            'CONMEBOL': 1.15,
            'CONCACAF': 1.0,
            'AFC': 0.95,
            'CAF': 0.9
        }
        
        overall_rating = round(
            (base_score + variation) * confederation_strength.get(team['confederation'], 1.0),
            1
        )
        
        data.append({
            'Team': team['name'],
            'FIFA_Ranking': team['ranking'],
            'Confederation': team['confederation'],
            'Goals_Scored': goals_scored,
            'Goals_Conceded': goals_conceded,
            'Avg_Age': avg_age,
            'Win_Rate': win_rate,
            'Matches_Played': matches_played,
            'Attack_Rating': attack_rating,
            'Defense_Rating': defense_rating,
            'Overall_Rating': overall_rating
        })
    
    return pd.DataFrame(data)

def train_prediction_model(df):
    historical_winners = ['Argentina', 'France', 'Brazil', 'Germany', 'Italy', 
                         'Spain', 'England', 'Uruguay']
    strong_contenders = ['Netherlands', 'Portugal', 'Belgium', 'Croatia']
    
    df_temp = df.copy()
    df_temp['Performance_Score'] = (
        (100 - df_temp['FIFA_Ranking']) / 100 * 30 +
        df_temp['Win_Rate'] * 25 +
        df_temp['Attack_Rating'] / 100 * 25 +
        df_temp['Defense_Rating'] / 100 * 20
    )
    
    top_third = df_temp['Performance_Score'].quantile(0.67)
    middle_third = df_temp['Performance_Score'].quantile(0.33)
    
    def assign_label(row):
        team_name = row['Team']
        perf_score = row['Performance_Score']
        
        if team_name in historical_winners[:5]:
            return 2
        elif team_name in historical_winners or team_name in strong_contenders:
            return 1
        elif perf_score >= top_third:
            return 2
        elif perf_score >= middle_third:
            return 1
        else:
            return 0
    
    y = df_temp.apply(assign_label, axis=1)
    
    if len(np.unique(y)) < 2:
        y = pd.cut(df_temp['Performance_Score'], bins=3, labels=[0, 1, 2]).astype(int)
    
    feature_cols = ['FIFA_Ranking', 'Goals_Scored', 'Goals_Conceded', 
                    'Avg_Age', 'Win_Rate', 'Attack_Rating', 'Defense_Rating', 
                    'Overall_Rating']
    X = df[feature_cols]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, feature_cols

with st.spinner("🔄 Loading team data..."):
    df = create_team_dataset()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Teams", len(df))
with col2:
    st.metric("Confederations", df['Confederation'].nunique())
with col3:
    avg_goals = df['Goals_Scored'].mean()
    st.metric("Avg Goals/Team", f"{avg_goals:.1f}")
with col4:
    avg_ranking = df['FIFA_Ranking'].mean()
    st.metric("Avg FIFA Ranking", f"{avg_ranking:.1f}")

st.markdown("---")

with st.spinner("🤖 Training AI prediction model..."):
    model, accuracy, feature_cols = train_prediction_model(df)

st.success(f"✅ Model trained with {accuracy*100:.1f}% accuracy")

df['Finalist_Probability'] = model.predict_proba(df[feature_cols])[:, 2] if len(np.unique(model.predict(df[feature_cols]))) > 2 else model.predict_proba(df[feature_cols])[:, 1]

df['Champion_Score'] = (
    df['Finalist_Probability'] * 40 +
    (100 - df['FIFA_Ranking']) / 100 * 25 +
    df['Win_Rate'] * 20 +
    df['Overall_Rating'] / 100 * 15
)

df_sorted = df.sort_values('Champion_Score', ascending=False).reset_index(drop=True)

st.markdown("## 🏆 Predicted Top Finalists")

top_2 = df_sorted.head(2)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🥇 Predicted Champion")
    champion = top_2.iloc[0]
    st.markdown(f"## **{champion['Team']}**")
    st.markdown(f"**FIFA Ranking:** #{champion['FIFA_Ranking']}")
    st.markdown(f"**Confederation:** {champion['Confederation']}")
    st.markdown(f"**Win Rate:** {champion['Win_Rate']*100:.0f}%")
    st.markdown(f"**Champion Score:** {champion['Champion_Score']:.2f}")
    
    st.progress(champion['Champion_Score'] / 100)

with col2:
    st.markdown("### 🥈 Predicted Runner-up")
    runner_up = top_2.iloc[1]
    st.markdown(f"## **{runner_up['Team']}**")
    st.markdown(f"**FIFA Ranking:** #{runner_up['FIFA_Ranking']}")
    st.markdown(f"**Confederation:** {runner_up['Confederation']}")
    st.markdown(f"**Win Rate:** {runner_up['Win_Rate']*100:.0f}%")
    st.markdown(f"**Champion Score:** {runner_up['Champion_Score']:.2f}")
    
    st.progress(runner_up['Champion_Score'] / 100)

st.markdown("---")
st.markdown("## 📊 Top 10 Championship Contenders")

top_10 = df_sorted.head(10)

fig = px.bar(
    top_10,
    x='Champion_Score',
    y='Team',
    orientation='h',
    title='Championship Probability Ranking',
    labels={'Champion_Score': 'Champion Score', 'Team': 'National Team'},
    color='Champion_Score',
    color_continuous_scale='RdYlGn',
    text='Champion_Score'
)
fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
fig.update_layout(height=500, showlegend=False)
st.plotly_chart(fig, width='stretch')

st.markdown("---")
st.markdown("## 📈 Detailed Team Statistics")

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Attack vs Defense", "🌍 By Confederation", "📋 Full Rankings", "🔍 Team Comparison"])

with tab1:
    fig2 = px.scatter(
        df_sorted.head(20),
        x='Attack_Rating',
        y='Defense_Rating',
        size='Champion_Score',
        color='Champion_Score',
        hover_name='Team',
        hover_data={
            'FIFA_Ranking': True,
            'Goals_Scored': True,
            'Goals_Conceded': True,
            'Champion_Score': ':.2f'
        },
        title='Attack vs Defense Rating (Top 20 Teams)',
        labels={
            'Attack_Rating': 'Attack Rating',
            'Defense_Rating': 'Defense Rating'
        },
        color_continuous_scale='Viridis'
    )
    fig2.update_layout(height=500)
    st.plotly_chart(fig2, width='stretch')

with tab2:
    conf_stats = df_sorted.groupby('Confederation').agg({
        'Team': 'count',
        'Champion_Score': 'mean',
        'FIFA_Ranking': 'mean',
        'Win_Rate': 'mean'
    }).reset_index()
    conf_stats.columns = ['Confederation', 'Teams', 'Avg Champion Score', 'Avg Ranking', 'Avg Win Rate']
    
    fig3 = px.bar(
        conf_stats,
        x='Confederation',
        y='Avg Champion Score',
        color='Confederation',
        title='Average Championship Score by Confederation',
        text='Avg Champion Score'
    )
    fig3.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    st.plotly_chart(fig3, width='stretch')
    
    st.dataframe(
        conf_stats.style.format({
            'Avg Champion Score': '{:.2f}',
            'Avg Ranking': '{:.1f}',
            'Avg Win Rate': '{:.2%}'
        }),
        width='stretch'
    )

with tab3:
    st.markdown("### Complete Team Rankings")
    
    display_df = df_sorted[['Team', 'FIFA_Ranking', 'Confederation', 'Goals_Scored', 
                             'Goals_Conceded', 'Win_Rate', 'Attack_Rating', 
                             'Defense_Rating', 'Champion_Score']].copy()
    
    display_df['Win_Rate'] = display_df['Win_Rate'].apply(lambda x: f"{x*100:.0f}%")
    
    st.dataframe(
        display_df.style.background_gradient(subset=['Champion_Score'], cmap='RdYlGn'),
        width='stretch',
        height=600
    )

with tab4:
    st.markdown("### Compare Teams Side by Side")
    
    col1, col2 = st.columns(2)
    
    with col1:
        team1 = st.selectbox("Select First Team", df_sorted['Team'].tolist(), index=0)
    with col2:
        team2 = st.selectbox("Select Second Team", df_sorted['Team'].tolist(), index=1)
    
    team1_data = df_sorted[df_sorted['Team'] == team1].iloc[0]
    team2_data = df_sorted[df_sorted['Team'] == team2].iloc[0]
    
    comparison_metrics = {
        'Metric': ['FIFA Ranking', 'Goals Scored', 'Goals Conceded', 'Win Rate (%)', 
                   'Attack Rating', 'Defense Rating', 'Champion Score'],
        team1: [
            team1_data['FIFA_Ranking'],
            team1_data['Goals_Scored'],
            team1_data['Goals_Conceded'],
            round(team1_data['Win_Rate'] * 100, 1),
            team1_data['Attack_Rating'],
            team1_data['Defense_Rating'],
            round(team1_data['Champion_Score'], 2)
        ],
        team2: [
            team2_data['FIFA_Ranking'],
            team2_data['Goals_Scored'],
            team2_data['Goals_Conceded'],
            round(team2_data['Win_Rate'] * 100, 1),
            team2_data['Attack_Rating'],
            team2_data['Defense_Rating'],
            round(team2_data['Champion_Score'], 2)
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_metrics)
    st.dataframe(comparison_df, width='stretch')
    
    categories = ['Attack Rating', 'Defense Rating', 'Win Rate', 'Overall Rating']
    
    fig4 = go.Figure()
    
    fig4.add_trace(go.Scatterpolar(
        r=[team1_data['Attack_Rating'], team1_data['Defense_Rating'], 
           team1_data['Win_Rate']*100, team1_data['Overall_Rating']],
        theta=categories,
        fill='toself',
        name=team1
    ))
    
    fig4.add_trace(go.Scatterpolar(
        r=[team2_data['Attack_Rating'], team2_data['Defense_Rating'], 
           team2_data['Win_Rate']*100, team2_data['Overall_Rating']],
        theta=categories,
        fill='toself',
        name=team2
    ))
    
    fig4.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title=f"{team1} vs {team2} - Performance Comparison"
    )
    
    st.plotly_chart(fig4, width='stretch')

st.markdown("---")
st.markdown("## 🧠 Model Feature Importance")

feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

fig5 = px.bar(
    feature_importance,
    x='Importance',
    y='Feature',
    orientation='h',
    title='Which Factors Matter Most for Predicting the Champion?',
    color='Importance',
    color_continuous_scale='Blues'
)
st.plotly_chart(fig5, width='stretch')

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>⚽ FIFA World Cup 2026 AI Prediction System</strong></p>
    <p>Powered by Machine Learning • Random Forest Algorithm • Real Football Data</p>
    <p><em>Predictions are based on current team statistics and historical performance patterns</em></p>
</div>
""", unsafe_allow_html=True)
