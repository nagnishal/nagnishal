import requests
import os
from datetime import datetime

API_KEY = os.getenv('API_FOOTBALL_KEY', '')
BASE_URL = "https://v3.football.api-sports.io"

HEADERS = {
    'x-rapidapi-host': 'v3.football.api-sports.io',
    'x-rapidapi-key': API_KEY
}

def get_team_rankings():
    """Fetch FIFA rankings for national teams"""
    endpoint = f"{BASE_URL}/standings"
    params = {
        'league': 1,
        'season': 2024
    }
    
    try:
        response = requests.get(endpoint, headers=HEADERS, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        print(f"Error fetching rankings: {e}")
        return None

def get_team_statistics(team_id, season=2024):
    """Fetch team statistics"""
    endpoint = f"{BASE_URL}/teams/statistics"
    params = {
        'team': team_id,
        'season': season,
        'league': 1
    }
    
    try:
        response = requests.get(endpoint, headers=HEADERS, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        print(f"Error fetching team statistics: {e}")
        return None

def get_national_teams():
    """Fetch list of national teams"""
    endpoint = f"{BASE_URL}/teams"
    params = {
        'country': ''
    }
    
    try:
        response = requests.get(endpoint, headers=HEADERS, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        print(f"Error fetching teams: {e}")
        return None

def get_team_matches(team_id, season=2024, last_n=10):
    """Fetch recent matches for a team"""
    endpoint = f"{BASE_URL}/fixtures"
    params = {
        'team': team_id,
        'season': season,
        'last': last_n
    }
    
    try:
        response = requests.get(endpoint, headers=HEADERS, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        print(f"Error fetching matches: {e}")
        return None

def get_country_teams(country_name):
    """Fetch teams from a specific country"""
    endpoint = f"{BASE_URL}/teams"
    params = {
        'country': country_name
    }
    
    try:
        response = requests.get(endpoint, headers=HEADERS, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        print(f"Error fetching country teams: {e}")
        return None
