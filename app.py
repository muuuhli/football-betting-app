"""
Fußballwetten-Analyse-App (Verbesserte Version mit Debugging)
Automatische Analyse aller Top-Ligen mit Dixon-Coles-Modell
Optimiert für mobile Geräte mit verbesserter API-Integration
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from scipy.stats import poisson
from scipy.optimize import minimize
import time
import json
import traceback
import logging

# Logging-Setup für Debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ============================================================================
# KONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="⚽ Wetten-Analyst Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "Professionelle Fußballwetten-Analyse mit Dixon-Coles-Modell v2.0"
    }
)

# Custom CSS für bessere mobile Darstellung
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    .stButton > button {
        width: 100%;
        padding: 0.75rem;
        font-size: 1rem;
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .dataframe {
        font-size: 0.85rem;
    }
    h1 {
        font-size: 1.8rem !important;
        text-align: center;
    }
    h2 {
        font-size: 1.4rem !important;
    }
    h3 {
        font-size: 1.2rem !important;
    }
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    div[data-testid="stExpander"] div[role="button"] {
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# KONSTANTEN
# ============================================================================

LEAGUES = {
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League": 39,
    "🇩🇪 Bundesliga": 78,
    "🇩🇪 2. Bundesliga": 79,
    "🇪🇸 La Liga": 140,
    "🇮🇹 Serie A": 135,
    "🇫🇷 Ligue 1": 61,
    "🇳🇱 Eredivisie": 88,
    "🇵🇹 Primeira Liga": 94,
    "🇧🇪 Pro League": 144,
    "🇹🇷 Süper Lig": 203
}

# ============================================================================
# SESSION STATE
# ============================================================================

if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 1000.0
    st.session_state.initial_bankroll = 1000.0
    st.session_state.bet_history = []
    st.session_state.api_key = ""
    st.session_state.analysis_cache = {}
    st.session_state.last_analysis = None
    st.session_state.selected_leagues = []

# ============================================================================
# API-FUNKTIONEN (VERBESSERT)
# ============================================================================

def check_api_status(api_key):
    """Überprüft API-Key und zeigt verbleibende Anfragen"""
    if not api_key:
        st.error("❌ Kein API-Key vorhanden")
        return None
    
    # Debug-Info anzeigen
    if st.session_state.get('debug_mode', False):
        st.info(f"🔍 Debug: Prüfe API-Key (erste 10 Zeichen): {api_key[:10]}...")
        
    url = "https://v3.football.api-sports.io/status"
    headers = {'x-apisports-key': api_key}
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        
        # Debug-Info
        if st.session_state.get('debug_mode', False):
            st.info(f"🔍 Debug: API Status Response Code: {response.status_code}")
            if response.text:
                try:
                    st.json(response.json())
                except:
                    st.text(f"Response Text: {response.text[:500]}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('response'):
                account = data['response']['account']
                requests_info = data['response']['requests']
                return {
                    'valid': True,
                    'firstname': account.get('firstname', 'User'),
                    'limit_day': requests_info.get('limit_day', 0),
                    'current': requests_info.get('current', 0),
                    'remaining': requests_info.get('limit_day', 0) - requests_info.get('current', 0)
                }
            else:
                if st.session_state.get('debug_mode', False):
                    st.warning(f"🔍 Debug: Keine 'response' in API-Antwort gefunden")
                    st.json(data)
        else:
            st.error(f"❌ API-Fehler: Status Code {response.status_code}")
            if st.session_state.get('debug_mode', False):
                st.text(f"Response: {response.text[:500]}")
                
    except requests.exceptions.Timeout:
        st.error("⏱️ API-Timeout: Verbindung dauerte zu lange")
    except requests.exceptions.ConnectionError:
        st.error("🔌 Verbindungsfehler: Bitte Internetverbindung prüfen")
    except Exception as e:
        st.error(f"❌ API-Verbindungsfehler: {str(e)}")
        if st.session_state.get('debug_mode', False):
            st.text(f"Traceback:\n{traceback.format_exc()}")
    
    return {'valid': False}

@st.cache_data(ttl=3600, show_spinner=False)
def load_fixtures(_api_key, league_id, season):
    """Lädt kommende Spielansetzungen mit verbesserter Fehlerbehandlung"""
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {'x-apisports-key': _api_key}
    
    # Datum für die nächsten 7 Tage
    today = datetime.now().strftime('%Y-%m-%d')
    next_week = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
    
    params = {
        'league': league_id,
        'season': season,
        'from': today,
        'to': next_week
    }
    
    # Debug-Info
    if st.session_state.get('debug_mode', False):
        st.info(f"🔍 Debug: Lade Fixtures für Liga {league_id}, Saison {season}")
        st.text(f"Zeitraum: {today} bis {next_week}")
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        
        # Debug Response
        if st.session_state.get('debug_mode', False):
            st.info(f"🔍 Debug: Fixtures Response Code: {response.status_code}")
            
        if response.status_code == 200:
            data = response.json()
            
            # Debug: Zeige Antwort-Struktur
            if st.session_state.get('debug_mode', False):
                st.text(f"Anzahl Fixtures empfangen: {len(data.get('response', []))}")
                if data.get('errors') and len(data['errors']) > 0:
                    st.warning(f"API Errors: {data['errors']}")
            
            fixtures = data.get('response', [])
            
            # Filtere nur wirklich zukünftige Spiele
            future_fixtures = []
            for fixture in fixtures:
                try:
                    fixture_date = datetime.fromisoformat(fixture['fixture']['date'].replace('Z', '+00:00'))
                    if fixture_date > datetime.now(fixture_date.tzinfo):
                        future_fixtures.append(fixture)
                    elif st.session_state.get('debug_mode', False):
                        st.text(f"Übersprungen (vergangen): {fixture['teams']['home']['name']} vs {fixture['teams']['away']['name']}")
                except Exception as e:
                    if st.session_state.get('debug_mode', False):
                        st.warning(f"Fehler beim Parsen von Fixture: {e}")
                    continue
            
            if st.session_state.get('debug_mode', False):
                st.success(f"✅ {len(future_fixtures)} zukünftige Fixtures gefunden")
            
            return future_fixtures
            
        elif response.status_code == 204:
            if st.session_state.get('debug_mode', False):
                st.warning("⚠️ Keine Daten verfügbar (204)")
            return []
        else:
            if st.session_state.get('debug_mode', False):
                st.error(f"❌ Fixtures Fehler: Status {response.status_code}")
                st.text(f"Response: {response.text[:500]}")
            return []
            
    except requests.exceptions.Timeout:
        if st.session_state.get('debug_mode', False):
            st.error("⏱️ Timeout beim Laden der Fixtures")
        return []
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"❌ Fehler beim Laden der Fixtures: {str(e)}")
            st.text(f"Traceback:\n{traceback.format_exc()}")
        return []

@st.cache_data(ttl=86400, show_spinner=False)
def load_historical_data(_api_key, league_id, season):
    """Lädt historische Spieldaten"""
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {'x-apisports-key': _api_key}
    
    params = {
        'league': league_id,
        'season': season,
        'status': 'FT'  # Nur beendete Spiele
    }
    
    # Debug-Info
    if st.session_state.get('debug_mode', False):
        st.info(f"🔍 Debug: Lade historische Daten für Liga {league_id}, Saison {season}")
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        
        # Debug Response
        if st.session_state.get('debug_mode', False):
            st.info(f"🔍 Debug: Historical Data Response Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            historical_data = data.get('response', [])
            
            if st.session_state.get('debug_mode', False):
                st.success(f"✅ {len(historical_data)} historische Spiele geladen")
                if len(historical_data) > 0:
                    # Zeige Beispielspiel
                    example = historical_data[0]
                    st.text(f"Beispiel: {example['teams']['home']['name']} {example['goals']['home']} - {example['goals']['away']} {example['teams']['away']['name']}")
            
            return historical_data
        else:
            if st.session_state.get('debug_mode', False):
                st.error(f"❌ Historical Data Fehler: Status {response.status_code}")
                st.text(f"Response: {response.text[:500]}")
            return []
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"❌ Fehler beim Laden historischer Daten: {str(e)}")
            st.text(f"Traceback:\n{traceback.format_exc()}")
        return []

def load_odds(api_key, fixture_id):
    """Lädt Quoten für ein Spiel mit Debug-Informationen"""
    url = "https://v3.football.api-sports.io/odds"
    headers = {'x-apisports-key': api_key}
    params = {
        'fixture': fixture_id,
        'bet': 1  # Match Winner
    }
    
    if st.session_state.get('debug_mode', False):
        st.text(f"🎲 Lade Quoten für Fixture ID: {fixture_id}")
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if st.session_state.get('debug_mode', False):
            st.text(f"Odds Response Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            odds_response = data.get('response', [])
            
            if st.session_state.get('debug_mode', False):
                st.text(f"Anzahl Buchmacher: {len(odds_response[0].get('bookmakers', [])) if odds_response else 0}")
            
            if odds_response and len(odds_response) > 0:
                best_odds = {'home': 0, 'draw': 0, 'away': 0}
                
                for bookmaker_data in odds_response[0].get('bookmakers', []):
                    bets = bookmaker_data.get('bets', [])
                    for bet in bets:
                        if bet.get('id') == 1:  # Match Winner
                            values = bet.get('values', [])
                            for value in values:
                                odd = float(value.get('odd', 0))
                                market = value.get('value', '')
                                
                                if market == 'Home' and odd > best_odds['home']:
                                    best_odds['home'] = odd
                                elif market == 'Draw' and odd > best_odds['draw']:
                                    best_odds['draw'] = odd
                                elif market == 'Away' and odd > best_odds['away']:
                                    best_odds['away'] = odd
                
                if best_odds['home'] > 0 and best_odds['draw'] > 0 and best_odds['away'] > 0:
                    if st.session_state.get('debug_mode', False):
                        st.text(f"Beste Quoten: H:{best_odds['home']:.2f} D:{best_odds['draw']:.2f} A:{best_odds['away']:.2f}")
                    return best_odds
                elif st.session_state.get('debug_mode', False):
                    st.warning(f"⚠️ Unvollständige Quoten: H:{best_odds['home']} D:{best_odds['draw']} A:{best_odds['away']}")
            elif st.session_state.get('debug_mode', False):
                st.warning("⚠️ Keine Quoten-Daten in der Antwort")
        
        return None
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"❌ Fehler beim Laden der Quoten: {str(e)}")
        return None

# ============================================================================
# DIXON-COLES MODELL (VERBESSERT)
# ============================================================================

def rho_correction(x, y, lambda_x, mu_y, rho):
    """Dixon-Coles Korrektur für niedrige Torzahlen"""
    if x == 0 and y == 0:
        return 1 - (lambda_x * mu_y * rho)
    elif x == 0 and y == 1:
        return 1 + (lambda_x * rho)
    elif x == 1 and y == 0:
        return 1 + (mu_y * rho)
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1.0

def dc_log_like(params, home_goals, away_goals, home_teams, away_teams, team_list):
    """Log-Likelihood für Dixon-Coles"""
    n_teams = len(team_list)
    attack = params[:n_teams]
    defence = params[n_teams:2*n_teams]
    home_adv = params[-2]
    rho = params[-1]
    
    rho = np.clip(rho, -0.3, 0.3)
    
    team_to_idx = {team: idx for idx, team in enumerate(team_list)}
    log_like = 0
    
    for i in range(len(home_goals)):
        if home_teams[i] not in team_to_idx or away_teams[i] not in team_to_idx:
            continue
            
        home_idx = team_to_idx[home_teams[i]]
        away_idx = team_to_idx[away_teams[i]]
        
        lambda_home = np.exp(np.clip(attack[home_idx] + defence[away_idx] + home_adv, -10, 10))
        mu_away = np.exp(np.clip(attack[away_idx] + defence[home_idx], -10, 10))
        
        lambda_home = np.clip(lambda_home, 0.1, 10)
        mu_away = np.clip(mu_away, 0.1, 10)
        
        try:
            correction = rho_correction(home_goals[i], away_goals[i], lambda_home, mu_away, rho)
            if correction > 0:
                log_like += (
                    np.log(correction) +
                    np.log(poisson.pmf(home_goals[i], lambda_home) + 1e-10) +
                    np.log(poisson.pmf(away_goals[i], mu_away) + 1e-10)
                )
        except:
            continue
    
    return -log_like

def train_dixon_coles_model(historical_data):
    """Trainiert verbessertes Dixon-Coles Modell mit Debugging"""
    # Debug-Info
    if st.session_state.get('debug_mode', False):
        st.info(f"🔍 Debug: Starte Modelltraining mit {len(historical_data) if historical_data else 0} historischen Spielen")
    
    if not historical_data or len(historical_data) < 30:
        if st.session_state.get('debug_mode', False):
            st.warning(f"⚠️ Zu wenig historische Daten: {len(historical_data) if historical_data else 0} Spiele")
        return None
    
    matches = []
    
    for match in historical_data:
        try:
            if match['fixture']['status']['short'] == 'FT':
                home_goals = match['goals']['home']
                away_goals = match['goals']['away']
                
                if home_goals is not None and away_goals is not None:
                    matches.append({
                        'home_team': match['teams']['home']['name'],
                        'away_team': match['teams']['away']['name'],
                        'home_goals': home_goals,
                        'away_goals': away_goals
                    })
        except Exception as e:
            if st.session_state.get('debug_mode', False):
                st.text(f"Fehler beim Verarbeiten eines Spiels: {e}")
            continue
    
    if st.session_state.get('debug_mode', False):
        st.text(f"Verarbeitete Matches: {len(matches)}")
    
    if len(matches) < 30:
        if st.session_state.get('debug_mode', False):
            st.warning(f"⚠️ Zu wenig gültige Matches nach Verarbeitung: {len(matches)}")
        return None
    
    df = pd.DataFrame(matches)
    
    # Filtere Teams mit zu wenigen Spielen
    home_counts = df['home_team'].value_counts()
    away_counts = df['away_team'].value_counts()
    team_counts = home_counts.add(away_counts, fill_value=0)
    valid_teams = team_counts[team_counts >= 5].index.tolist()
    
    if st.session_state.get('debug_mode', False):
        st.text(f"Teams mit >= 5 Spielen: {len(valid_teams)}")
    
    df = df[(df['home_team'].isin(valid_teams)) & (df['away_team'].isin(valid_teams))]
    
    if len(df) < 30:
        if st.session_state.get('debug_mode', False):
            st.warning(f"⚠️ Zu wenig Daten nach Teamfilterung: {len(df)}")
        return None
    
    teams = sorted(list(set(df['home_team'].unique()) | set(df['away_team'].unique())))
    n_teams = len(teams)
    
    if st.session_state.get('debug_mode', False):
        st.text(f"Finale Teamanzahl: {n_teams}")
        st.text(f"Finale Spielanzahl: {len(df)}")
        st.text(f"Durchschn. Tore - Heim: {df['home_goals'].mean():.2f}, Auswärts: {df['away_goals'].mean():.2f}")
    
    if n_teams < 8:
        if st.session_state.get('debug_mode', False):
            st.warning(f"⚠️ Zu wenig Teams für Modell: {n_teams}")
        return None
    
    def constraint_func(params):
        return np.mean(np.exp(params[:n_teams])) - 1.0
    
    constraints = {'type': 'eq', 'fun': constraint_func}
    
    initial_params = np.concatenate([
        np.zeros(n_teams),      # Attack
        np.zeros(n_teams),      # Defence
        [0.3],                  # Home advantage
        [-0.1]                  # Rho
    ])
    
    bounds = [(None, None)] * (2 * n_teams) + [(0, 1), (-0.3, 0.3)]
    
    try:
        if st.session_state.get('debug_mode', False):
            st.text("🤖 Starte Optimierung...")
        
        result = minimize(
            dc_log_like,
            initial_params,
            args=(
                df['home_goals'].values,
                df['away_goals'].values,
                df['home_team'].values,
                df['away_team'].values,
                teams
            ),
            constraints=constraints,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 200, 'disp': False}
        )
        
        if st.session_state.get('debug_mode', False):
            st.text(f"Optimierung abgeschlossen - Funktionswert: {result.fun:.2f}")
            st.text(f"Erfolg: {result.success}, Iterationen: {result.nit}")
        
        if result.fun < np.inf:
            params = result.x
            
            attack_dict = {teams[i]: params[i] for i in range(n_teams)}
            defence_dict = {teams[i]: params[n_teams + i] for i in range(n_teams)}
            
            if st.session_state.get('debug_mode', False):
                st.success("✅ Modell erfolgreich trainiert!")
                st.text(f"Home Advantage: {params[-2]:.3f}")
                st.text(f"Rho: {params[-1]:.3f}")
                
                # Top Teams
                best_attack = max(attack_dict.items(), key=lambda x: x[1])
                worst_attack = min(attack_dict.items(), key=lambda x: x[1])
                st.text(f"Bester Angriff: {best_attack[0][:20]} ({best_attack[1]:.3f})")
                st.text(f"Schwächster Angriff: {worst_attack[0][:20]} ({worst_attack[1]:.3f})")
            
            return {
                'teams': teams,
                'attack': attack_dict,
                'defence': defence_dict,
                'home_adv': params[-2],
                'rho': params[-1],
                'matches_used': len(df),
                'convergence': result.success
            }
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"❌ Fehler bei Modelloptimierung: {str(e)}")
            st.text(f"Traceback:\n{traceback.format_exc()}")
        pass
    
    return None

def predict_match(model, home_team, away_team, max_goals=6):
    """Verbesserte Spielvorhersage"""
    if not model or home_team not in model['attack'] or away_team not in model['attack']:
        return None
    
    try:
        lambda_home = np.exp(
            model['attack'][home_team] + 
            model['defence'][away_team] + 
            model['home_adv']
        )
        mu_away = np.exp(
            model['attack'][away_team] + 
            model['defence'][home_team]
        )
        
        lambda_home = np.clip(lambda_home, 0.5, 5)
        mu_away = np.clip(mu_away, 0.5, 5)
        
        prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
        
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                prob_matrix[i, j] = (
                    rho_correction(i, j, lambda_home, mu_away, model['rho']) *
                    poisson.pmf(i, lambda_home) *
                    poisson.pmf(j, mu_away)
                )
        
        prob_matrix = prob_matrix / prob_matrix.sum()
        
        prob_home = np.sum(np.tril(prob_matrix, -1))
        prob_draw = np.sum(np.diag(prob_matrix))
        prob_away = np.sum(np.triu(prob_matrix, 1))
        
        confidence = 'high' if model.get('matches_used', 0) > 100 else 'medium' if model.get('matches_used', 0) > 50 else 'low'
        
        return {
            'prob_home': prob_home,
            'prob_draw': prob_draw,
            'prob_away': prob_away,
            'expected_home_goals': lambda_home,
            'expected_away_goals': mu_away,
            'confidence': confidence
        }
    except:
        return None

# ============================================================================
# VALUE BETTING
# ============================================================================

def calculate_value(fair_prob, market_odds):
    """Berechnet Value/Edge"""
    if market_odds <= 1.01 or fair_prob <= 0:
        return 0
    
    edge = fair_prob * market_odds - 1
    value_percent = edge * 100
    
    return value_percent

def calculate_kelly_stake(fair_prob, market_odds, bankroll, fraction=0.2):
    """Konservativerer Kelly-Einsatz"""
    if market_odds <= 1.01 or fair_prob <= 0.05:
        return 0
    
    b = market_odds - 1
    p = fair_prob
    q = 1 - fair_prob
    
    kelly = (b * p - q) / b
    
    if kelly <= 0:
        return 0
    
    kelly_fraction = kelly * fraction
    max_stake = bankroll * 0.03
    min_stake = bankroll * 0.01
    
    stake = bankroll * kelly_fraction
    stake = min(stake, max_stake)
    stake = max(stake, min_stake) if kelly_fraction > 0 else 0
    
    return round(stake, 2)

# ============================================================================
# HAUPTANALYSE
# ============================================================================

def analyze_leagues(api_key, selected_leagues, season):
    """Analysiert ausgewählte Ligen mit erweitertem Debugging"""
    all_value_bets = []
    analysis_summary = {
        'total_fixtures_analyzed': 0,
        'models_trained': 0,
        'value_bets_found': 0,
        'api_calls_used': 0,
        'errors': []
    }
    
    # Debug-Info
    if st.session_state.get('debug_mode', False):
        st.info(f"🔍 Debug: Starte Analyse für {len(selected_leagues)} Ligen, Saison {season}")
    
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        details_text = st.empty()
    
    total_leagues = len(selected_leagues)
    
    for idx, league_name in enumerate(selected_leagues):
        league_id = LEAGUES[league_name]
        
        progress = (idx + 1) / total_leagues
        progress_bar.progress(progress)
        status_text.text(f"Analysiere {league_name}... ({idx + 1}/{total_leagues})")
        
        try:
            # Lade historische Daten
            details_text.text("📊 Lade historische Daten...")
            historical_data = load_historical_data(api_key, league_id, season)
            analysis_summary['api_calls_used'] += 1
            
            if not historical_data or len(historical_data) < 30:
                error_msg = f"⚠️ {league_name}: Nicht genug Daten ({len(historical_data) if historical_data else 0} Spiele)"
                details_text.warning(error_msg)
                if st.session_state.get('debug_mode', False):
                    analysis_summary['errors'].append(error_msg)
                time.sleep(1)
                continue
            
            # Trainiere Modell
            details_text.text("🤖 Trainiere Modell...")
            model = train_dixon_coles_model(historical_data)
            
            if not model:
                error_msg = f"⚠️ {league_name}: Modelltraining fehlgeschlagen"
                details_text.warning(error_msg)
                if st.session_state.get('debug_mode', False):
                    analysis_summary['errors'].append(error_msg)
                time.sleep(1)
                continue
            
            analysis_summary['models_trained'] += 1
            
            # Lade kommende Spiele
            details_text.text("🔮 Lade kommende Spiele...")
            fixtures = load_fixtures(api_key, league_id, season)
            analysis_summary['api_calls_used'] += 1
            
            if not fixtures:
                if st.session_state.get('debug_mode', False):
                    error_msg = f"⚠️ {league_name}: Keine kommenden Spiele gefunden"
                    analysis_summary['errors'].append(error_msg)
                continue
            
            if st.session_state.get('debug_mode', False):
                st.text(f"📅 {len(fixtures)} kommende Spiele gefunden")
            
            # Analysiere Spiele
            for fixture in fixtures[:5]:
                try:
                    home_team = fixture['teams']['home']['name']
                    away_team = fixture['teams']['away']['name']
                    fixture_id = fixture['fixture']['id']
                    fixture_date = fixture['fixture']['date']
                    
                    if st.session_state.get('debug_mode', False):
                        details_text.text(f"⚽ Analysiere: {home_team} vs {away_team}")
                    
                    prediction = predict_match(model, home_team, away_team)
                    
                    if not prediction:
                        if st.session_state.get('debug_mode', False):
                            st.text(f"⚠️ Keine Vorhersage möglich für {home_team} vs {away_team}")
                        continue
                    
                    analysis_summary['total_fixtures_analyzed'] += 1
                    
                    # Lade Quoten
                    odds = load_odds(api_key, fixture_id)
                    analysis_summary['api_calls_used'] += 1
                    
                    if not odds:
                        if st.session_state.get('debug_mode', False):
                            st.text(f"⚠️ Keine Quoten für {home_team} vs {away_team}")
                        continue
                    
                    # Value-Berechnung
                    value_home = calculate_value(prediction['prob_home'], odds['home'])
                    value_draw = calculate_value(prediction['prob_draw'], odds['draw'])
                    value_away = calculate_value(prediction['prob_away'], odds['away'])
                    
                    if st.session_state.get('debug_mode', False):
                        st.text(f"Value: H:{value_home:.1f}% D:{value_draw:.1f}% A:{value_away:.1f}%")
                    
                    markets = [
                        ('Heimsieg', value_home, odds['home'], prediction['prob_home'], '1'),
                        ('Unentschieden', value_draw, odds['draw'], prediction['prob_draw'], 'X'),
                        ('Auswärtssieg', value_away, odds['away'], prediction['prob_away'], '2')
                    ]
                    
                    best_market = max(markets, key=lambda x: x[1])
                    
                    if best_market[1] >= 5:  # Mindestens 5% Value
                        stake = calculate_kelly_stake(best_market[3], best_market[2], st.session_state.bankroll)
                        
                        if stake > 0:
                            try:
                                match_datetime = datetime.fromisoformat(fixture_date.replace('Z', '+00:00'))
                                formatted_date = match_datetime.strftime('%d.%m. %H:%M')
                            except:
                                formatted_date = fixture_date
                            
                            all_value_bets.append({
                                'Liga': league_name,
                                'Spiel': f"{home_team} - {away_team}",
                                'Datum': formatted_date,
                                'Tipp': best_market[4],
                                'Markt': best_market[0],
                                'Quote': f"{best_market[2]:.2f}",
                                'Fair': f"{best_market[3]*100:.1f}%",
                                'Value': f"{best_market[1]:.1f}%",
                                'Einsatz': f"€{stake:.2f}",
                                'Konfidenz': prediction['confidence']
                            })
                            analysis_summary['value_bets_found'] += 1
                            
                            if st.session_state.get('debug_mode', False):
                                st.success(f"✅ Value Bet gefunden: {home_team} vs {away_team}")
                        
                except Exception as e:
                    if st.session_state.get('debug_mode', False):
                        st.error(f"❌ Fehler bei Spielanalyse: {str(e)}")
                        analysis_summary['errors'].append(f"Spielanalyse-Fehler: {str(e)}")
                    continue
                
                time.sleep(0.5)
                
        except Exception as e:
            error_msg = f"❌ Fehler bei Liga {league_name}: {str(e)}"
            if st.session_state.get('debug_mode', False):
                st.error(error_msg)
                st.text(f"Traceback:\n{traceback.format_exc()}")
                analysis_summary['errors'].append(error_msg)
    
    progress_bar.empty()
    status_text.empty()
    details_text.empty()
    
    # Debug-Zusammenfassung
    if st.session_state.get('debug_mode', False) and analysis_summary['errors']:
        with st.expander("🔍 Debug: Fehlerzusammenfassung", expanded=False):
            for error in analysis_summary['errors']:
                st.text(f"• {error}")
    
    return all_value_bets, analysis_summary

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.title("⚽ Wetten-Analyst Pro")
    st.caption(f"Version 2.0 | Dixon-Coles Modell | {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    
    # Tabs
    tab_settings, tab_analysis, tab_help = st.tabs([
        "⚙️ Einstellungen", 
        "🎯 Analyse", 
        "❓ Hilfe"
    ])
    
    with tab_settings:
        st.header("⚙️ Einstellungen")
        
        # Debug-Modus Toggle
        st.subheader("🔧 Debug-Modus")
        debug_mode = st.checkbox(
            "Debug-Modus aktivieren",
            value=st.session_state.get('debug_mode', False),
            help="Zeigt detaillierte Informationen zur Fehlersuche an"
        )
        st.session_state.debug_mode = debug_mode
        
        if debug_mode:
            st.info("🔍 Debug-Modus ist AKTIV - Zusätzliche Informationen werden angezeigt")
        
        st.divider()
        
        # API-Key
        st.subheader("🔑 API-Konfiguration")
        api_key = st.text_input(
            "API-Football Key",
            type="password",
            value=st.session_state.api_key,
            placeholder="Dein API-Key hier eingeben...",
            help="Kostenlos auf api-football.com erhältlich"
        )
        st.session_state.api_key = api_key
        
        if api_key:
            status = check_api_status(api_key)
            if status and status.get('valid'):
                col1, col2, col3 = st.columns(3)
                col1.metric("Status", "✅ Aktiv")
                col2.metric("Heute genutzt", f"{status['current']}/{status['limit_day']}")
                col3.metric("Verbleibend", status['remaining'])
        
        # Liga-Auswahl
        st.subheader("🏆 Liga-Auswahl")
        
        selected_leagues = st.multiselect(
            "Wähle Ligen für die Analyse",
            options=list(LEAGUES.keys()),
            default=list(LEAGUES.keys())[:3]
        )
        st.session_state.selected_leagues = selected_leagues
        
        # Saison
        current_year = datetime.now().year
        season = st.selectbox(
            "Saison",
            options=[current_year, current_year - 1],
            format_func=lambda x: f"{x}/{x+1}"
        )
        
        # Bankroll
        st.subheader("💰 Bankroll")
        st.session_state.bankroll = st.number_input(
            "Aktuelle Bankroll (€)",
            min_value=10.0,
            max_value=100000.0,
            value=st.session_state.bankroll,
            step=100.0
        )
    
    with tab_analysis:
        st.header("🎯 Value-Bet Analyse")
        
        if not st.session_state.api_key:
            st.warning("⚠️ Bitte zuerst API-Key eingeben")
            st.stop()
        
        if not st.session_state.selected_leagues:
            st.info("📋 Bitte Ligen auswählen")
            st.stop()
        
        if st.button("🚀 ANALYSE STARTEN", type="primary", use_container_width=True):
            with st.spinner("Analysiere..."):
                value_bets, summary = analyze_leagues(
                    st.session_state.api_key,
                    st.session_state.selected_leagues,
                    season
                )
                
                st.session_state.last_analysis = {
                    'bets': value_bets,
                    'summary': summary,
                    'timestamp': datetime.now()
                }
        
        # Ergebnisse
        if st.session_state.last_analysis:
            analysis = st.session_state.last_analysis
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Spiele analysiert", analysis['summary']['total_fixtures_analyzed'])
            col2.metric("Value Bets", analysis['summary']['value_bets_found'])
            col3.metric("API-Anfragen", analysis['summary']['api_calls_used'])
            
            if analysis['bets']:
                st.subheader(f"🎯 {len(analysis['bets'])} Value Bets gefunden")
                
                df_bets = pd.DataFrame(analysis['bets'])
                st.dataframe(df_bets, use_container_width=True, height=400)
                
                # Zusammenfassung
                with st.expander("📊 Zusammenfassung", expanded=True):
                    total_stake = sum([float(bet['Einsatz'].replace('€', '')) for bet in analysis['bets']])
                    avg_value = np.mean([float(bet['Value'].replace('%', '')) for bet in analysis['bets']])
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Gesamt-Einsatz", f"€{total_stake:.2f}")
                    col2.metric("Durchschn. Value", f"{avg_value:.1f}%")
                    col3.metric("ROI Erwartung", f"{avg_value * 0.8:.1f}%")
            else:
                st.info("Keine Value Bets gefunden. Versuche andere Ligen oder warte auf bessere Quoten.")
    
    with tab_help:
        st.header("❓ Hilfe & FAQ")
        
        with st.expander("🚀 Schnellstart"):
            st.markdown("""
            1. **API-Key holen**: Registriere dich kostenlos auf [api-football.com](https://www.api-football.com/)
            2. **Einstellungen**: Gib deinen API-Key ein und wähle Ligen
            3. **Debug-Modus**: Aktiviere für detaillierte Fehlersuche
            4. **Analyse starten**: Klicke auf "ANALYSE STARTEN"
            5. **Ergebnisse**: Sieh dir die Value Bets an
            """)
        
        with st.expander("🔍 Debug-Modus"):
            st.markdown("""
            **Was zeigt der Debug-Modus?**
            - API-Verbindungsstatus und Antworten
            - Anzahl geladener Daten
            - Modelltraining-Details
            - Value-Berechnungen für jedes Spiel
            - Vollständige Fehlermeldungen
            
            **Wann aktivieren?**
            - Wenn die Analyse nicht funktioniert
            - Bei API-Verbindungsproblemen
            - Um zu verstehen, was im Hintergrund passiert
            """)
        
        with st.expander("⚠️ Häufige Probleme"):
            st.markdown("""
            **"Nicht genug Daten"**
            - Liga hat < 30 historische Spiele
            - Lösung: Andere Saison oder Liga wählen
            
            **"Keine kommenden Spiele"**
            - Spielpause oder keine Fixtures in 7 Tagen
            - Lösung: Andere Liga wählen
            
            **"API-Limit erreicht"**
            - 100 Anfragen/Tag im kostenlosen Plan
            - Lösung: Morgen wieder versuchen
            
            **"Modelltraining fehlgeschlagen"**
            - Zu wenige Teams/Spiele
            - Lösung: Liga mit mehr Teams wählen
            """)
        
        with st.expander("📈 Value Betting erklärt"):
            st.markdown("""
            **Was ist Value Betting?**
            Eine Wette hat "Value", wenn die faire Wahrscheinlichkeit höher ist als die implizite Wahrscheinlichkeit der Quote.
            
            **Beispiel:**
            - Quote: 2.50 (implizite Wahrscheinlichkeit: 40%)
            - Faire Wahrscheinlichkeit laut Modell: 45%
            - Value: 45% × 2.50 - 1 = 12.5%
            
            **Kelly Criterion:**
            Berechnet den optimalen Einsatz basierend auf Value und Bankroll.
            Wir nutzen 20% Kelly für konservativere Einsätze.
            """)
        
        st.divider()
        st.caption("⚠️ Glücksspiel kann süchtig machen. Spiele verantwortungsvoll!")

if __name__ == "__main__":
    main()
