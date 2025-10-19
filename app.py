"""
Fußballwetten-Analyse-App (Verbesserte Version)
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
        return None
        
    url = "https://v3.football.api-sports.io/status"
    headers = {'x-apisports-key': api_key}
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
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
    except Exception as e:
        st.error(f"API-Verbindungsfehler: {str(e)}")
    
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
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            fixtures = data.get('response', [])
            
            # Filtere nur wirklich zukünftige Spiele
            future_fixtures = []
            for fixture in fixtures:
                try:
                    fixture_date = datetime.fromisoformat(fixture['fixture']['date'].replace('Z', '+00:00'))
                    if fixture_date > datetime.now(fixture_date.tzinfo):
                        future_fixtures.append(fixture)
                except:
                    continue
            
            return future_fixtures
        elif response.status_code == 204:
            return []  # Keine Daten verfügbar
        else:
            return []
    except requests.exceptions.Timeout:
        return []
    except Exception:
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
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data.get('response', [])
        return []
    except Exception:
        return []

def load_odds(api_key, fixture_id):
    """Lädt Quoten für ein Spiel"""
    url = "https://v3.football.api-sports.io/odds"
    headers = {'x-apisports-key': api_key}
    params = {
        'fixture': fixture_id,
        'bet': 1  # Match Winner
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            odds_response = data.get('response', [])
            
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
                    return best_odds
        
        return None
    except:
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
    """Trainiert verbessertes Dixon-Coles Modell"""
    if not historical_data or len(historical_data) < 30:
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
        except:
            continue
    
    if len(matches) < 30:
        return None
    
    df = pd.DataFrame(matches)
    
    # Filtere Teams mit zu wenigen Spielen
    home_counts = df['home_team'].value_counts()
    away_counts = df['away_team'].value_counts()
    team_counts = home_counts.add(away_counts, fill_value=0)
    valid_teams = team_counts[team_counts >= 5].index.tolist()
    
    df = df[(df['home_team'].isin(valid_teams)) & (df['away_team'].isin(valid_teams))]
    
    if len(df) < 30:
        return None
    
    teams = sorted(list(set(df['home_team'].unique()) | set(df['away_team'].unique())))
    n_teams = len(teams)
    
    if n_teams < 8:
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
        
        if result.fun < np.inf:
            params = result.x
            return {
                'teams': teams,
                'attack': {teams[i]: params[i] for i in range(n_teams)},
                'defence': {teams[i]: params[n_teams + i] for i in range(n_teams)},
                'home_adv': params[-2],
                'rho': params[-1],
                'matches_used': len(df),
                'convergence': result.success
            }
    except:
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
    """Analysiert ausgewählte Ligen"""
    all_value_bets = []
    analysis_summary = {
        'total_fixtures_analyzed': 0,
        'models_trained': 0,
        'value_bets_found': 0,
        'api_calls_used': 0
    }
    
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
        
        # Lade historische Daten
        details_text.text("📊 Lade historische Daten...")
        historical_data = load_historical_data(api_key, league_id, season)
        analysis_summary['api_calls_used'] += 1
        
        if not historical_data or len(historical_data) < 30:
            details_text.warning(f"⚠️ {league_name}: Nicht genug Daten")
            time.sleep(1)
            continue
        
        # Trainiere Modell
        details_text.text("🤖 Trainiere Modell...")
        model = train_dixon_coles_model(historical_data)
        
        if not model:
            details_text.warning(f"⚠️ {league_name}: Modelltraining fehlgeschlagen")
            time.sleep(1)
            continue
        
        analysis_summary['models_trained'] += 1
        
        # Lade kommende Spiele
        details_text.text("🔮 Lade kommende Spiele...")
        fixtures = load_fixtures(api_key, league_id, season)
        analysis_summary['api_calls_used'] += 1
        
        if not fixtures:
            continue
        
        # Analysiere Spiele
        for fixture in fixtures[:5]:
            try:
                home_team = fixture['teams']['home']['name']
                away_team = fixture['teams']['away']['name']
                fixture_id = fixture['fixture']['id']
                fixture_date = fixture['fixture']['date']
                
                prediction = predict_match(model, home_team, away_team)
                
                if not prediction:
                    continue
                
                analysis_summary['total_fixtures_analyzed'] += 1
                
                # Lade Quoten
                odds = load_odds(api_key, fixture_id)
                analysis_summary['api_calls_used'] += 1
                
                if not odds:
                    continue
                
                # Value-Berechnung
                value_home = calculate_value(prediction['prob_home'], odds['home'])
                value_draw = calculate_value(prediction['prob_draw'], odds['draw'])
                value_away = calculate_value(prediction['prob_away'], odds['away'])
                
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
                    
            except:
                continue
            
            time.sleep(0.5)
    
    progress_bar.empty()
    status_text.empty()
    details_text.empty()
    
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
        
        # API-Key
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
                st.dataframe(df_bets, use_container_width=True, hide_index=True)
                
                csv = df_bets.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Als CSV herunterladen",
                    csv,
                    f"value_bets_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )
            else:
                st.info("Keine Value Bets mit >5% Edge gefunden")
    
    with tab_help:
        st.header("❓ Hilfe")
        
        with st.expander("🚀 Schnellstart"):
            st.markdown("""
            1. **API-Key eingeben** (api-football.com)
            2. **Ligen auswählen**
            3. **Analyse starten**
            4. **Value Bets prüfen**
            """)
        
        with st.expander("📊 Dixon-Coles Modell"):
            st.markdown("""
            Statistisches Modell zur Fußballvorhersage:
            - Poisson-Verteilung für Tore
            - Team-spezifische Stärken
            - Heimvorteil
            - Korrektur für niedrige Torzahlen
            """)
        
        with st.expander("💰 Value Betting"):
            st.markdown("""
            Value = Faire Quote höher als Marktquote
            - Nur Wetten mit >5% Edge
            - Kelly-Kriterium für Einsätze
            - Langfristige Strategie (100+ Wetten)
            """)

if __name__ == "__main__":
    main()
