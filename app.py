"""
Fußballwetten-Analyse-App
Automatische Analyse aller Top-Ligen mit Dixon-Coles-Modell
Optimiert für mobile Geräte (iPhone, Android)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from scipy.stats import poisson
from scipy.optimize import minimize
import time

# ============================================================================
# KONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="⚽ Wetten-Analyst",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "Professionelle Fußballwetten-Analyse mit Dixon-Coles-Modell"
    }
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    .stButton > button {
        width: 100%;
        padding: 0.75rem;
        font-size: 1rem;
    }
    .dataframe {
        font-size: 0.85rem;
    }
    h1 {
        font-size: 1.8rem !important;
    }
    h2 {
        font-size: 1.4rem !important;
    }
    h3 {
        font-size: 1.2rem !important;
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
    "🇫🇷 Ligue 1": 61
}

# ============================================================================
# SESSION STATE
# ============================================================================

if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 1000.0
    st.session_state.initial_bankroll = 1000.0
    st.session_state.bet_history = []
    st.session_state.api_key = ""

# ============================================================================
# API-FUNKTIONEN
# ============================================================================

@st.cache_data(ttl=3600)
def load_fixtures(_api_key, league_id, season):
    """Lädt kommende Spielansetzungen (gecacht für 1 Stunde)"""
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {'x-apisports-key': _api_key}
    params = {'league': league_id, 'season': season, 'next': 5}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('response', [])
        return []
    except Exception as e:
        st.error(f"API-Fehler bei Liga {league_id}: {str(e)}")
        return []

@st.cache_data(ttl=86400)
def load_historical_data(_api_key, league_id, season):
    """Lädt historische Spieldaten (gecacht für 24 Stunden)"""
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {'x-apisports-key': _api_key}
    params = {'league': league_id, 'season': season, 'status': 'FT'}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data.get('response', [])
        return []
    except Exception as e:
        st.error(f"Fehler beim Laden historischer Daten (Liga {league_id}): {str(e)}")
        return []

def load_odds(api_key, fixture_id):
    """Lädt Quoten für ein Spiel"""
    url = "https://v3.football.api-sports.io/odds"
    headers = {'x-apisports-key': api_key}
    params = {'fixture': fixture_id, 'bet': 1}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('response', [])
        return []
    except:
        return []

# ============================================================================
# DIXON-COLES MODELL
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
    
    team_to_idx = {team: idx for idx, team in enumerate(team_list)}
    log_like = 0
    
    for i in range(len(home_goals)):
        home_idx = team_to_idx[home_teams[i]]
        away_idx = team_to_idx[away_teams[i]]
        
        lambda_home = np.exp(attack[home_idx] + defence[away_idx] + home_adv)
        mu_away = np.exp(attack[away_idx] + defence[home_idx])
        
        try:
            log_like += (
                np.log(rho_correction(home_goals[i], away_goals[i], lambda_home, mu_away, rho)) +
                np.log(poisson.pmf(home_goals[i], lambda_home)) +
                np.log(poisson.pmf(away_goals[i], mu_away))
            )
        except:
            continue
    
    return -log_like

def train_dixon_coles_model(historical_data):
    """Trainiert Dixon-Coles Modell"""
    if not historical_data or len(historical_data) < 50:
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
    
    if len(matches) < 50:
        return None
    
    df = pd.DataFrame(matches)
    teams = sorted(list(set(df['home_team'].unique()) | set(df['away_team'].unique())))
    n_teams = len(teams)
    
    if n_teams < 10:
        return None
    
    def constraint_func(params):
        return np.sum(params[:n_teams]) - n_teams
    
    constraints = {'type': 'eq', 'fun': constraint_func}
    
    initial_params = np.concatenate([
        np.ones(n_teams),
        np.ones(n_teams),
        [0.3],
        [-0.13]
    ])
    
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
            method='SLSQP',
            options={'maxiter': 100, 'disp': False}
        )
        
        if result.success:
            params = result.x
            return {
                'teams': teams,
                'attack': {teams[i]: params[i] for i in range(n_teams)},
                'defence': {teams[i]: params[n_teams + i] for i in range(n_teams)},
                'home_adv': params[-2],
                'rho': params[-1]
            }
    except:
        pass
    
    return None

def predict_match(model, home_team, away_team, max_goals=5):
    """Berechnet Spielvorhersage"""
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
        
        prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
        
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                prob_matrix[i, j] = (
                    rho_correction(i, j, lambda_home, mu_away, model['rho']) *
                    poisson.pmf(i, lambda_home) *
                    poisson.pmf(j, mu_away)
                )
        
        prob_home = np.sum(np.tril(prob_matrix, -1))
        prob_draw = np.sum(np.diag(prob_matrix))
        prob_away = np.sum(np.triu(prob_matrix, 1))
        
        return {
            'prob_home': prob_home,
            'prob_draw': prob_draw,
            'prob_away': prob_away
        }
    except:
        return None

# ============================================================================
# VALUE BETTING
# ============================================================================

def calculate_value(fair_prob, market_odds):
    """Berechnet Value/Edge"""
    if market_odds <= 1.0:
        return 0
    implied_prob = 1 / market_odds
    value = (fair_prob - implied_prob) / implied_prob * 100
    return value

def calculate_kelly_stake(fair_prob, market_odds, bankroll, fraction=0.25):
    """Berechnet Kelly-Einsatz"""
    if market_odds <= 1.0 or fair_prob <= 0:
        return 0
    
    q = 1 - fair_prob
    b = market_odds - 1
    kelly = (fair_prob * b - q) / b
    kelly_fraction = kelly * fraction
    
    if kelly_fraction < 0:
        return 0
    if kelly_fraction > 0.05:
        kelly_fraction = 0.05
    
    stake = bankroll * kelly_fraction
    return stake

# ============================================================================
# HAUPTANALYSE
# ============================================================================

def analyze_all_leagues(api_key, season):
    """Analysiert alle konfigurierten Ligen"""
    all_value_bets = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_leagues = len(LEAGUES)
    
    for idx, (league_name, league_id) in enumerate(LEAGUES.items()):
        status_text.text(f"Analysiere {league_name}...")
        progress_bar.progress((idx + 1) / total_leagues)
        
        # Historische Daten laden
        historical_data = load_historical_data(api_key, league_id, season - 1)
        
        if not historical_data or len(historical_data) < 50:
            st.warning(f"⚠️ {league_name}: Nicht genug Daten")
            continue
        
        # Modell trainieren
        model = train_dixon_coles_model(historical_data)
        
        if not model:
            st.warning(f"⚠️ {league_name}: Modelltraining fehlgeschlagen")
            continue
        
        # Kommende Spiele laden
        fixtures = load_fixtures(api_key, league_id, season)
        
        if not fixtures:
            continue
        
        # Spiele analysieren
        for fixture in fixtures[:3]:  # Max 3 Spiele pro Liga
            try:
                home_team = fixture['teams']['home']['name']
                away_team = fixture['teams']['away']['name']
                fixture_id = fixture['fixture']['id']
                fixture_date = fixture['fixture']['date']
                
                prediction = predict_match(model, home_team, away_team)
                
                if not prediction:
                    continue
                
                # Quoten laden
                odds_data = load_odds(api_key, fixture_id)
                
                if not odds_data or len(odds_data) == 0:
                    continue
                
                try:
                    bookmaker = odds_data[0]['bookmakers'][0]
                    bets = bookmaker['bets'][0]['values']
                    
                    odds_home = float([b['odd'] for b in bets if b['value'] == 'Home'][0])
                    odds_draw = float([b['odd'] for b in bets if b['value'] == 'Draw'][0])
                    odds_away = float([b['odd'] for b in bets if b['value'] == 'Away'][0])
                    
                    # Value berechnen
                    value_home = calculate_value(prediction['prob_home'], odds_home)
                    value_draw = calculate_value(prediction['prob_draw'], odds_draw)
                    value_away = calculate_value(prediction['prob_away'], odds_away)
                    
                    values = [
                        ('1', value_home, odds_home, prediction['prob_home']),
                        ('X', value_draw, odds_draw, prediction['prob_draw']),
                        ('2', value_away, odds_away, prediction['prob_away'])
                    ]
                    
                    best_bet = max(values, key=lambda x: x[1])
                    
                    if best_bet[1] > 5:  # Nur Value > 5%
                        stake = calculate_kelly_stake(
                            best_bet[3],
                            best_bet[2],
                            st.session_state.bankroll
                        )
                        
                        all_value_bets.append({
                            'Liga': league_name,
                            'Spiel': f"{home_team} - {away_team}",
                            'Datum': datetime.fromisoformat(fixture_date.replace('Z', '+00:00')).strftime('%d.%m %H:%M'),
                            'Tipp': best_bet[0],
                            'Quote': f"{best_bet[2]:.2f}",
                            'Value': f"{best_bet[1]:.1f}%",
                            'Einsatz': f"€{stake:.0f}"
                        })
                except:
                    continue
                    
            except:
                continue
            
            time.sleep(0.1)  # Kleine Pause zwischen Anfragen
    
    progress_bar.empty()
    status_text.empty()
    
    return all_value_bets

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.title("⚽ Wetten-Analyst")
    st.caption("Automatische Multi-Liga-Analyse mit Dixon-Coles")
    
    # Einstellungen
    with st.expander("⚙️ Einstellungen", expanded=not st.session_state.api_key):
        api_key = st.text_input(
            "API-Football Key",
            type="password",
            value=st.session_state.api_key,
            help="Kostenlos auf api-football.com"
        )
        st.session_state.api_key = api_key
        
        st.info(f"📊 Analysiert werden: {', '.join([name.split()[1] for name in LEAGUES.keys()])}")
        
        current_year = datetime.now().year
        season = st.selectbox("Saison", [current_year, current_year - 1], index=0)
    
    # Bankroll
    with st.expander("💰 Bankroll", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Aktuell", f"€{st.session_state.bankroll:.0f}")
        
        with col2:
            change = ((st.session_state.bankroll / st.session_state.initial_bankroll - 1) * 100)
            st.metric("Änderung", f"{change:.1f}%")
        
        with col3:
            if st.session_state.bet_history:
                total_staked = sum([bet['stake'] for bet in st.session_state.bet_history])
                total_return = sum([bet.get('return', 0) for bet in st.session_state.bet_history])
                roi = ((total_return - total_staked) / total_staked * 100) if total_staked > 0 else 0
                st.metric("ROI", f"{roi:.1f}%")
        
        if st.button("🔄 Reset"):
            st.session_state.bankroll = st.session_state.initial_bankroll
            st.session_state.bet_history = []
            st.rerun()
    
    if not api_key:
        st.warning("⚠️ API-Key erforderlich")
        st.info("""
        **Kostenloser API-Key:**
        1. Besuche [api-football.com](https://www.api-football.com/)
        2. Registriere dich kostenlos
        3. Kopiere deinen Key
        4. Füge ihn oben ein
        """)
        return
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Analyse", "📊 Protokoll", "📈 Stats"])
    
    with tab1:
        st.info("🔍 Die App analysiert automatisch alle Top-Ligen und zeigt nur die besten Value Bets (>5% Edge)")
        
        if st.button("🚀 Alle Ligen analysieren", type="primary", use_container_width=True):
            value_bets = analyze_all_leagues(api_key, season)
            
            if value_bets:
                st.success(f"✅ {len(value_bets)} Value Bets gefunden!")
                
                df_value = pd.DataFrame(value_bets)
                st.dataframe(df_value, use_container_width=True, hide_index=True)
                
                st.download_button(
                    "📥 Als CSV herunterladen",
                    df_value.to_csv(index=False).encode('utf-8'),
                    "value_bets.csv",
                    "text/csv",
                    use_container_width=True
                )
            else:
                st.info("Keine Value Bets mit >5% Edge gefunden")
    
    with tab2:
        st.subheader("📊 Wett-Protokoll")
        
        if st.session_state.bet_history:
            df_history = pd.DataFrame(st.session_state.bet_history)
            st.dataframe(df_history, use_container_width=True, hide_index=True)
        else:
            st.info("Noch keine Wetten")
        
        with st.expander("➕ Wette hinzufügen"):
            col1, col2 = st.columns(2)
            with col1:
                bet_stake = st.number_input("Einsatz €", min_value=0.0, step=10.0)
                bet_odds = st.number_input("Quote", min_value=1.0, step=0.1, value=2.0)
            with col2:
                bet_result = st.selectbox("Ergebnis", ["Offen", "Gewonnen", "Verloren"])
            
            if st.button("Speichern", use_container_width=True):
                bet_return = 0
                if bet_result == "Gewonnen":
                    bet_return = bet_stake * bet_odds
                    st.session_state.bankroll += (bet_return - bet_stake)
                elif bet_result == "Verloren":
                    st.session_state.bankroll -= bet_stake
                
                st.session_state.bet_history.append({
                    'Datum': datetime.now().strftime('%d.%m'),
                    'Einsatz': f"€{bet_stake:.0f}",
                    'Quote': bet_odds,
                    'Status': bet_result,
                    'return': bet_return,
                    'stake': bet_stake
                })
                st.success("✅ Gespeichert")
                st.rerun()
    
    with tab3:
        st.subheader("📈 Statistiken")
        
        if st.session_state.bet_history:
            df = pd.DataFrame(st.session_state.bet_history)
            
            won_bets = df[df['Status'] == 'Gewonnen']
            lost_bets = df[df['Status'] == 'Verloren']
            
            total_staked = sum([bet['stake'] for bet in st.session_state.bet_history])
            total_return = sum([bet.get('return', 0) for bet in st.session_state.bet_history])
            roi = ((total_return - total_staked) / total_staked * 100) if total_staked > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Gewinnrate", f"{len(won_bets) / len(df) * 100:.0f}%")
            col2.metric("ROI", f"{roi:.1f}%")
            col3.metric("Wetten", len(df))
            
            st.divider()
            
            if roi < -10:
                st.error("⚠️ **Kritisch:** ROI < -10%. Pause empfohlen!")
            elif roi < 0:
                st.warning("⚠️ Leicht negativ. Mehr Daten nötig.")
            elif roi < 5:
                st.info("✅ Im Rahmen. Weiter so!")
            else:
                st.success(f"🎉 Sehr gut! ROI: {roi:.1f}%")
        else:
            st.info("Noch keine Daten")

if __name__ == "__main__":
    main()
