"""
Fußballwetten-Analyse-App v4.0 Pro - FIXED
Fehlerbehandlung optimiert für Teams und HTTP Headers
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from scipy.stats import poisson
import time

st.set_page_config(
    page_title="⚽ Wetten-Analyst Pro v4.0",
    page_icon="⚽",
    layout="wide"
)

st.markdown("""
<style>
    .value-bet {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

LEAGUES = {
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League": 39,
    "🇩🇪 Bundesliga": 78,
    "🇩🇪 2. Bundesliga": 79,
    "🇪🇸 La Liga": 140,
    "🇮🇹 Serie A": 135,
    "🇫🇷 Ligue 1": 61,
}

# Session State initialisieren
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 1000.0
if 'bet_history' not in st.session_state:
    st.session_state.bet_history = []
if 'kelly_fraction' not in st.session_state:
    st.session_state.kelly_fraction = 0.25
if 'max_bet_percent' not in st.session_state:
    st.session_state.max_bet_percent = 5.0
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

# ============================================================================
# STATISTIK-TRACKING
# ============================================================================

def add_bet_to_history(bet_info):
    """Füge Wette zur Historie hinzu"""
    bet_info['timestamp'] = datetime.now().isoformat()
    bet_info['status'] = 'pending'
    st.session_state.bet_history.append(bet_info)

def update_bet_result(bet_index, result, actual_return):
    """Update Wett-Ergebnis"""
    if bet_index < len(st.session_state.bet_history):
        st.session_state.bet_history[bet_index]['status'] = 'completed'
        st.session_state.bet_history[bet_index]['result'] = result
        st.session_state.bet_history[bet_index]['actual_return'] = actual_return
        
        if result == 'won':
            st.session_state.bankroll += actual_return
        elif result == 'lost':
            st.session_state.bankroll -= st.session_state.bet_history[bet_index]['stake']

def calculate_statistics():
    """Berechne Wett-Statistiken"""
    completed = [b for b in st.session_state.bet_history if b['status'] == 'completed']
    
    if not completed:
        return None
    
    total_bets = len(completed)
    won_bets = len([b for b in completed if b['result'] == 'won'])
    
    total_staked = sum([b['stake'] for b in completed])
    total_returns = sum([b.get('actual_return', 0) for b in completed if b['result'] == 'won'])
    
    profit = total_returns - total_staked
    roi = (profit / total_staked * 100) if total_staked > 0 else 0
    win_rate = (won_bets / total_bets * 100) if total_bets > 0 else 0
    
    return {
        'total_bets': total_bets,
        'won_bets': won_bets,
        'win_rate': win_rate,
        'total_staked': total_staked,
        'total_returns': total_returns,
        'profit': profit,
        'roi': roi
    }

# ============================================================================
# API FUNKTIONEN
# ============================================================================

def get_current_season():
    now = datetime.now()
    return now.year if now.month >= 7 else now.year - 1

def test_api_connection(api_key):
    headers = {
        'x-rapidapi-host': 'v3.football.api-sports.io',
        'x-rapidapi-key': api_key
    }
    
    try:
        response = requests.get(
            'https://v3.football.api-sports.io/status',
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'response' in data:
                account = data['response'].get('account', {})
                requests_today = account.get('requests', {}).get('current', 0)
                requests_limit = account.get('requests', {}).get('limit_day', 0)
                return True, f"API OK | Requests: {requests_today}/{requests_limit}"
        return False, f"API-Fehler: Status {response.status_code}"
    except Exception as e:
        return False, f"Verbindungsfehler: {str(e)}"

def get_historical_fixtures(api_key, league_id, season, days_back=150):
    """
    FIXED: Bessere Fehlerbehandlung ohne Emojis in Ausgaben
    """
    headers = {
        'x-rapidapi-host': 'v3.football.api-sports.io',
        'x-rapidapi-key': api_key
    }
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    url = (
        f'https://v3.football.api-sports.io/fixtures'
        f'?league={league_id}'
        f'&season={season}'
        f'&from={start_date.strftime("%Y-%m-%d")}'
        f'&to={end_date.strftime("%Y-%m-%d")}'
    )
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'response' in data:
                completed = [
                    f for f in data['response']
                    if f['fixture']['status']['short'] in ['FT', 'AET', 'PEN']
                ]
                return completed
        return []
            
    except Exception as e:
        # FIXED: Speichere Fehler in Session State statt direkt auszugeben
        if 'api_errors' not in st.session_state:
            st.session_state.api_errors = []
        st.session_state.api_errors.append(f"Fehler bei historischen Daten: {str(e)}")
        return []

def get_todays_fixtures(api_key, league_id):
    headers = {
        'x-rapidapi-host': 'v3.football.api-sports.io',
        'x-rapidapi-key': api_key
    }
    
    today = datetime.now().strftime('%Y-%m-%d')
    url = f'https://v3.football.api-sports.io/fixtures?league={league_id}&date={today}'
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            return data.get('response', [])
        return []
    except:
        return []

def get_odds_for_fixture(api_key, fixture_id):
    headers = {
        'x-rapidapi-host': 'v3.football.api-sports.io',
        'x-rapidapi-key': api_key
    }
    
    url = f'https://v3.football.api-sports.io/odds?fixture={fixture_id}&bookmaker=8'
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if data.get('response') and len(data['response']) > 0:
                bookmaker = data['response'][0].get('bookmakers', [])
                if bookmaker:
                    bets = bookmaker[0].get('bets', [])
                    for bet in bets:
                        if bet.get('name') == 'Match Winner':
                            values = bet.get('values', [])
                            odds = {}
                            for v in values:
                                if v['value'] == 'Home':
                                    odds['home'] = float(v['odd'])
                                elif v['value'] == 'Draw':
                                    odds['draw'] = float(v['odd'])
                                elif v['value'] == 'Away':
                                    odds['away'] = float(v['odd'])
                            return odds
        return None
    except:
        return None

# ============================================================================
# DIXON-COLES MODELL - ERWEITERT
# ============================================================================

def calculate_team_strengths(df):
    """Berechne Team-Stärken aus historischen Daten"""
    teams = set(df['home'].unique()).union(set(df['away'].unique()))
    
    attack = {team: 1.0 for team in teams}
    defense = {team: 1.0 for team in teams}
    
    iterations = 20
    for _ in range(iterations):
        new_attack = {}
        new_defense = {}
        
        for team in teams:
            home_games = df[df['home'] == team]
            away_games = df[df['away'] == team]
            
            goals_scored = home_games['score_home'].sum() + away_games['score_away'].sum()
            goals_conceded = home_games['score_away'].sum() + away_games['score_home'].sum()
            games_played = len(home_games) + len(away_games)
            
            if games_played > 0:
                avg_scored = goals_scored / games_played
                avg_conceded = goals_conceded / games_played
                
                new_attack[team] = avg_scored / 1.5
                new_defense[team] = avg_conceded / 1.5
            else:
                new_attack[team] = 1.0
                new_defense[team] = 1.0
        
        attack = new_attack
        defense = new_defense
    
    return attack, defense

def calculate_home_advantage(df):
    """Berechne Heimvorteil für jedes Team"""
    home_advantages = {}
    teams = set(df['home'].unique())
    
    for team in teams:
        home_games = df[df['home'] == team]
        if len(home_games) > 0:
            home_goals = home_games['score_home'].mean()
            away_goals = home_games['score_away'].mean()
            home_advantages[team] = max(0.1, min(0.5, (home_goals - away_goals) / 3))
        else:
            home_advantages[team] = 0.3
    
    return home_advantages

def calculate_form_factor(team, df, days=30):
    """Berechne Form-Faktor basierend auf letzten Spielen"""
    cutoff_date = datetime.now() - timedelta(days=days)
    recent = df[df['date'] >= cutoff_date]
    
    team_games = recent[(recent['home'] == team) | (recent['away'] == team)]
    
    if len(team_games) == 0:
        return 1.0
    
    points = 0
    for _, game in team_games.iterrows():
        if game['home'] == team:
            if game['score_home'] > game['score_away']:
                points += 3
            elif game['score_home'] == game['score_away']:
                points += 1
        else:
            if game['score_away'] > game['score_home']:
                points += 3
            elif game['score_away'] == game['score_home']:
                points += 1
    
    avg_points = points / len(team_games)
    return 0.7 + (avg_points / 3.0) * 0.6

def calculate_h2h_factor(home_team, away_team, df, days=730):
    """Head-to-Head Faktor"""
    cutoff_date = datetime.now() - timedelta(days=days)
    h2h = df[
        ((df['home'] == home_team) & (df['away'] == away_team)) |
        ((df['home'] == away_team) & (df['away'] == home_team))
    ]
    h2h = h2h[h2h['date'] >= cutoff_date]
    
    if len(h2h) < 2:
        return 1.0
    
    home_wins = len(h2h[(h2h['home'] == home_team) & (h2h['score_home'] > h2h['score_away'])])
    away_wins = len(h2h[(h2h['away'] == home_team) & (h2h['score_away'] > h2h['score_home'])])
    
    total_h2h = len(h2h)
    if total_h2h > 0:
        win_rate = (home_wins + away_wins) / total_h2h
        return 0.9 + (win_rate * 0.2)
    
    return 1.0

def calculate_match_probabilities_advanced(home_team, away_team, attack, defense,
                                          home_advantages, form_factors, df):
    """
    FIXED: Erweiterte Wahrscheinlichkeitsberechnung mit verbesserter Fehlerbehandlung
    """
    try:
        # WICHTIG: Prüfe ob Teams überhaupt im Modell existieren
        if home_team not in attack or away_team not in attack:
            # FIXED: Sammle fehlende Teams ohne direkte st.warning Ausgabe
            if 'missing_teams' not in st.session_state:
                st.session_state.missing_teams = []
            
            missing = []
            if home_team not in attack:
                missing.append(home_team)
            if away_team not in attack:
                missing.append(away_team)
            
            st.session_state.missing_teams.append(f"{home_team} vs {away_team}: {', '.join(missing)}")
            
            # Gib Default-Werte zurück
            return {
                'home_win': 0.33, 
                'draw': 0.33, 
                'away_win': 0.33,
                'expected_goals_home': 1.5,
                'expected_goals_away': 1.5
            }
        
        # Hole Werte mit Fehlerbehandlung
        home_attack = attack.get(home_team, 1.0)
        away_attack = attack.get(away_team, 1.0)
        home_defense = defense.get(home_team, 1.0)
        away_defense = defense.get(away_team, 1.0)
        
        # FIXED: Lambda-Werte IMMER initialisieren BEVOR sie verwendet werden
        base_lambda_home = home_attack * away_defense
        base_lambda_away = away_attack * home_defense
        
        # Heimvorteil
        home_adv = home_advantages.get(home_team, 0.3)
        lambda_home = base_lambda_home * (1 + home_adv)
        lambda_away = base_lambda_away  # Kein Heimvorteil für Auswärtsteam
        
        # Form-Faktoren
        home_form = form_factors.get(home_team, 1.0)
        away_form = form_factors.get(away_team, 1.0)
        
        lambda_home *= home_form
        lambda_away *= away_form
        
        # Head-to-Head nur auf Heimteam anwenden
        h2h_factor = calculate_h2h_factor(home_team, away_team, df)
        lambda_home *= h2h_factor
        
        # Debug-Ausgabe NACH allen Berechnungen
        if st.session_state.get('debug_mode', False):
            if 'debug_output' not in st.session_state:
                st.session_state.debug_output = []
            st.session_state.debug_output.append({
                'match': f"{home_team} vs {away_team}",
                'home_attack': home_attack,
                'away_attack': away_attack,
                'home_defense': home_defense,
                'away_defense': away_defense,
                'lambda_home': lambda_home,
                'lambda_away': lambda_away
            })
        
        # Sicherstellen dass Lambda-Werte sinnvoll sind
        lambda_home = max(0.5, min(5.0, lambda_home))
        lambda_away = max(0.5, min(5.0, lambda_away))
        
        # Poisson-Verteilung
        max_goals = 6
        prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
        
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                prob_matrix[i, j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
        
        prob_home = np.sum(np.tril(prob_matrix, -1))
        prob_draw = np.sum(np.diag(prob_matrix))
        prob_away = np.sum(np.triu(prob_matrix, 1))
        
        total = prob_home + prob_draw + prob_away
        if total > 0:
            prob_home /= total
            prob_draw /= total
            prob_away /= total
        
        return {
            'home_win': prob_home,
            'draw': prob_draw,
            'away_win': prob_away,
            'expected_goals_home': lambda_home,
            'expected_goals_away': lambda_away
        }
        
    except Exception as e:
        # FIXED: Fehler in Session State speichern statt direkt ausgeben
        if 'calculation_errors' not in st.session_state:
            st.session_state.calculation_errors = []
        st.session_state.calculation_errors.append(f"{home_team} vs {away_team}: {str(e)}")
        
        return {
            'home_win': 0.33, 
            'draw': 0.33, 
            'away_win': 0.33,
            'expected_goals_home': 1.5,
            'expected_goals_away': 1.5
        }

# ============================================================================
# ANALYSE-FUNKTIONEN
# ============================================================================

def load_historical_data(api_key, league_id, season):
    """Lade historische Daten und konvertiere zu DataFrame"""
    fixtures = get_historical_fixtures(api_key, league_id, season)
    
    if not fixtures:
        return None
    
    data = []
    for f in fixtures:
        data.append({
            'date': datetime.strptime(f['fixture']['date'][:10], '%Y-%m-%d'),
            'home': f['teams']['home']['name'],
            'away': f['teams']['away']['name'],
            'score_home': f['goals']['home'],
            'score_away': f['goals']['away']
        })
    
    return pd.DataFrame(data)

def load_todays_fixtures_with_odds(api_key, league_id):
    """Lade heutige Spiele mit echten Quoten"""
    fixtures = get_todays_fixtures(api_key, league_id)
    
    if not fixtures:
        return []
    
    matches = []
    for f in fixtures:
        odds = get_odds_for_fixture(api_key, f['fixture']['id'])
        
        if odds and all(k in odds for k in ['home', 'draw', 'away']):
            matches.append({
                'fixture_id': f['fixture']['id'],
                'home_team': f['teams']['home']['name'],
                'away_team': f['teams']['away']['name'],
                'kickoff': f['fixture']['date'],
                'odds_home': odds['home'],
                'odds_draw': odds['draw'],
                'odds_away': odds['away']
            })
        
        time.sleep(0.5)
    
    return matches

def analyze_league(api_key, league_name, league_id):
    """
    FIXED: Hauptanalysefunktion mit verbesserter Fehlersammlung
    """
    # Reset Fehlerspeicher
    st.session_state.missing_teams = []
    st.session_state.calculation_errors = []
    st.session_state.debug_output = []
    st.session_state.api_errors = []
    
    season = get_current_season()
    
    st.info(f"Lade historische Daten für {league_name} (Saison {season})...")
    df = load_historical_data(api_key, league_id, season)
    
    if df is None or len(df) < 50:
        st.warning(f"Zu wenig Daten für {league_name}")
        return []
    
    st.success(f"{len(df)} historische Spiele geladen")
    
    attack, defense = calculate_team_strengths(df)
    home_advantages = calculate_home_advantage(df)
    
    form_factors = {}
    for team in attack.keys():
        form_factors[team] = calculate_form_factor(team, df)
    
    if st.session_state.get('debug_mode', False):
        st.write(f"**Gesamt Teams in Modell:** {len(attack)}")
    
    st.info("Lade heutige Spiele und Quoten...")
    todays_matches = load_todays_fixtures_with_odds(api_key, league_id)
    
    if not todays_matches:
        st.info(f"Keine Spiele heute in {league_name}")
        return []
    
    st.success(f"{len(todays_matches)} Spiele heute")
    
    value_bets = []
    
    for match in todays_matches:
        home_team = match['home_team']
        away_team = match['away_team']
        
        # FIXED: Prüfe VORHER ob Teams im Modell sind
        if home_team not in attack or away_team not in attack:
            continue  # Überspringe ohne Berechnung
        
        probs = calculate_match_probabilities_advanced(
            home_team, away_team, attack, defense,
            home_advantages, form_factors, df
        )
        
        odds = {
            'home': match['odds_home'],
            'draw': match['odds_draw'],
            'away': match['odds_away']
        }
        
        # Kelly-Kriterium für alle 3 Märkte
        for market, prob in [('home', probs['home_win']), 
                             ('draw', probs['draw']), 
                             ('away', probs['away_win'])]:
            
            odd = odds[market]
            implied_prob = 1 / odd
            edge = prob - implied_prob
            
            if edge > 0.05:
                kelly = (prob * odd - 1) / (odd - 1)
                kelly_stake = kelly * st.session_state.kelly_fraction
                
                if kelly_stake > 0:
                    max_stake = st.session_state.bankroll * (st.session_state.max_bet_percent / 100)
                    stake = min(kelly_stake * st.session_state.bankroll, max_stake)
                    
                    expected_return = (prob * odd * stake) - stake
                    
                    value_bets.append({
                        'league': league_name,
                        'home_team': home_team,
                        'away_team': away_team,
                        'market': market,
                        'model_prob': prob,
                        'odds': odd,
                        'implied_prob': implied_prob,
                        'edge': edge,
                        'kelly_pct': kelly * 100,
                        'stake': stake,
                        'expected_return': expected_return,
                        'kickoff': match['kickoff'],
                        'fixture_id': match['fixture_id']
                    })
    
    # FIXED: Zeige gesammelte Fehler NACH der Analyse
    if st.session_state.api_errors and st.session_state.debug_mode:
        with st.expander("API Fehler"):
            for err in st.session_state.api_errors:
                st.text(err)
    
    if st.session_state.missing_teams:
        with st.expander("Teams nicht im Modell"):
            for msg in st.session_state.missing_teams:
                st.text(msg)
    
    if st.session_state.calculation_errors and st.session_state.debug_mode:
        with st.expander("Berechnungsfehler"):
            for err in st.session_state.calculation_errors:
                st.text(err)
    
    if st.session_state.debug_output and st.session_state.debug_mode:
        with st.expander("Debug Berechnungen"):
            for debug in st.session_state.debug_output:
                st.json(debug)
    
    return value_bets

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.title("⚽ Fußball-Wetten Analyst Pro v4.0")
st.caption("Dixon-Coles Modell + Kelly-Kriterium + Bankroll-Management")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Analyse", "💰 Tracking", "📈 Statistik", "⚙️ Einstellungen"])

with tab4:
    st.header("⚙️ Einstellungen")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("API Konfiguration")
        api_key = st.text_input("API-Football Key", 
                                value=st.session_state.api_key,
                                type="password")
        
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
        
        if api_key:
            if st.button("🔍 API testen"):
                with st.spinner("Teste Verbindung..."):
                    success, message = test_api_connection(api_key)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
    
    with col2:
        st.subheader("Bankroll Management")
        bankroll = st.number_input("Startkapital (€)", 
                                   min_value=100.0,
                                   value=st.session_state.bankroll,
                                   step=100.0)
        st.session_state.bankroll = bankroll
        
        kelly_fraction = st.slider("Kelly Fraction", 
                                   min_value=0.1,
                                   max_value=1.0,
                                   value=st.session_state.kelly_fraction,
                                   step=0.05)
        st.session_state.kelly_fraction = kelly_fraction
        
        max_bet = st.slider("Max Einsatz (%)", 
                           min_value=1.0,
                           max_value=10.0,
                           value=st.session_state.max_bet_percent,
                           step=0.5)
        st.session_state.max_bet_percent = max_bet
    
    st.divider()
    
    debug = st.checkbox("🐛 Debug-Modus", value=st.session_state.debug_mode)
    st.session_state.debug_mode = debug
    
    if debug:
        st.info("Debug-Modus aktiviert: Detaillierte Ausgaben werden angezeigt")

with tab2:
    st.header("💰 Wett-Tracking")
    
    if st.session_state.bet_history:
        pending = [b for b in st.session_state.bet_history if b['status'] == 'pending']
        
        if pending:
            st.subheader("Offene Wetten")
            for idx, bet in enumerate(pending):
                with st.expander(f"{bet['match']} - {bet['market']} - {bet['stake']:.2f}€"):
                    st.write(f"**Quote:** {bet['odds']:.2f}")
                    st.write(f"**Erwarteter Gewinn:** {bet['expected_return']:.2f}€")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("Gewonnen", key=f"won_{idx}"):
                            actual_return = bet['stake'] * bet['odds']
                            update_bet_result(idx, 'won', actual_return)
                            st.success("Ergebnis gespeichert!")
                            st.rerun()
                    
                    with col2:
                        if st.button("Verloren", key=f"lost_{idx}"):
                            update_bet_result(idx, 'lost', 0)
                            st.success("Ergebnis gespeichert!")
                            st.rerun()
                    
                    with col3:
                        if st.button("Abbrechen", key=f"cancel_{idx}"):
                            st.session_state.bet_history.pop(idx)
                            st.rerun()
    else:
        st.info("Noch keine getrackten Wetten")

with tab3:
    st.header("📈 Performance-Statistik")
    
    stats = calculate_statistics()
    
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Gesamt Wetten", stats['total_bets'])
        with col2:
            st.metric("Gewinnrate", f"{stats['win_rate']:.1f}%")
        with col3:
            st.metric("Profit/Loss", f"{stats['profit']:.2f}€",
                     delta=f"{stats['roi']:.1f}%")
        with col4:
            st.metric("Aktuelles Kapital", f"{st.session_state.bankroll:.2f}€")
        
        st.divider()
        
        df_hist = pd.DataFrame(st.session_state.bet_history)
        df_hist['date'] = pd.to_datetime(df_hist['timestamp']).dt.date
        
        st.dataframe(df_hist[['date', 'match', 'market', 'stake', 'odds', 'status', 'result']])
    else:
        st.info("Noch keine abgeschlossenen Wetten")

with tab1:
    st.header("📊 Value Bets Analyse")
    
    if not st.session_state.api_key:
        st.warning("Bitte API-Key in Einstellungen eingeben!")
    else:
        selected_leagues = st.multiselect(
            "Ligen auswählen",
            options=list(LEAGUES.keys()),
            default=["🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League"]
        )
        
        if st.button("🔍 Value Bets suchen", type="primary"):
            if selected_leagues:
                all_value_bets = []
                
                for league_name in selected_leagues:
                    league_id = LEAGUES[league_name]
                    
                    with st.spinner(f"Analysiere {league_name}..."):
                        bets = analyze_league(st.session_state.api_key, league_name, league_id)
                        all_value_bets.extend(bets)
                    
                    time.sleep(1)
                
                if all_value_bets:
                    st.success(f"{len(all_value_bets)} Value Bets gefunden!")
                    
                    df_bets = pd.DataFrame(all_value_bets)
                    df_bets = df_bets.sort_values('expected_return', ascending=False)
                    
                    for idx, bet in df_bets.iterrows():
                        market_emoji = {
                            'home': '🏠',
                            'draw': '🤝',
                            'away': '✈️'
                        }
                        
                        with st.container():
                            st.markdown(f"""
                            <div class="value-bet">
                                <h3>{market_emoji[bet['market']]} {bet['home_team']} vs {bet['away_team']}</h3>
                                <p><strong>Liga:</strong> {bet['league']} | <strong>Markt:</strong> {bet['market'].upper()}</p>
                                <p><strong>Quote:</strong> {bet['odds']:.2f} | <strong>Modell-Wahrscheinlichkeit:</strong> {bet['model_prob']*100:.1f}%</p>
                                <p><strong>Edge:</strong> {bet['edge']*100:.1f}% | <strong>Kelly:</strong> {bet['kelly_pct']:.1f}%</p>
                                <p><strong>Empfohlener Einsatz:</strong> {bet['stake']:.2f}€ | <strong>Erwarteter Gewinn:</strong> {bet['expected_return']:.2f}€</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if st.button("Platziert", key=f"track_{idx}"):
                                bet_info = {
                                    'match': f"{bet['home_team']} vs {bet['away_team']}",
                                    'market': bet['market'],
                                    'odds': bet['odds'],
                                    'stake': bet['stake'],
                                    'expected_return': bet['expected_return']
                                }
                                add_bet_to_history(bet_info)
                                st.success("Getrackt!")
                else:
                    st.info("Keine Value Bets gefunden")
            else:
                st.warning("Bitte mindestens eine Liga auswählen")

st.divider()
with st.expander("ℹ️ Über diese App"):
    st.markdown("""
    ### Features v4.0
    - **Dixon-Coles Modell** mit erweiterten Statistiken
    - **Kelly-Kriterium** für optimales Bankroll-Management
    - **Wett-Tracking** mit Performance-Analyse
    - **Erweiterte Faktoren**: Form, Head-to-Head, Heimvorteil
    
    ### Wichtig
    - Nur mit echten historischen Daten von API-Football
    - Bankroll-Management beachten
    - Verantwortungsbewusst wetten
    """)
    
    st.caption("Glücksspiel kann süchtig machen. Hilfe: www.bzga.de")
