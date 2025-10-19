"""
Fußballwetten-Analyse-App v4.0 Pro - FIXED v2
Vollständige Fehlerbehandlung mit Debug-Info
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
    FIXED: Bessere Fehlerbehandlung ohne Emojis in HTTP-Requests
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
        # Speichere Fehler für spätere Anzeige
        if 'api_errors' not in st.session_state:
            st.session_state.api_errors = []
        st.session_state.api_errors.append(f"Historical data error: {str(e)}")
        return []

def get_todays_fixtures(api_key, league_id, season):
    """
    FIXED: Mit season Parameter und verbessertem Debugging
    """
    headers = {
        'x-rapidapi-host': 'v3.football.api-sports.io',
        'x-rapidapi-key': api_key
    }
    
    today = datetime.now().strftime("%Y-%m-%d")
    url = f'https://v3.football.api-sports.io/fixtures?league={league_id}&season={season}&date={today}'
    
    # Debug: Speichere Request-Info
    if 'fixture_requests' not in st.session_state:
        st.session_state.fixture_requests = []
    st.session_state.fixture_requests.append({
        'url': url,
        'date': today,
        'league_id': league_id,
        'season': season
    })
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            # Debug: Speichere Rohdaten
            if st.session_state.get('debug_mode', False):
                if 'fixture_responses' not in st.session_state:
                    st.session_state.fixture_responses = []
                st.session_state.fixture_responses.append({
                    'league_id': league_id,
                    'status': response.status_code,
                    'fixtures_found': len(data.get('response', []))
                })
            
            if 'response' in data and len(data['response']) > 0:
                fixtures_with_odds = []
                
                for fixture in data['response']:
                    fixture_id = fixture['fixture']['id']
                    odds = get_fixture_odds(api_key, fixture_id)
                    
                    if odds:
                        fixture['odds_data'] = odds
                        fixtures_with_odds.append(fixture)
                    
                    time.sleep(0.3)  # Rate limiting
                
                return fixtures_with_odds
        
        return []
            
    except Exception as e:
        if 'fixture_errors' not in st.session_state:
            st.session_state.fixture_errors = []
        st.session_state.fixture_errors.append(f"League {league_id}: {str(e)}")
        return []

def get_fixture_odds(api_key, fixture_id):
    """Hole Quoten für ein Spiel von Bet365"""
    headers = {
        'x-rapidapi-host': 'v3.football.api-sports.io',
        'x-rapidapi-key': api_key
    }
    
    url = f'https://v3.football.api-sports.io/odds?fixture={fixture_id}&bookmaker=8'
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'response' in data and len(data['response']) > 0:
                bookmaker_data = data['response'][0]
                
                if 'bookmakers' in bookmaker_data and len(bookmaker_data['bookmakers']) > 0:
                    bets = bookmaker_data['bookmakers'][0].get('bets', [])
                    
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
                            
                            if len(odds) == 3:
                                return odds
        return None
            
    except Exception as e:
        return None

def process_fixtures_to_dataframe(fixtures):
    """Konvertiere Fixtures zu DataFrame"""
    matches = []
    
    for fixture in fixtures:
        try:
            matches.append({
                'date': datetime.strptime(fixture['fixture']['date'][:10], '%Y-%m-%d'),
                'home': fixture['teams']['home']['name'],
                'away': fixture['teams']['away']['name'],
                'score_home': fixture['goals']['home'],
                'score_away': fixture['goals']['away']
            })
        except:
            continue
    
    return pd.DataFrame(matches)

# ============================================================================
# DIXON-COLES MODELL - ERWEITERT
# ============================================================================

def calculate_team_strengths_advanced(df):
    """Berechne erweiterte Team-Stärken"""
    teams = set(df['home'].unique()).union(set(df['away'].unique()))
    
    attack = {team: 1.0 for team in teams}
    defense = {team: 1.0 for team in teams}
    home_advantages = {team: 0.3 for team in teams}
    form_factors = {team: 1.0 for team in teams}
    
    # Basis-Stärken berechnen
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
    
    # Heimvorteil berechnen
    for team in teams:
        home_games = df[df['home'] == team]
        if len(home_games) > 3:
            home_goals = home_games['score_home'].mean()
            away_goals = home_games['score_away'].mean()
            home_advantages[team] = max(0.1, min(0.5, (home_goals - away_goals) / 3))
    
    # Form-Faktoren (letzte 30 Tage)
    cutoff_date = datetime.now() - timedelta(days=30)
    recent = df[df['date'] >= cutoff_date]
    
    for team in teams:
        team_games = recent[(recent['home'] == team) | (recent['away'] == team)]
        
        if len(team_games) > 0:
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
            form_factors[team] = 0.7 + (avg_points / 3.0) * 0.6
    
    return attack, defense, home_advantages, form_factors

def calculate_h2h_factor(home_team, away_team, df):
    """Head-to-Head Faktor (letzte 2 Jahre)"""
    cutoff_date = datetime.now() - timedelta(days=730)
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
    FIXED: Erweiterte Wahrscheinlichkeitsberechnung
    """
    try:
        # WICHTIG: Teams-Check VOR allen Berechnungen
        if home_team not in attack or away_team not in attack:
            if 'missing_teams' not in st.session_state:
                st.session_state.missing_teams = []
            
            missing = []
            if home_team not in attack:
                missing.append(home_team)
            if away_team not in attack:
                missing.append(away_team)
            
            st.session_state.missing_teams.append(f"{home_team} vs {away_team}: {', '.join(missing)}")
            
            return {
                'home_win': 0.33, 
                'draw': 0.33, 
                'away_win': 0.33,
                'expected_goals_home': 1.5,
                'expected_goals_away': 1.5
            }
        
        # FIXED: Alle Werte holen
        home_attack = attack.get(home_team, 1.0)
        away_attack = attack.get(away_team, 1.0)
        home_defense = defense.get(home_team, 1.0)
        away_defense = defense.get(away_team, 1.0)
        
        # FIXED: Lambda-Werte IMMER initialisieren
        base_lambda_home = home_attack * away_defense
        base_lambda_away = away_attack * home_defense
        
        home_adv = home_advantages.get(home_team, 0.3)
        lambda_home = base_lambda_home * (1 + home_adv)
        lambda_away = base_lambda_away
        
        home_form = form_factors.get(home_team, 1.0)
        away_form = form_factors.get(away_team, 1.0)
        
        lambda_home *= home_form
        lambda_away *= away_form
        
        h2h_factor = calculate_h2h_factor(home_team, away_team, df)
        lambda_home *= h2h_factor
        
        # Werte begrenzen
        lambda_home = max(0.5, min(5.0, lambda_home))
        lambda_away = max(0.5, min(5.0, lambda_away))
        
        # Poisson-Matrix
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

def calculate_kelly_stake(prob, odd, bankroll, kelly_fraction, max_bet_percent):
    """Berechne Kelly-Einsatz"""
    if prob <= 0 or odd <= 1:
        return 0, 0
    
    kelly = (prob * odd - 1) / (odd - 1)
    kelly_stake = kelly * kelly_fraction
    
    if kelly_stake <= 0:
        return 0, 0
    
    stake = kelly_stake * bankroll
    max_stake = bankroll * (max_bet_percent / 100)
    stake = min(stake, max_stake)
    
    stake_percent = (stake / bankroll) * 100
    
    return round(stake, 2), round(stake_percent, 2)

# ============================================================================
# ANALYSE
# ============================================================================

def analyze_league(api_key, league_name, league_id):
    """
    FIXED: Hauptanalysefunktion mit vollständigem Debugging
    """
    # Reset Fehlerspeicher
    st.session_state.missing_teams = []
    st.session_state.calculation_errors = []
    st.session_state.api_errors = []
    st.session_state.fixture_requests = []
    st.session_state.fixture_responses = []
    st.session_state.fixture_errors = []
    
    season = get_current_season()
    
    st.info(f"Analysiere {league_name}...")
    
    # Historische Daten laden
    historical = get_historical_fixtures(api_key, league_id, season, days_back=150)
    
    if not historical or len(historical) < 30:
        st.warning(f"Zu wenig historische Daten für {league_name}")
        return []
    
    st.success(f"{len(historical)} historische Spiele geladen")
    
    df = process_fixtures_to_dataframe(historical)
    if df.empty:
        return []
    
    attack, defense, home_advantages, form_factors = calculate_team_strengths_advanced(df)
    
    if st.session_state.get('debug_mode', False):
        st.write(f"**Teams im Modell:** {len(attack)}")
    
    # Heutige Spiele laden
    todays_fixtures = get_todays_fixtures(api_key, league_id, season)
    
    # DEBUG: Zeige API-Request Info
    if st.session_state.get('debug_mode', False) and st.session_state.fixture_requests:
        with st.expander("🔍 API Request Details"):
            for req in st.session_state.fixture_requests:
                st.json(req)
        
        if st.session_state.fixture_responses:
            with st.expander("📊 API Response Details"):
                for resp in st.session_state.fixture_responses:
                    st.json(resp)
    
    if not todays_fixtures:
        msg = f"Keine Spiele heute in {league_name}"
        if st.session_state.get('debug_mode', False):
            msg += f" (Datum: {datetime.now().strftime('%Y-%m-%d')}, Season: {season})"
        st.info(msg)
        return []
    
    st.success(f"{len(todays_fixtures)} Spiele heute gefunden")
    
    value_bets = []
    
    for fixture in todays_fixtures:
        try:
            home_team = fixture['teams']['home']['name']
            away_team = fixture['teams']['away']['name']
            fixture_time = datetime.fromisoformat(fixture['fixture']['date'].replace('Z', '+00:00'))
            
            odds = fixture.get('odds_data')
            if not odds:
                continue
            
            # Teams-Check VORHER
            if home_team not in attack or away_team not in attack:
                continue
            
            probs = calculate_match_probabilities_advanced(
                home_team, away_team, attack, defense, 
                home_advantages, form_factors, df
            )
            
            # Prüfe ob Default-Werte (= Teams fehlen)
            if probs['home_win'] == 0.33 and probs['draw'] == 0.33 and probs['away_win'] == 0.33:
                continue
            
            markets = [
                ('home', f'Sieg {home_team}', probs['home_win'], odds.get('home')),
                ('draw', 'Unentschieden', probs['draw'], odds.get('draw')),
                ('away', f'Sieg {away_team}', probs['away_win'], odds.get('away'))
            ]
            
            for market_type, market_name, prob, odd in markets:
                if odd and prob * odd > 1.05:
                    stake, stake_percent = calculate_kelly_stake(
                        prob, odd, st.session_state.bankroll,
                        st.session_state.kelly_fraction,
                        st.session_state.max_bet_percent
                    )
                    
                    if stake >= 1:
                        value = ((prob * odd - 1) * 100)
                        expected_profit = stake * (prob * odd - 1)
                        
                        value_bets.append({
                            'liga': league_name,
                            'zeit': fixture_time.strftime('%H:%M'),
                            'spiel': f"{home_team} vs {away_team}",
                            'wette': market_name,
                            'wahrscheinlichkeit': f"{prob*100:.1f}%",
                            'quote': f"{odd:.2f}",
                            'value': value,
                            'einsatz': stake,
                            'einsatz_prozent': stake_percent,
                            'erwarteter_gewinn': expected_profit,
                            'xg_home': probs['expected_goals_home'],
                            'xg_away': probs['expected_goals_away'],
                            'form_home': form_factors.get(home_team, 1.0),
                            'form_away': form_factors.get(away_team, 1.0),
                            '_raw_stake': stake,
                            '_raw_odds': odd,
                            '_raw_prob': prob
                        })
        
        except Exception as e:
            if 'processing_errors' not in st.session_state:
                st.session_state.processing_errors = []
            st.session_state.processing_errors.append(str(e))
            continue
    
    # FIXED: Zeige gesammelte Fehler
    if st.session_state.get('debug_mode', False):
        if st.session_state.missing_teams:
            with st.expander("⚠️ Teams nicht im Modell"):
                for msg in st.session_state.missing_teams:
                    st.text(msg)
        
        if st.session_state.calculation_errors:
            with st.expander("⚠️ Berechnungsfehler"):
                for err in st.session_state.calculation_errors:
                    st.text(err)
        
        if st.session_state.fixture_errors:
            with st.expander("⚠️ Fixture-Ladefehler"):
                for err in st.session_state.fixture_errors:
                    st.text(err)
    
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
        st.info("Debug-Modus aktiviert: Detaillierte API-Requests und Fehler werden angezeigt")

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
                        if st.button("✅ Gewonnen", key=f"won_{idx}"):
                            actual_return = bet['stake'] * bet['odds']
                            update_bet_result(idx, 'won', actual_return)
                            st.success("Gespeichert!")
                            st.rerun()
                    
                    with col2:
                        if st.button("❌ Verloren", key=f"lost_{idx}"):
                            update_bet_result(idx, 'lost', 0)
                            st.success("Gespeichert!")
                            st.rerun()
                    
                    with col3:
                        if st.button("🗑️ Abbrechen", key=f"cancel_{idx}"):
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
        if not df_hist.empty:
            df_hist['date'] = pd.to_datetime(df_hist['timestamp']).dt.date
            st.dataframe(df_hist[['date', 'match', 'market', 'stake', 'odds', 'status', 'result']])
    else:
        st.info("Noch keine abgeschlossenen Wetten")

with tab1:
    st.header("📊 Value Bets Analyse")
    
    # Zeige aktuelles Datum
    st.caption(f"Suche Spiele für: {datetime.now().strftime('%Y-%m-%d (%A)')}")
    
    if not st.session_state.api_key:
        st.warning("⚠️ Bitte API-Key in Einstellungen eingeben!")
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
                    st.success(f"✅ {len(all_value_bets)} Value Bets gefunden!")
                    
                    df_bets = pd.DataFrame(all_value_bets)
                    df_bets = df_bets.sort_values('erwarteter_gewinn', ascending=False)
                    
                    for idx, bet in df_bets.iterrows():
                        market_emoji = {
                            'home': '🏠',
                            'draw': '🤝',
                            'away': '✈️'
                        }
                        
                        market_type = 'home' if 'Sieg' in bet['wette'] and bet['spiel'].split(' vs ')[0] in bet['wette'] else (
                            'draw' if 'Unent' in bet['wette'] else 'away'
                        )
                        
                        with st.container():
                            st.markdown(f"""
                            <div class="value-bet">
                                <h3>{bet['spiel']} - {bet['zeit']} Uhr</h3>
                                <p><strong>Wette:</strong> {bet['wette']} | <strong>Quote:</strong> {bet['quote']}</p>
                                <p><strong>Wahrscheinlichkeit:</strong> {bet['wahrscheinlichkeit']} | <strong>Value:</strong> {bet['value']:.1f}%</p>
                                <p><strong>Empfohlener Einsatz:</strong> {bet['einsatz']:.2f}€ ({bet['einsatz_prozent']:.1f}%)</p>
                                <p><strong>Erwarteter Gewinn:</strong> {bet['erwarteter_gewinn']:.2f}€</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if st.button("📝 Tracken", key=f"track_{idx}"):
                                bet_info = {
                                    'match': bet['spiel'],
                                    'market': bet['wette'],
                                    'odds': bet['_raw_odds'],
                                    'stake': bet['_raw_stake'],
                                    'expected_return': bet['erwarteter_gewinn']
                                }
                                add_bet_to_history(bet_info)
                                st.success("✅ Zur Historie hinzugefügt!")
                else:
                    st.info("ℹ️ Keine Value Bets gefunden")
            else:
                st.warning("⚠️ Bitte mindestens eine Liga auswählen")

st.divider()
with st.expander("ℹ️ Über diese App"):
    st.markdown("""
    ### Features v4.0 Pro - Fixed
    - ✅ **Korrekte HTTP-Header** (keine Emoji-Fehler mehr)
    - ✅ **Robuste Lambda-Berechnung** (keine UnboundLocalError mehr)
    - 📊 **Verbessertes Debugging** (API-Requests sichtbar)
    - 🎯 **Dixon-Coles Modell** mit erweiterten Statistiken
    - 💰 **Kelly-Kriterium** für optimales Bankroll-Management
    - 📈 **Wett-Tracking** mit Performance-Analyse
    
    ### Debug-Modus
    Aktiviere den Debug-Modus in den Einstellungen um zu sehen:
    - API-Request Details (URL, Parameter)
    - API-Response Status
    - Fehlende Teams
    - Berechnungsfehler
    """)
    
    st.caption("⚠️ Glücksspiel kann süchtig machen. Hilfe: www.bzga.de")
