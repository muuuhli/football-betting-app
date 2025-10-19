"""
Fußballwetten-Analyse-App v4.0 Pro
Erweiterte Statistiken + Bankroll-Management + Wett-Tracking
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
        'won': won_bets,
        'lost': total_bets - won_bets,
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
                return True, f"✅ API OK | Requests: {requests_today}/{requests_limit}"
        return False, f"❌ API-Fehler: Status {response.status_code}"
    except Exception as e:
        return False, f"❌ Verbindungsfehler: {str(e)}"

def get_historical_fixtures(api_key, league_id, season, days_back=150):
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
        if st.session_state.debug_mode:
            st.error(f"Fehler bei historischen Daten: {str(e)}")
        return []

def get_todays_fixtures(api_key, league_id, season):
    headers = {
        'x-rapidapi-host': 'v3.football.api-sports.io',
        'x-rapidapi-key': api_key
    }
    
    today = datetime.now().strftime("%Y-%m-%d")
    url = f'https://v3.football.api-sports.io/fixtures?league={league_id}&season={season}&date={today}'
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'response' in data and len(data['response']) > 0:
                fixtures_with_odds = []
                
                for fixture in data['response']:
                    fixture_id = fixture['fixture']['id']
                    odds = get_fixture_odds(api_key, fixture_id)
                    
                    if odds:
                        fixture['odds_data'] = odds
                        fixtures_with_odds.append(fixture)
                    
                    time.sleep(0.3)
                
                return fixtures_with_odds
        return []
            
    except Exception as e:
        if st.session_state.debug_mode:
            st.error(f"Fehler bei heutigen Spielen: {str(e)}")
        return []

def get_fixture_odds(api_key, fixture_id):
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
    matches = []
    
    for fixture in fixtures:
        try:
            match_data = {
                'date': fixture['fixture']['date'],
                'home_team': fixture['teams']['home']['name'],
                'away_team': fixture['teams']['away']['name'],
                'home_goals': fixture['goals']['home'],
                'away_goals': fixture['goals']['away'],
                'status': fixture['fixture']['status']['short']
            }
            matches.append(match_data)
        except Exception as e:
            continue
    
    if matches:
        df = pd.DataFrame(matches)
        # Konvertiere zu timezone-aware datetime
        df['date'] = pd.to_datetime(df['date'], utc=True)
        return df
    return pd.DataFrame()

# ============================================================================
# ERWEITERTE STATISTIK-FUNKTIONEN
# ============================================================================

def calculate_time_weights(df):
    """Zeitgewichtung - neuere Spiele wichtiger"""
    # Verwende timezone-aware datetime
    now = pd.Timestamp.now(tz='UTC')
    days_old = (now - df['date']).dt.total_seconds() / 86400
    weights = np.exp(-days_old / 45)  # 45 Tage Halbwertszeit
    return weights

def calculate_form_factor(team, df, last_n=5):
    """Form-Faktor basierend auf letzten N Spielen"""
    team_matches = df[
        (df['home_team'] == team) | (df['away_team'] == team)
    ].sort_values('date', ascending=False).head(last_n)
    
    if len(team_matches) == 0:
        return 1.0
    
    points = 0
    for _, match in team_matches.iterrows():
        if match['home_team'] == team:
            if match['home_goals'] > match['away_goals']:
                points += 3
            elif match['home_goals'] == match['away_goals']:
                points += 1
        else:
            if match['away_goals'] > match['home_goals']:
                points += 3
            elif match['home_goals'] == match['away_goals']:
                points += 1
    
    max_points = last_n * 3
    form = 0.5 + (points / max_points)
    return form

def calculate_individual_home_advantage(team, df):
    """Individueller Heimvorteil pro Team"""
    home_matches = df[df['home_team'] == team]
    away_matches = df[df['away_team'] == team]
    
    if len(home_matches) == 0 or len(away_matches) == 0:
        return 0.3
    
    home_ppg = 0
    for _, match in home_matches.iterrows():
        if match['home_goals'] > match['away_goals']:
            home_ppg += 3
        elif match['home_goals'] == match['away_goals']:
            home_ppg += 1
    home_ppg /= len(home_matches)
    
    away_ppg = 0
    for _, match in away_matches.iterrows():
        if match['away_goals'] > match['home_goals']:
            away_ppg += 3
        elif match['home_goals'] == match['away_goals']:
            away_ppg += 1
    away_ppg /= len(away_matches)
    
    advantage = (home_ppg - away_ppg) / 3
    return max(0.0, min(0.6, 0.3 + advantage))

def calculate_h2h_factor(home_team, away_team, df, min_matches=3):
    """Head-to-Head Faktor"""
    h2h = df[
        ((df['home_team'] == home_team) & (df['away_team'] == away_team))
    ]
    
    if len(h2h) < min_matches:
        return 1.0
    
    home_wins = len(h2h[h2h['home_goals'] > h2h['away_goals']])
    total = len(h2h)
    win_rate = home_wins / total
    
    factor = 0.8 + (win_rate * 0.8)
    return factor

def calculate_team_strengths_advanced(df):
    """Erweiterte Team-Stärken Berechnung"""
    teams = pd.concat([df['home_team'], df['away_team']]).unique()
    
    weights = calculate_time_weights(df)
    
    attack = {}
    defense = {}
    home_advantages = {}
    form_factors = {}
    
    for team in teams:
        # ANGRIFF: Durchschnitt aller geschossenen Tore (Heim + Auswärts)
        all_goals = []
        all_weights = []
        
        home_games = df[df['home_team'] == team]
        for idx in home_games.index:
            all_goals.append(home_games.loc[idx, 'home_goals'])
            all_weights.append(weights[idx])
        
        away_games = df[df['away_team'] == team]
        for idx in away_games.index:
            all_goals.append(away_games.loc[idx, 'away_goals'])
            all_weights.append(weights[idx])
        
        if len(all_goals) > 0:
            attack[team] = np.average(all_goals, weights=all_weights)
        else:
            attack[team] = 1.0
        
        # VERTEIDIGUNG: Durchschnitt aller kassierten Tore (Heim + Auswärts)
        all_conceded = []
        all_weights_def = []
        
        for idx in home_games.index:
            all_conceded.append(home_games.loc[idx, 'away_goals'])
            all_weights_def.append(weights[idx])
        
        for idx in away_games.index:
            all_conceded.append(away_games.loc[idx, 'home_goals'])
            all_weights_def.append(weights[idx])
        
        if len(all_conceded) > 0:
            avg_conceded = np.average(all_conceded, weights=all_weights_def)
            defense[team] = 1.0 / (avg_conceded + 0.1)
        else:
            defense[team] = 1.0
        
        # HEIMVORTEIL & FORM
        home_advantages[team] = calculate_individual_home_advantage(team, df)
        form_factors[team] = calculate_form_factor(team, df, last_n=5)
    
    return attack, defense, home_advantages, form_factors
    
    return attack, defense, home_advantages, form_factors

def calculate_match_probabilities_advanced(home_team, away_team, attack, defense, 
                                          home_advantages, form_factors, df):
    """Erweiterte Wahrscheinlichkeitsberechnung"""
    try:
        # WICHTIG: Prüfe ob Teams überhaupt im Modell existieren
        if home_team not in attack or away_team not in attack:
            if st.session_state.get('debug_mode', False):
                missing = []
                if home_team not in attack:
                    missing.append(home_team)
                if away_team not in attack:
                    missing.append(away_team)
                st.warning(f"⚠️ Teams nicht in historischen Daten: {', '.join(missing)}")
            
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
        
        # Debug-Ausgabe wenn im Debug-Modus
        if st.session_state.get('debug_mode', False):
            st.write(f"**{home_team} vs {away_team}**")
            st.write(f"Attack: {home_attack:.2f} vs {away_attack:.2f}")
            st.write(f"Defense: {home_defense:.2f} vs {away_defense:.2f}")
        
        base_lambda_home = home_attack * away_defense
        base_lambda_away = away_attack * home_defense
        
        home_adv = home_advantages.get(home_team, 0.3)
        lambda_home = base_lambda_home * (1 + home_adv)
        
        home_form = form_factors.get(home_team, 1.0)
        away_form = form_factors.get(away_team, 1.0)
        
        lambda_home *= home_form
        lambda_away *= away_form
        
        h2h_factor = calculate_h2h_factor(home_team, away_team, df)
        lambda_home *= h2h_factor
        
        # Sicherstellen dass Lambda-Werte sinnvoll sind
        if lambda_home < 0.1:
            lambda_home = 0.5
        if lambda_away < 0.1:
            lambda_away = 0.5
        if lambda_home > 5:
            lambda_home = 5
        if lambda_away > 5:
            lambda_away = 5
        
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
        if st.session_state.get('debug_mode', False):
            st.error(f"Fehler bei {home_team} vs {away_team}: {str(e)}")
        
        return {
            'home_win': 0.33, 
            'draw': 0.33, 
            'away_win': 0.33,
            'expected_goals_home': 1.5,
            'expected_goals_away': 1.5
        }

def calculate_kelly_stake(probability, odds, bankroll, kelly_fraction, max_percent):
    """Berechne optimalen Einsatz mit Kelly-Kriterium"""
    if odds <= 1.01:
        return 0, 0
    
    edge = (probability * odds) - 1
    if edge <= 0:
        return 0, 0
    
    kelly = (probability * odds - 1) / (odds - 1)
    conservative_kelly = kelly * kelly_fraction
    stake_percent = conservative_kelly * 100
    stake_percent = min(stake_percent, max_percent)
    stake = (stake_percent / 100) * bankroll
    
    return round(stake, 2), round(stake_percent, 2)

# ============================================================================
# ANALYSE
# ============================================================================

def analyze_league(api_key, league_name, league_id):
    """Analysiere Liga mit erweiterten Statistiken"""
    season = get_current_season()
    
    st.info(f"🔍 Analysiere {league_name}...")
    
    with st.spinner("Lade historische Daten..."):
        historical = get_historical_fixtures(api_key, league_id, season, days_back=150)
    
    if not historical or len(historical) < 30:
        st.warning(f"⚠️ Zu wenig Daten für {league_name}")
        return []
    
    st.success(f"✅ {len(historical)} historische Spiele")
    
    df = process_fixtures_to_dataframe(historical)
    if df.empty:
        return []
    
    attack, defense, home_advantages, form_factors = calculate_team_strengths_advanced(df)
    
    # Debug: Zeige berechnete Team-Stärken
    if st.session_state.get('debug_mode', False):
        st.subheader("🔬 Berechnete Team-Stärken")
        debug_data = []
        for team in list(attack.keys())[:10]:  # Zeige erste 10 Teams
            debug_data.append({
                'Team': team,
                'Angriff': f"{attack[team]:.2f}",
                'Verteidigung': f"{defense[team]:.2f}",
                'Heimvorteil': f"{home_advantages[team]:.2f}",
                'Form': f"{form_factors[team]:.2f}"
            })
        st.dataframe(pd.DataFrame(debug_data), use_container_width=True)
        st.write(f"**Gesamt Teams in Modell:** {len(attack)}")
    
    with st.spinner("Lade heutige Spiele..."):
        todays_fixtures = get_todays_fixtures(api_key, league_id, season)
    
    if not todays_fixtures:
        st.info(f"ℹ️ Keine Spiele heute")
        return []
    
    st.success(f"✅ {len(todays_fixtures)} Spiele heute")
    
    value_bets = []
    
    for fixture in todays_fixtures:
        try:
            home_team = fixture['teams']['home']['name']
            away_team = fixture['teams']['away']['name']
            fixture_time = datetime.fromisoformat(fixture['fixture']['date'].replace('Z', '+00:00'))
            
            odds = fixture.get('odds_data')
            if not odds:
                continue
            
            probs = calculate_match_probabilities_advanced(
                home_team, away_team, attack, defense, 
                home_advantages, form_factors, df
            )
            
            # Prüfe ob Modell valide Werte liefert
            if probs['home_win'] == 0.33 and probs['draw'] == 0.33 and probs['away_win'] == 0.33:
                if st.session_state.get('debug_mode', False):
                    st.warning(f"⚠️ {home_team} vs {away_team}: Teams nicht in historischen Daten gefunden - übersprungen")
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
            if st.session_state.debug_mode:
                st.write(f"⚠️ Fehler: {str(e)}")
            continue
    
    return value_bets

def run_full_analysis(api_key, selected_leagues):
    all_value_bets = []
    
    progress_bar = st.progress(0)
    total_leagues = len(selected_leagues)
    
    for idx, (league_name, league_id) in enumerate(selected_leagues.items()):
        progress_bar.progress((idx + 1) / total_leagues)
        
        with st.expander(f"📊 {league_name}", expanded=False):
            value_bets = analyze_league(api_key, league_name, league_id)
            all_value_bets.extend(value_bets)
        
        time.sleep(0.5)
    
    progress_bar.empty()
    return all_value_bets

# ============================================================================
# UI
# ============================================================================

def show_statistics_tab():
    st.header("📊 Wett-Statistiken")
    
    stats = calculate_statistics()
    
    if stats is None:
        st.info("📝 Noch keine abgeschlossenen Wetten")
        st.write("Platziere Wetten über den Analyse-Tab und tracke sie hier!")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{stats['total_bets']}</h3>
            <p>Wetten gesamt</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{stats['win_rate']:.1f}%</h3>
            <p>Gewinnrate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        profit_color = "28a745" if stats['profit'] >= 0 else "dc3545"
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #{profit_color} 0%, #555 100%);">
            <h3>€{stats['profit']:.2f}</h3>
            <p>Gewinn/Verlust</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{stats['roi']:.1f}%</h3>
            <p>ROI</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    if st.session_state.bet_history:
        st.subheader("📋 Wett-Historie")
        
        df_history = pd.DataFrame(st.session_state.bet_history)
        st.dataframe(df_history, use_container_width=True, hide_index=True)
        
        csv = df_history.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Historie exportieren",
            csv,
            f"wett_historie_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )
        
        st.divider()
        st.subheader("✏️ Ergebnis nachtragen")
        
        pending_bets = [b for b in st.session_state.bet_history if b['status'] == 'pending']
        
        if pending_bets:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                bet_options = [f"{i}: {b.get('match', 'Unbekannt')}" for i, b in enumerate(pending_bets)]
                selected_bet = st.selectbox("Wette auswählen", bet_options)
                bet_idx = int(selected_bet.split(':')[0])
            
            with col2:
                result = st.selectbox("Ergebnis", ["won", "lost", "void"])
            
            with col3:
                if result == 'won':
                    actual_return = st.number_input("Gewinn (€)", min_value=0.0, value=pending_bets[bet_idx]['stake'] * pending_bets[bet_idx]['odds'])
                else:
                    actual_return = 0.0
            
            if st.button("💾 Ergebnis speichern"):
                original_idx = st.session_state.bet_history.index(pending_bets[bet_idx])
                update_bet_result(original_idx, result, actual_return)
                st.success("✅ Ergebnis gespeichert!")
                st.rerun()
        else:
            st.info("Keine offenen Wetten")

def main():
    st.title("⚽ Wetten-Analyst Pro v4.0")
    st.caption(f"Erweiterte Statistiken + Bankroll | {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    
    tab1, tab2, tab3, tab4 = st.tabs(["⚙️ Setup", "🎯 Analyse", "📊 Statistiken", "❓ Info"])
    
    with tab1:
        st.header("⚙️ Einstellungen")
        
        api_key = st.text_input(
            "API-Football Key",
            type="password",
            value=st.session_state.api_key
        )
        st.session_state.api_key = api_key
        
        if api_key:
            if st.button("🔍 API Testen"):
                success, message = test_api_connection(api_key)
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        st.divider()
        
        st.subheader("💰 Bankroll Management")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.bankroll = st.number_input(
                "Aktuelle Bankroll (€)",
                min_value=10.0,
                max_value=1000000.0,
                value=st.session_state.bankroll,
                step=10.0
            )
        
        with col2:
            st.metric("Aktuelle Bankroll", f"€{st.session_state.bankroll:.2f}")
        
        col3, col4 = st.columns(2)
        with col3:
            st.session_state.kelly_fraction = st.slider(
                "Kelly-Fraktion",
                min_value=0.1,
                max_value=0.5,
                value=st.session_state.kelly_fraction,
                step=0.05,
                help="25% = sehr konservativ, 50% = aggressiver"
            )
        
        with col4:
            st.session_state.max_bet_percent = st.slider(
                "Max. Einsatz pro Wette (%)",
                min_value=1.0,
                max_value=10.0,
                value=st.session_state.max_bet_percent,
                step=0.5
            )
        
        st.divider()
        
        st.subheader("🏆 Ligen")
        selected_leagues = {}
        
        col1, col2 = st.columns(2)
        with col1:
            for i, (name, id) in enumerate(list(LEAGUES.items())[:3]):
                if st.checkbox(name, value=True, key=f"league_{i}"):
                    selected_leagues[name] = id
        
        with col2:
            for i, (name, id) in enumerate(list(LEAGUES.items())[3:], start=3):
                if st.checkbox(name, value=True, key=f"league_{i}"):
                    selected_leagues[name] = id
        
        st.divider()
        st.session_state.debug_mode = st.checkbox("🐛 Debug-Modus", value=st.session_state.debug_mode)
    
    with tab2:
        st.header("🎯 Heute's Value Bets")
        
        if not st.session_state.api_key:
            st.warning("⚠️ Bitte API-Key eingeben!")
            return
        
        if st.button("🚀 Analyse Starten", type="primary"):
            if not selected_leagues:
                st.error("❌ Bitte Ligen auswählen!")
                return
            
            st.info(f"🔄 Analysiere {len(selected_leagues)} Ligen...")
            
            value_bets = run_full_analysis(st.session_state.api_key, selected_leagues)
            
            st.divider()
            
            if value_bets:
                st.success(f"✅ {len(value_bets)} Value Bets gefunden!")
                
                df_bets = pd.DataFrame(value_bets)
                df_bets = df_bets.sort_values('value', ascending=False)
                
                display_df = df_bets.copy()
                display_df['value'] = display_df['value'].apply(lambda x: f"{x:.1f}%")
                display_df['einsatz'] = display_df['einsatz'].apply(lambda x: f"€{x:.2f}")
                display_df['einsatz_prozent'] = display_df['einsatz_prozent'].apply(lambda x: f"{x:.2f}%")
                display_df['erwarteter_gewinn'] = display_df['erwarteter_gewinn'].apply(lambda x: f"€{x:.2f}")
                display_df['xg_home'] = display_df['xg_home'].apply(lambda x: f"{x:.2f}")
                display_df['xg_away'] = display_df['xg_away'].apply(lambda x: f"{x:.2f}")
                display_df['form_home'] = display_df['form_home'].apply(lambda x: f"{x:.2f}")
                display_df['form_away'] = display_df['form_away'].apply(lambda x: f"{x:.2f}")
                
                display_cols = ['liga', 'zeit', 'spiel', 'wette', 'wahrscheinlichkeit', 
                               'quote', 'value', 'einsatz', 'einsatz_prozent', 
                               'erwarteter_gewinn', 'xg_home', 'xg_away', 'form_home', 'form_away']
                
                st.dataframe(
                    display_df[display_cols],
                    use_container_width=True,
                    hide_index=True
                )
                
                st.download_button(
                    "📥 Als CSV",
                    df_bets[display_cols].to_csv(index=False).encode('utf-8'),
                    f"value_bets_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv"
                )
                
                st.divider()
                st.subheader("🏆 Top 3 Empfehlungen")
                
                for idx, row in df_bets.head(3).iterrows():
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="value-bet">
                            <h4>🎯 {row['spiel']} ({row['zeit']} Uhr)</h4>
                            <p><strong>Wette:</strong> {row['wette']} @ {row['quote']}</p>
                            <p><strong>💰 Empfohlener Einsatz:</strong> €{row['einsatz']:.2f} ({row['einsatz_prozent']:.2f}% der Bankroll)</p>
                            <p><strong>📈 Value:</strong> {row['value']:.1f}% | <strong>Erwarteter Gewinn:</strong> €{row['erwarteter_gewinn']:.2f}</p>
                            <p><strong>⚽ xG:</strong> {row['xg_home']:.2f} - {row['xg_away']:.2f} | <strong>Form:</strong> {row['form_home']:.2f} vs {row['form_away']:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        if st.button("✅ Platziert", key=f"track_{idx}"):
                            bet_info = {
                                'match': row['spiel'],
                                'bet_type': row['wette'],
                                'odds': row['_raw_odds'],
                                'stake': row['_raw_stake'],
                                'probability': row['_raw_prob']
                            }
                            add_bet_to_history(bet_info)
                            st.success("✅ Getrackt!")
                            st.rerun()
            else:
                st.info("ℹ️ Keine Value Bets heute")
    
    with tab3:
        show_statistics_tab()
    
    with tab4:
        st.header("❓ Anleitung v4.0")
        
        st.markdown("""
        ### 🆕 Neu in Version 4.0
        
        **Erweiterte Statistiken:**
        - ⏰ Zeitgewichtung (neuere Spiele wichtiger)
        - 📈 Form-Faktor (letzte 5 Spiele)
        - 🏠 Individueller Heimvorteil
        - ⚔️ Head-to-Head Historie
        - ⚽ Expected Goals (xG)
        
        **Bankroll-Management:**
        - 💰 Automatische Einsatz-Berechnung
        - 📊 Kelly-Kriterium
        - 🛡️ Max-Einsatz-Limit
        
        **Statistik-Tracking:**
        - 📈 Gewinnrate & ROI
        - 📋 Wett-Historie
        - 💾 CSV-Export
        
        ### 🎯 Verwendete Faktoren
        
        | Faktor | Gewichtung | Beschreibung |
        |--------|-----------|--------------|
        | Zeitgewichtung | 30% | Neuere Spiele zählen mehr |
        | Form | 25% | Letzte 5 Spiele |
        | Heimvorteil | 20% | Team-spezifisch |
        | H2H | 15% | Direkte Duelle |
        | xG | 10% | Erwartete Tore |
        
        ### 💡 Nutzungstipps
        
        1. **Start konservativ:** Kelly 25%, Max 5%
        2. **Tracke alle Wetten** für Statistiken
        3. **Mindestens 100 Wetten** für Aussagekraft
        4. **Bankroll anpassen** nach jedem Monat
        
        ### ⚠️ Wichtig
        
        - Value Betting = Langzeitstrategie
        - Keine Garantie für Gewinne
        - Erwarteter ROI: 5-8% langfristig
        - Glücksspiel kann süchtig machen
        """)
        
        st.divider()
        st.caption("⚠️ Glücksspiel kann süchtig machen. Hilfe: www.bzga.de")

if __name__ == "__main__":
    main()
