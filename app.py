"""
Fußballwetten-Analyse-App v4.1 - MIT BACKTEST
Validiere die Model-Performance mit historischen Daten
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from scipy.stats import poisson
import time

st.set_page_config(
    page_title="⚽ Wetten-Analyst Pro v4.1",
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
    .backtest-loss {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
    }
    .backtest-win {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
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
# API FUNKTIONEN (gekürzt - siehe vorherige Version für vollständigen Code)
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
        return []

def get_fixture_odds_historical(api_key, fixture_id):
    """Hole historische Quoten für Backtest"""
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
            matches.append({
                'fixture_id': fixture['fixture']['id'],
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
# DIXON-COLES MODELL
# ============================================================================

def calculate_team_strengths_advanced(df):
    teams = set(df['home'].unique()).union(set(df['away'].unique()))
    
    attack = {team: 1.0 for team in teams}
    defense = {team: 1.0 for team in teams}
    home_advantages = {team: 0.3 for team in teams}
    form_factors = {team: 1.0 for team in teams}
    
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
    
    for team in teams:
        home_games = df[df['home'] == team]
        if len(home_games) > 3:
            home_goals = home_games['score_home'].mean()
            away_goals = home_games['score_away'].mean()
            home_advantages[team] = max(0.1, min(0.5, (home_goals - away_goals) / 3))
    
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

def calculate_match_probabilities_advanced(home_team, away_team, attack, defense,
                                          home_advantages, form_factors, df):
    try:
        if home_team not in attack or away_team not in attack:
            return {
                'home_win': 0.33, 
                'draw': 0.33, 
                'away_win': 0.33,
                'expected_goals_home': 1.5,
                'expected_goals_away': 1.5
            }
        
        home_attack = attack.get(home_team, 1.0)
        away_attack = attack.get(away_team, 1.0)
        home_defense = defense.get(home_team, 1.0)
        away_defense = defense.get(away_team, 1.0)
        
        base_lambda_home = home_attack * away_defense
        base_lambda_away = away_attack * home_defense
        
        home_adv = home_advantages.get(home_team, 0.3)
        lambda_home = base_lambda_home * (1 + home_adv)
        lambda_away = base_lambda_away
        
        home_form = form_factors.get(home_team, 1.0)
        away_form = form_factors.get(away_team, 1.0)
        
        lambda_home *= home_form
        lambda_away *= away_form
        
        lambda_home = max(0.5, min(5.0, lambda_home))
        lambda_away = max(0.5, min(5.0, lambda_away))
        
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
        return {
            'home_win': 0.33, 
            'draw': 0.33, 
            'away_win': 0.33,
            'expected_goals_home': 1.5,
            'expected_goals_away': 1.5
        }

def calculate_kelly_stake(prob, odd, bankroll, kelly_fraction, max_bet_percent):
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
# BACKTEST FUNKTION - NEU!
# ============================================================================

def run_backtest(api_key, league_id, league_name, days_back=60, test_days=14):
    """
    Backteste das Modell mit historischen Daten
    
    days_back: Wie viele Tage für Training
    test_days: Wie viele Tage für Testing
    """
    season = get_current_season()
    
    st.info(f"🔬 Starte Backtest für {league_name}...")
    
    # Lade ALLE historischen Daten
    with st.spinner("Lade historische Daten..."):
        all_fixtures = get_historical_fixtures(api_key, league_id, season, days_back=days_back + test_days)
    
    if len(all_fixtures) < 50:
        st.error("Zu wenig Daten für Backtest")
        return None
    
    df_all = process_fixtures_to_dataframe(all_fixtures)
    
    # Split: Training vs Test
    cutoff_date = datetime.now() - timedelta(days=test_days)
    df_train = df_all[df_all['date'] < cutoff_date]
    df_test = df_all[df_all['date'] >= cutoff_date]
    
    st.write(f"📊 Training: {len(df_train)} Spiele | Test: {len(df_test)} Spiele")
    
    if len(df_test) < 5:
        st.warning("Zu wenig Test-Daten")
        return None
    
    # Trainiere Modell auf alten Daten
    attack, defense, home_advantages, form_factors = calculate_team_strengths_advanced(df_train)
    
    # Teste auf neuen Daten
    backtest_results = []
    bankroll = st.session_state.bankroll
    
    progress_bar = st.progress(0)
    
    for idx, (_, fixture) in enumerate(df_test.iterrows()):
        progress_bar.progress((idx + 1) / len(df_test))
        
        home_team = fixture['home']
        away_team = fixture['away']
        actual_home = fixture['score_home']
        actual_away = fixture['score_away']
        
        # Hole historische Quoten (wenn verfügbar)
        odds = None
        if 'fixture_id' in fixture:
            odds = get_fixture_odds_historical(api_key, fixture['fixture_id'])
            time.sleep(0.3)  # Rate limiting
        
        if not odds:
            continue
        
        # Berechne Vorhersage
        probs = calculate_match_probabilities_advanced(
            home_team, away_team, attack, defense,
            home_advantages, form_factors, df_train
        )
        
        if probs['home_win'] == 0.33:  # Skip wenn Default-Werte
            continue
        
        # Bestimme tatsächliches Ergebnis
        if actual_home > actual_away:
            actual_result = 'home'
        elif actual_home < actual_away:
            actual_result = 'away'
        else:
            actual_result = 'draw'
        
        # Prüfe alle 3 Märkte
        markets = [
            ('home', probs['home_win'], odds.get('home')),
            ('draw', probs['draw'], odds.get('draw')),
            ('away', probs['away_win'], odds.get('away'))
        ]
        
        for market, prob, odd in markets:
            if odd and prob * odd > 1.05:  # Value Bet Schwelle
                stake, _ = calculate_kelly_stake(
                    prob, odd, bankroll,
                    st.session_state.kelly_fraction,
                    st.session_state.max_bet_percent
                )
                
                if stake >= 1:
                    won = (market == actual_result)
                    profit = (stake * odd - stake) if won else -stake
                    bankroll += profit
                    
                    backtest_results.append({
                        'datum': fixture['date'].strftime('%Y-%m-%d'),
                        'spiel': f"{home_team} vs {away_team}",
                        'markt': market,
                        'quote': odd,
                        'wahrscheinlichkeit': prob,
                        'einsatz': stake,
                        'ergebnis': f"{actual_home}:{actual_away}",
                        'gewonnen': won,
                        'profit': profit,
                        'bankroll': bankroll
                    })
    
    progress_bar.empty()
    
    if not backtest_results:
        st.warning("Keine Value Bets im Test-Zeitraum gefunden")
        return None
    
    return pd.DataFrame(backtest_results)

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.title("⚽ Fußball-Wetten Analyst Pro v4.1")
st.caption("Dixon-Coles Modell + Kelly-Kriterium + BACKTEST")

tab1, tab2, tab3 = st.tabs(["🔬 Backtest", "📊 Live Analyse", "⚙️ Einstellungen"])

with tab3:
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

with tab1:
    st.header("🔬 Model Backtest")
    
    st.markdown("""
    **Teste wie das Modell in der Vergangenheit performt hätte:**
    - Trainiere auf älteren Daten
    - Teste auf neueren Spielen
    - Simuliere echte Wetten mit Kelly-Kriterium
    """)
    
    if not st.session_state.api_key:
        st.warning("⚠️ Bitte API-Key in Einstellungen eingeben!")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            backtest_league = st.selectbox(
                "Liga für Backtest",
                options=list(LEAGUES.keys())
            )
        
        with col2:
            training_days = st.slider("Training Tage", 30, 120, 60)
        
        with col3:
            test_days = st.slider("Test Tage", 7, 30, 14)
        
        if st.button("🚀 Backtest starten", type="primary"):
            league_id = LEAGUES[backtest_league]
            
            results = run_backtest(
                st.session_state.api_key,
                league_id,
                backtest_league,
                days_back=training_days,
                test_days=test_days
            )
            
            if results is not None:
                st.success("✅ Backtest abgeschlossen!")
                
                # Zusammenfassung
                total_bets = len(results)
                won_bets = results['gewonnen'].sum()
                win_rate = (won_bets / total_bets * 100) if total_bets > 0 else 0
                
                total_staked = results['einsatz'].sum()
                total_profit = results['profit'].sum()
                roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
                
                final_bankroll = results['bankroll'].iloc[-1]
                bankroll_change = ((final_bankroll - st.session_state.bankroll) / st.session_state.bankroll) * 100
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Wetten", total_bets)
                with col2:
                    st.metric("Gewinnrate", f"{win_rate:.1f}%")
                with col3:
                    st.metric("ROI", f"{roi:.1f}%", 
                             delta=f"{total_profit:.2f}€")
                with col4:
                    st.metric("Bankroll", f"{final_bankroll:.0f}€",
                             delta=f"{bankroll_change:+.1f}%")
                
                # Chart: Bankroll-Entwicklung
                st.subheader("📈 Bankroll-Entwicklung")
                st.line_chart(results.set_index(results.index)['bankroll'])
                
                # Detaillierte Wetten
                st.subheader("📋 Detaillierte Wetten")
                
                # Farbcodierung
                display_df = results.copy()
                
                st.dataframe(
                    display_df[[
                        'datum', 'spiel', 'markt', 'quote', 
                        'wahrscheinlichkeit', 'einsatz', 'ergebnis', 
                        'gewonnen', 'profit', 'bankroll'
                    ]].style.applymap(
                        lambda x: 'background-color: #d4edda' if x == True else (
                            'background-color: #f8d7da' if x == False else ''
                        ),
                        subset=['gewonnen']
                    ),
                    use_container_width=True
                )
                
                # Markt-Analyse
                st.subheader("🎯 Performance nach Markt")
                market_stats = results.groupby('markt').agg({
                    'gewonnen': ['sum', 'count'],
                    'profit': 'sum',
                    'einsatz': 'sum'
                }).round(2)
                
                market_stats.columns = ['Gewonnen', 'Gesamt', 'Profit', 'Einsatz']
                market_stats['Gewinnrate %'] = (market_stats['Gewonnen'] / market_stats['Gesamt'] * 100).round(1)
                market_stats['ROI %'] = (market_stats['Profit'] / market_stats['Einsatz'] * 100).round(1)
                
                st.dataframe(market_stats, use_container_width=True)
                
                # Warnung bei negativer Performance
                if total_profit < 0:
                    st.error(f"""
                    ⚠️ **WARNUNG: Negatives Ergebnis!**
                    
                    Das Modell hätte in den letzten {test_days} Tagen **{total_profit:.2f}€ Verlust** gemacht.
                    
                    **Mögliche Gründe:**
                    - Underdog-Bias (zu optimistische Einschätzung von Außenseitern)
                    - Unentschieden werden überbewertet
                    - Modell zu simpel (keine Verletzungen, Taktik, Motivation)
                    - Value-Schwelle zu niedrig (1.05x = 5% Edge ist zu aggressiv)
                    
                    **Empfehlung:** Erhöhe die Value-Schwelle auf 1.10 oder höher!
                    """)
                else:
                    st.success(f"""
                    ✅ **Positives Ergebnis!**
                    
                    Das Modell hätte {total_profit:.2f}€ Gewinn gemacht ({roi:.1f}% ROI).
                    """)

with tab2:
    st.header("📊 Live Analyse")
    st.info("Diese Funktion wurde gekürzt. Siehe app_fixed_v2.py für vollständige Live-Analyse")
    
    st.markdown("""
    **Wichtig:** Nutze erst den **Backtest** um zu validieren ob das Modell profitabel ist!
    
    Wenn der Backtest negativ ist, solltest du:
    1. Die Value-Schwelle erhöhen (z.B. auf 1.10 statt 1.05)
    2. Nur Favoriten-Wetten nehmen (ignoriere Underdogs und Draws)
    3. Weitere Faktoren hinzufügen (Verletzungen, Form, etc.)
    """)

st.divider()
with st.expander("ℹ️ Über diese App v4.1"):
    st.markdown("""
    ### NEU in v4.1: Backtest-Funktion 🔬
    
    **Warum Backtest wichtig ist:**
    - Zeigt reale Performance mit historischen Daten
    - Deckt Underdog-Bias und Unentschieden-Probleme auf
    - Hilft Value-Schwelle zu optimieren
    - Verhindert Verluste durch unrealistische Modelle
    
    **Typische Probleme einfacher Dixon-Coles Modelle:**
    - ❌ Überschätzt Underdogs
    - ❌ Überschätzt Unentschieden
    - ❌ Ignoriert Verletzungen, Taktik, Motivation
    - ❌ Zu simpel für moderne Fußball-Analysen
    
    **Was fehlt diesem Modell:**
    - Verletzte Spieler
    - Trainer-Wechsel
    - Aktuelle Form (letzte 5 Spiele)
    - Expected Goals (xG)
    - Müdigkeit / Fixture Congestion
    - Psychologische Faktoren
    
    **Empfehlung:**
    Nutze Backtest um zu sehen ob das Modell profitabel ist. 
    Wenn nicht → erhöhe Value-Schwelle oder ergänze das Modell!
    """)
    
    st.caption("⚠️ Glücksspiel kann süchtig machen. Hilfe: www.bzga.de")
