"""
Fußballwetten-Analyse-App v3.0 (Heute's Spiele + Echte Quoten)
Zeigt heutige Spiele mit echten Quoten von API-Football
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from scipy.stats import poisson
import time

# ============================================================================
# KONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="⚽ Wetten-Analyst Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    .stButton > button {
        width: 100%;
        padding: 0.75rem;
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    h1 {
        font-size: 1.8rem !important;
        text-align: center;
    }
    .value-bet {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin-bottom: 0.5rem;
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
    "🇪🇸 Segunda División": 141,
    "🇮🇹 Serie A": 135,
    "🇮🇹 Serie B": 136,
    "🇫🇷 Ligue 1": 61,
    "🇫🇷 Ligue 2": 62,
}

# ============================================================================
# SESSION STATE
# ============================================================================

if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

# ============================================================================
# API FUNKTIONEN
# ============================================================================

def get_current_season():
    """Bestimme die aktuelle Saison basierend auf dem Datum"""
    now = datetime.now()
    if now.month >= 7:
        return now.year
    else:
        return now.year - 1

def test_api_connection(api_key):
    """Teste die API-Verbindung"""
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

def get_historical_fixtures(api_key, league_id, season, days_back=120):
    """Hole historische Spiele für Modell-Training"""
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
                # Nur abgeschlossene Spiele für Training
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
    """Hole HEUTIGE Spiele mit Quoten"""
    headers = {
        'x-rapidapi-host': 'v3.football.api-sports.io',
        'x-rapidapi-key': api_key
    }
    
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Hole heutige Fixtures
    url = f'https://v3.football.api-sports.io/fixtures?league={league_id}&season={season}&date={today}'
    
    if st.session_state.debug_mode:
        st.write(f"🔍 Hole heute's Spiele: {url}")
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'response' in data and len(data['response']) > 0:
                fixtures_with_odds = []
                
                for fixture in data['response']:
                    fixture_id = fixture['fixture']['id']
                    
                    # Hole Quoten für dieses Spiel
                    odds = get_fixture_odds(api_key, fixture_id)
                    
                    if odds:
                        fixture['odds_data'] = odds
                        fixtures_with_odds.append(fixture)
                    
                    time.sleep(0.3)  # Rate limiting
                
                return fixtures_with_odds
        return []
            
    except Exception as e:
        if st.session_state.debug_mode:
            st.error(f"Fehler bei heutigen Spielen: {str(e)}")
        return []

def get_fixture_odds(api_key, fixture_id):
    """Hole Quoten für ein einzelnes Spiel"""
    headers = {
        'x-rapidapi-host': 'v3.football.api-sports.io',
        'x-rapidapi-key': api_key
    }
    
    # Hole Quoten (Bet365 = Bookmaker ID 8)
    url = f'https://v3.football.api-sports.io/odds?fixture={fixture_id}&bookmaker=8'
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'response' in data and len(data['response']) > 0:
                bookmaker_data = data['response'][0]
                
                if 'bookmakers' in bookmaker_data and len(bookmaker_data['bookmakers']) > 0:
                    bets = bookmaker_data['bookmakers'][0].get('bets', [])
                    
                    # Suche nach "Match Winner" Markt
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
        if st.session_state.debug_mode:
            st.write(f"⚠️ Keine Quoten für Fixture {fixture_id}: {str(e)}")
        return None

def process_fixtures_to_dataframe(fixtures):
    """Verarbeite historische Fixtures zu DataFrame"""
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
        df['date'] = pd.to_datetime(df['date'])
        return df
    return pd.DataFrame()

# ============================================================================
# DIXON-COLES MODELL
# ============================================================================

def calculate_team_strengths(df):
    """Berechne Team-Stärken mit Dixon-Coles Ansatz"""
    teams = pd.concat([df['home_team'], df['away_team']]).unique()
    
    # Initialisiere Stärken
    attack = {team: 1.0 for team in teams}
    defense = {team: 1.0 for team in teams}
    home_advantage = 0.3
    
    # Iterative Optimierung
    for iteration in range(10):
        for team in teams:
            home_matches = df[df['home_team'] == team]
            away_matches = df[df['away_team'] == team]
            
            if len(home_matches) > 0:
                attack[team] = home_matches['home_goals'].mean() / (1 + home_advantage)
            if len(away_matches) > 0:
                defense[team] = 1.0 / (away_matches['away_goals'].mean() + 0.1)
    
    return attack, defense, home_advantage

def calculate_match_probabilities(home_team, away_team, attack, defense, home_adv):
    """Berechne Spielwahrscheinlichkeiten"""
    try:
        lambda_home = attack.get(home_team, 1.0) * defense.get(away_team, 1.0) * (1 + home_adv)
        lambda_away = attack.get(away_team, 1.0) * defense.get(home_team, 1.0)
        
        max_goals = 5
        prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
        
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                prob_matrix[i, j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
        
        prob_home = np.sum(np.tril(prob_matrix, -1))
        prob_draw = np.sum(np.diag(prob_matrix))
        prob_away = np.sum(np.triu(prob_matrix, 1))
        
        # Normalisiere
        total = prob_home + prob_draw + prob_away
        if total > 0:
            prob_home /= total
            prob_draw /= total
            prob_away /= total
        
        return {
            'home_win': prob_home,
            'draw': prob_draw,
            'away_win': prob_away
        }
    except:
        return {'home_win': 0.33, 'draw': 0.33, 'away_win': 0.33}

def calculate_kelly_bet(probability, odds, fraction=0.25):
    """Berechne Kelly-Einsatz (konservativ)"""
    if odds <= 1.01:
        return 0
    
    edge = (probability * odds) - 1
    if edge <= 0:
        return 0
    
    kelly = (probability * odds - 1) / (odds - 1)
    kelly_bet = kelly * fraction * 100
    
    return max(0, min(kelly_bet, 10))

# ============================================================================
# ANALYSE-FUNKTIONEN
# ============================================================================

def analyze_league(api_key, league_name, league_id):
    """Analysiere eine Liga mit heutigen Spielen"""
    season = get_current_season()
    
    st.info(f"🔍 Analysiere {league_name}...")
    
    # 1. Hole historische Daten für Modell-Training
    with st.spinner("Lade historische Daten..."):
        historical = get_historical_fixtures(api_key, league_id, season, days_back=120)
    
    if not historical or len(historical) < 20:
        st.warning(f"⚠️ Zu wenig historische Daten für {league_name} ({len(historical) if historical else 0} Spiele)")
        return []
    
    st.success(f"✅ {len(historical)} historische Spiele geladen")
    
    # 2. Trainiere Modell
    df = process_fixtures_to_dataframe(historical)
    
    if df.empty:
        st.warning(f"⚠️ Keine verwertbaren Daten")
        return []
    
    attack, defense, home_adv = calculate_team_strengths(df)
    
    # 3. Hole heutige Spiele mit Quoten
    with st.spinner("Lade heutige Spiele mit Quoten..."):
        todays_fixtures = get_todays_fixtures(api_key, league_id, season)
    
    if not todays_fixtures:
        st.info(f"ℹ️ Keine Spiele heute für {league_name}")
        return []
    
    st.success(f"✅ {len(todays_fixtures)} Spiele heute gefunden")
    
    # 4. Analysiere heutige Spiele
    value_bets = []
    
    for fixture in todays_fixtures:
        try:
            home_team = fixture['teams']['home']['name']
            away_team = fixture['teams']['away']['name']
            fixture_time = datetime.fromisoformat(fixture['fixture']['date'].replace('Z', '+00:00'))
            
            odds = fixture.get('odds_data')
            
            if not odds:
                continue
            
            # Berechne Wahrscheinlichkeiten
            probs = calculate_match_probabilities(home_team, away_team, attack, defense, home_adv)
            
            # Prüfe alle drei Märkte
            markets = [
                ('home', 'Sieg ' + home_team, probs['home_win'], odds.get('home')),
                ('draw', 'Unentschieden', probs['draw'], odds.get('draw')),
                ('away', 'Sieg ' + away_team, probs['away_win'], odds.get('away'))
            ]
            
            for market_type, market_name, prob, odd in markets:
                if odd and prob * odd > 1.05:  # Mindestens 5% Value
                    kelly = calculate_kelly_bet(prob, odd)
                    
                    if kelly > 0.5:  # Mindestens 0.5% Kelly
                        value = ((prob * odd - 1) * 100)
                        
                        value_bets.append({
                            'liga': league_name,
                            'zeit': fixture_time.strftime('%H:%M'),
                            'spiel': f"{home_team} vs {away_team}",
                            'wette': market_name,
                            'wahrscheinlichkeit': f"{prob*100:.1f}%",
                            'quote': f"{odd:.2f}",
                            'value': f"{value:.1f}%",
                            'kelly_empfehlung': f"{kelly:.1f}%",
                            'erwarteter_gewinn': f"{value * kelly / 100:.2f}%"
                        })
        
        except Exception as e:
            if st.session_state.debug_mode:
                st.write(f"⚠️ Fehler bei Spiel: {str(e)}")
            continue
    
    return value_bets

def run_full_analysis(api_key, selected_leagues):
    """Führe komplette Analyse durch"""
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
# MAIN UI
# ============================================================================

def main():
    st.title("⚽ Wetten-Analyst Pro")
    st.caption(f"v3.0 | Heutige Spiele | {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["⚙️ Setup", "🎯 Heute's Analyse", "❓ Info"])
    
    with tab1:
        st.header("⚙️ Einstellungen")
        
        api_key = st.text_input(
            "API-Football Key",
            type="password",
            value=st.session_state.api_key,
            help="Key von api-football.com"
        )
        st.session_state.api_key = api_key
        
        st.session_state.debug_mode = st.checkbox(
            "🐛 Debug-Modus",
            value=st.session_state.debug_mode
        )
        
        if api_key:
            if st.button("🔍 API Testen"):
                success, message = test_api_connection(api_key)
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        st.divider()
        
        st.subheader("🏆 Ligen auswählen")
        selected_leagues = {}
        
        col1, col2 = st.columns(2)
        with col1:
            for i, (name, id) in enumerate(list(LEAGUES.items())[:5]):
                if st.checkbox(name, value=True, key=f"league_{i}"):
                    selected_leagues[name] = id
        
        with col2:
            for i, (name, id) in enumerate(list(LEAGUES.items())[5:], start=5):
                if st.checkbox(name, value=True, key=f"league_{i}"):
                    selected_leagues[name] = id
    
    with tab2:
        st.header("🎯 Heute's Value Bets")
        
        if not st.session_state.api_key:
            st.warning("⚠️ Bitte zuerst API-Key eingeben!")
            return
        
        if st.button("🚀 Analyse Starten", type="primary"):
            if not selected_leagues:
                st.error("❌ Bitte mindestens eine Liga auswählen!")
                return
            
            st.info(f"🔄 Analysiere {len(selected_leagues)} Ligen...")
            
            value_bets = run_full_analysis(st.session_state.api_key, selected_leagues)
            
            st.divider()
            
            if value_bets:
                st.success(f"✅ {len(value_bets)} Value Bets gefunden!")
                
                # Sortiere nach Value
                df_bets = pd.DataFrame(value_bets)
                df_bets = df_bets.sort_values('value', ascending=False)
                
                # Zeige als schöne Tabelle
                st.dataframe(
                    df_bets,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download
                st.download_button(
                    "📥 Als CSV Herunterladen",
                    df_bets.to_csv(index=False).encode('utf-8'),
                    f"value_bets_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv"
                )
                
                # Zeige Top 3 hervorgehoben
                st.divider()
                st.subheader("🏆 Top 3 Value Bets")
                
                for idx, row in df_bets.head(3).iterrows():
                    st.markdown(f"""
                    <div class="value-bet">
                        <h4>🎯 {row['spiel']} ({row['zeit']} Uhr)</h4>
                        <p><strong>Wette:</strong> {row['wette']} @ {row['quote']}</p>
                        <p><strong>Value:</strong> {row['value']} | <strong>Kelly:</strong> {row['kelly_empfehlung']} | <strong>Erwarteter Gewinn:</strong> {row['erwarteter_gewinn']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("ℹ️ Keine Value Bets für heute gefunden")
                st.caption("Entweder keine Spiele heute oder keine positiven Wert-Wetten gefunden.")
    
    with tab3:
        st.header("❓ Anleitung")
        
        st.markdown("""
        ### 📚 So funktioniert's:
        
        1. **API-Key eingeben** (Setup-Tab)
        2. **Ligen auswählen**
        3. **"Analyse Starten"** klicken
        
        ### 🎯 Was macht die App?
        
        1. **Lädt historische Daten** (letzte 120 Tage) zum Modell-Training
        2. **Trainiert Dixon-Coles Modell** mit diesen Daten
        3. **Holt HEUTIGE Spiele** mit echten Quoten von Bet365
        4. **Vergleicht** Modell-Wahrscheinlichkeiten mit Buchmacher-Quoten
        5. **Zeigt Value Bets** wo deine Wahrscheinlichkeit höher ist
        
        ### 💰 Was bedeutet "Value"?
        
        - **Value = (Wahrscheinlichkeit × Quote - 1) × 100**
        - **Beispiel**: 40% Wahrscheinlichkeit × Quote 2.80 = 12% Value
        - **Positiver Value** = statistischer Vorteil gegenüber Buchmacher
        
        ### 📊 Kelly-Kriterium
        
        - Berechnet optimalen Einsatz basierend auf Value
        - **Konservativ**: 25% des theoretischen Kelly
        - **Maximal 10%** der Bankroll pro Wette
        
        ### ⚠️ Wichtig
        
        - Dies ist **keine Garantie** für Gewinne
        - Value Betting ist **langfristige Strategie**
        - Benötigt große Stichprobe und Disziplin
        - **Glücksspiel kann süchtig machen**
        
        ### 🆕 Neu in v3.0
        
        ✅ **Echte heutige Spiele** mit Uhrzeit
        ✅ **Echte Quoten** von Bet365 (via API)
        ✅ **Alle drei Märkte** (Heim/Unentschieden/Auswärts)
        ✅ **Sortierung nach Value**
        ✅ **Top 3 hervorgehoben**
        
        ### 📝 API-Nutzung
        
        Pro Analyse werden etwa verbraucht:
        - 1 Request pro Liga (historische Daten)
        - 1 Request pro Liga (heutige Spiele)
        - 1 Request pro Spiel (Quoten)
        
        **Beispiel**: 3 Ligen mit je 2 Spielen = ca. 12 Requests
        """)
        
        st.divider()
        st.caption("⚠️ Glücksspiel kann süchtig machen. Hilfe: www.bzga.de")

if __name__ == "__main__":
    main()
