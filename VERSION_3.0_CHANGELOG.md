# 🎯 Version 3.0 - Heutige Spiele mit echten Quoten

## Das Problem in v2.2

Die App hat **nur historische Daten** geladen und dann mit **Beispiel-Quoten** gearbeitet:
- ❌ Keine echten heutigen Spiele
- ❌ Nur Demo-Quoten (Bayern vs Dortmund, City vs Liverpool)
- ❌ Keine echten Buchmacher-Quoten
- ❌ Keine Uhrzeiten

## Die Lösung in v3.0

Die App holt jetzt:
- ✅ **ECHTE heutige Spiele** aus der API
- ✅ **ECHTE Quoten** von Bet365 für jedes Spiel
- ✅ **Alle drei Märkte** (Heimsieg, Unentschieden, Auswärtssieg)
- ✅ **Uhrzeiten** der Spiele
- ✅ **Sortierung** nach Value (beste zuerst)

## Was wurde geändert?

### 1. Neue Funktion: `get_todays_fixtures()`

```python
def get_todays_fixtures(api_key, league_id, season):
    """Hole HEUTIGE Spiele mit Quoten"""
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Hole heutige Fixtures
    url = f'https://v3.football.api-sports.io/fixtures?league={league_id}&season={season}&date={today}'
    
    response = requests.get(url, headers=headers, timeout=15)
    
    # Für jedes Spiel: Hole Quoten
    for fixture in data['response']:
        fixture_id = fixture['fixture']['id']
        odds = get_fixture_odds(api_key, fixture_id)
        
        if odds:
            fixture['odds_data'] = odds
            fixtures_with_odds.append(fixture)
```

**Was macht das?**
1. Fragt API nach Spielen die **HEUTE** stattfinden
2. Holt für jedes Spiel die **echten Quoten** von Bet365
3. Gibt nur Spiele zurück, für die Quoten verfügbar sind

### 2. Neue Funktion: `get_fixture_odds()`

```python
def get_fixture_odds(api_key, fixture_id):
    """Hole Quoten für ein einzelnes Spiel"""
    # Bookmaker ID 8 = Bet365
    url = f'https://v3.football.api-sports.io/odds?fixture={fixture_id}&bookmaker=8'
    
    response = requests.get(url, headers=headers, timeout=10)
    
    # Suche "Match Winner" Markt
    for bet in bets:
        if bet.get('name') == 'Match Winner':
            # Extrahiere Home/Draw/Away Quoten
            odds = {
                'home': float(home_odd),
                'draw': float(draw_odd),
                'away': float(away_odd)
            }
```

**Was macht das?**
1. Fragt API nach Quoten für ein bestimmtes Spiel
2. Nutzt Bet365 als Buchmacher (ID 8)
3. Extrahiert die drei Quoten: Heimsieg, Unentschieden, Auswärtssieg

### 3. Verbesserte Analyse-Funktion

```python
def analyze_league(api_key, league_name, league_id):
    # 1. Historische Daten laden (120 Tage)
    historical = get_historical_fixtures(...)
    
    # 2. Modell trainieren
    attack, defense, home_adv = calculate_team_strengths(df)
    
    # 3. HEUTE'S Spiele holen (NEU!)
    todays_fixtures = get_todays_fixtures(api_key, league_id, season)
    
    # 4. Für jedes heutige Spiel:
    for fixture in todays_fixtures:
        home_team = fixture['teams']['home']['name']
        away_team = fixture['teams']['away']['name']
        odds = fixture.get('odds_data')  # ECHTE Quoten!
        
        # Berechne Wahrscheinlichkeiten
        probs = calculate_match_probabilities(home_team, away_team, ...)
        
        # Prüfe ALLE drei Märkte
        if probs['home_win'] * odds['home'] > 1.05:
            # VALUE BET gefunden!
```

**Was ist neu?**
- Verwendet `get_todays_fixtures()` statt Beispiel-Quoten
- Analysiert **alle drei Märkte** statt nur Heimsieg
- Zeigt **echte Uhrzeiten** der Spiele

### 4. Verbesserte Ausgabe

```python
value_bets.append({
    'liga': league_name,
    'zeit': fixture_time.strftime('%H:%M'),  # NEU: Uhrzeit
    'spiel': f"{home_team} vs {away_team}",
    'wette': market_name,  # z.B. "Sieg Bayern München"
    'wahrscheinlichkeit': f"{prob*100:.1f}%",
    'quote': f"{odd:.2f}",  # ECHTE Quote!
    'value': f"{value:.1f}%",
    'kelly_empfehlung': f"{kelly:.1f}%",
    'erwarteter_gewinn': f"{value * kelly / 100:.2f}%"
})
```

**Was ist neu?**
- **Zeit**: Wann das Spiel stattfindet (z.B. "15:30")
- **Quote**: Echte Quote von Bet365
- **Wette**: Genau welcher Markt (Heim/Draw/Away)
- **Erwarteter Gewinn**: Value × Kelly = erwarteter ROI

### 5. Top 3 Hervorhebung

```python
# Zeige Top 3 hervorgehoben
for idx, row in df_bets.head(3).iterrows():
    st.markdown(f"""
    <div class="value-bet">
        <h4>🎯 {row['spiel']} ({row['zeit']} Uhr)</h4>
        <p><strong>Wette:</strong> {row['wette']} @ {row['quote']}</p>
        <p><strong>Value:</strong> {row['value']} | <strong>Kelly:</strong> {row['kelly_empfehlung']}</p>
    </div>
    """, unsafe_allow_html=True)
```

**Was macht das?**
- Zeigt die 3 besten Value Bets visuell hervorgehoben
- Grüne Box mit allen wichtigen Infos
- Leicht lesbar und übersichtlich

## Workflow Vergleich

### ALT (v2.2):
1. Lade historische Daten ✅
2. Trainiere Modell ✅
3. Verwende **Demo-Quoten** ❌
4. Zeige Beispiel-Spiele ❌

### NEU (v3.0):
1. Lade historische Daten ✅
2. Trainiere Modell ✅
3. **Hole HEUTE'S Spiele** ✅
4. **Hole ECHTE Quoten** für jedes Spiel ✅
5. Analysiere alle Märkte ✅
6. Zeige nur Spiele von **HEUTE** ✅

## Beispiel-Output

### Vorher (v2.2):
```
Bayern München vs Borussia Dortmund
Sieg Bayern München
Quote: 1.85
Value: 8.5%
```
*(Demo-Spiel, nicht heute)*

### Nachher (v3.0):
```
🎯 Bayer Leverkusen vs Eintracht Frankfurt (15:30 Uhr)
Wette: Sieg Bayer Leverkusen @ 1.73
Value: 12.3% | Kelly: 3.2% | Erwarteter Gewinn: 0.39%
```
*(ECHTES Spiel von HEUTE mit ECHTEN Quoten)*

## API-Request Verbrauch

**Pro Analyse:**
- 1 Request: Historische Daten laden
- 1 Request: Heutige Spiele laden
- N Requests: Quoten für N Spiele

**Beispiel (Bundesliga mit 3 Spielen heute):**
- 1 Request: Historische Bundesliga-Daten
- 1 Request: Heute's Bundesliga-Spiele
- 3 Requests: Quoten für 3 Spiele
- **= 5 Requests total**

**Wichtig:** 
- Ohne Spiele heute = nur 2 Requests (historisch + heute)
- Mit Debug-Modus siehst du genau, wie viele Requests gemacht werden

## Installation

1. Ersetze `app.py` mit der neuen Version
2. Commit und Push zu GitHub
3. Streamlit deployed automatisch neu
4. **Wichtig:** Erste Analyse kann etwas dauern (Quoten laden)

## Nutzung

1. **API-Key eingeben** im Setup-Tab
2. **Ligen auswählen** (empfohlen: 2-3 Ligen)
3. **"Analyse Starten"** klicken
4. **Warten** (kann 1-2 Minuten dauern wegen Quoten)
5. **Ergebnisse** werden sortiert nach Value angezeigt

## Tipps

### Wenn keine Spiele gefunden werden:
- ✅ Prüfe ob heute Spieltag ist
- ✅ Versuche mehrere Ligen
- ✅ Aktiviere Debug-Modus um zu sehen was passiert

### Wenn keine Quoten gefunden werden:
- ✅ Bet365 hat nicht für alle Spiele Quoten
- ✅ Manchmal werden Quoten erst kurz vorher veröffentlicht
- ✅ Probiere andere Ligen

### Optimale Nutzung:
- 🕐 **Beste Zeit:** 1-2 Stunden vor Spielbeginn
- 📅 **Beste Tage:** Sa/So (Hauptspieltage)
- 🎯 **Beste Ligen:** Top-5-Ligen haben meist die besten Quoten

## Troubleshooting

### "Keine Spiele heute gefunden"
→ Heute ist kein Spieltag in dieser Liga
→ Wähle andere Ligen oder warte auf Spieltag

### "Keine Quoten verfügbar"
→ Bet365 hat noch keine Quoten veröffentlicht
→ Versuche es später nochmal (1-2h vor Spielbeginn)

### "API-Limit erreicht"
→ Zu viele Requests heute
→ Wähle weniger Ligen oder warte bis morgen

### "Fehler beim Laden"
→ Aktiviere Debug-Modus
→ Prüfe API-Key
→ Prüfe Internet-Verbindung

## Wichtige Hinweise

⚠️ **Dies ist keine Anlageberatung!**
- Die App zeigt statistischen Value
- Garantiert keine Gewinne
- Value Betting funktioniert nur langfristig
- Benötigt Disziplin und große Stichprobe

⚠️ **Glücksspiel kann süchtig machen!**
- Nur mit Geld spielen, das du verlieren kannst
- Setze dir Limits
- Bei Problemen: www.bzga.de

## Changelog

### v3.0 (Aktuell)
- ✅ Heutige Spiele mit API
- ✅ Echte Quoten von Bet365
- ✅ Alle drei Märkte analysiert
- ✅ Uhrzeiten angezeigt
- ✅ Sortierung nach Value
- ✅ Top 3 hervorgehoben

### v2.2
- ✅ API-Requests funktionieren
- ❌ Nur Demo-Quoten
- ❌ Keine heutigen Spiele

### v2.1
- ❌ API-Requests nicht gezählt
- ❌ Nur Demo-Quoten

## Support

Bei Fragen oder Problemen:
1. Aktiviere Debug-Modus
2. Screenshot der Ausgabe
3. Kontaktiere mich mit Details
