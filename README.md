# ⚽ Football Betting Analyst Pro v2.1

Professionelle Fußballwetten-Analyse mit Dixon-Coles-Modell und Kelly-Kriterium.

## 🆕 Version 2.1 - Wichtige Fixes

### ✅ Behobenes Problem: API gibt keine Daten zurück

**Das Problem:**
- Die App verwendete `status=FT` für API-Anfragen
- API-Football v3 benötigt aber mehrere Status-Werte kombiniert
- Spiele mit Verlängerung/Elfmeterschießen wurden nicht gefunden

**Die Lösung:**
```python
# ❌ ALT (funktioniert nicht richtig):
status = "FT"

# ✅ NEU (korrekt):
status = "FT-AET-PEN"
```

**Status-Codes erklärt:**
- `FT` = Full Time (nach 90 Minuten beendet)
- `AET` = After Extra Time (nach Verlängerung beendet)
- `PEN` = After Penalty (nach Elfmeterschießen beendet)

## 🚀 Quick Start

### 1. Repository klonen oder herunterladen

```bash
git clone https://github.com/muuuhli/football-betting-app.git
cd football-betting-app
```

### 2. Dependencies installieren

```bash
pip install -r requirements.txt
```

### 3. API-Key besorgen

- Gehe zu [api-football.com](https://www.api-football.com)
- Registriere dich kostenlos
- Hole deinen API-Key aus dem Dashboard

### 4. App starten

```bash
streamlit run app.py
```

## 📦 Deployment auf Streamlit Cloud

### Option A: Über GitHub (empfohlen)

1. **Repository auf GitHub pushen:**
   ```bash
   git add .
   git commit -m "v2.1 - Fixed API status parameter"
   git push origin main
   ```

2. **Auf Streamlit Cloud deployen:**
   - Gehe zu [share.streamlit.io](https://share.streamlit.io)
   - Klicke "New app"
   - Wähle dein Repository
   - Branch: `main`
   - Main file: `app.py`
   - Klicke "Deploy"

3. **Secrets konfigurieren (optional):**
   - Settings → Secrets
   - Füge hinzu:
     ```toml
     api_key = "dein_api_key_hier"
     ```

### Option B: Lokales Deployment

1. **Streamlit lokal starten:**
   ```bash
   streamlit run app.py
   ```

2. **Oder mit Docker:**
   ```bash
   docker build -t betting-app .
   docker run -p 8501:8501 betting-app
   ```

## 🔧 Konfiguration

### Unterstützte Ligen

Die App unterstützt folgende Top-Ligen:

- 🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League (39)
- 🇩🇪 Bundesliga (78)
- 🇩🇪 2. Bundesliga (79)
- 🇪🇸 La Liga (140)
- 🇪🇸 Segunda División (141)
- 🇮🇹 Serie A (135)
- 🇮🇹 Serie B (136)
- 🇫🇷 Ligue 1 (61)
- 🇫🇷 Ligue 2 (62)

### Saison-Auswahl

Die App erkennt automatisch die aktuelle Saison:
- **Juli - Dezember**: Aktuelles Jahr (z.B. 2024 für Saison 2024/25)
- **Januar - Juni**: Vorheriges Jahr (z.B. 2024 für Saison 2024/25)

## 📊 Features

### Dixon-Coles Modell
- Statistische Spielvorhersage
- Team-Stärken (Angriff/Verteidigung)
- Heimvorteil-Berechnung

### Kelly-Kriterium
- Optimale Einsatzberechnung
- 20% Kelly (konservativ)
- Bankroll-Management

### Value Betting
- Nur Wetten mit positivem Erwartungswert
- Mindestens 5% Edge
- Automatische Filterung

## 🐛 Troubleshooting

### Problem: "Keine Daten gefunden"

**Ursache:** Falsche Saison oder Status-Parameter

**Lösung:**
1. Prüfe, ob die aktuelle Saison korrekt erkannt wird
2. Stelle sicher, dass `COMPLETED_STATUSES = "FT-AET-PEN"` verwendet wird
3. Teste mit `include_live=True` für mehr Ergebnisse

### Problem: "API-Fehler 401"

**Ursache:** Ungültiger oder abgelaufener API-Key

**Lösung:**
1. Gehe zu [api-football.com](https://www.api-football.com)
2. Überprüfe deinen API-Key
3. Erneuere bei Bedarf

### Problem: "Rate Limit erreicht"

**Ursache:** Zu viele API-Anfragen

**Lösung:**
1. Warte 1 Minute
2. Reduziere Anzahl der Ligen
3. Upgrade API-Plan bei Bedarf

## 🔍 Debug-Modus

Für detaillierte Fehlersuche:

```python
# In der App aktivieren:
# Settings → Debug-Modus aktivieren

# Oder im Code:
st.session_state.debug_mode = True
```

## 📝 Changelog

### v2.1 (2024-10-19)
- ✅ **FIX**: Korrekter Status-Parameter `FT-AET-PEN`
- ✅ **FIX**: Automatische Saison-Erkennung
- ✅ Bessere Fehlerbehandlung
- ✅ Verbesserte UI-Meldungen

### v2.0 (2024-10-18)
- 🎯 Dixon-Coles Implementierung
- 📊 Kelly-Kriterium Integration
- 🎨 Mobile-optimiertes Design
- 🔧 Debug-Modus

## ⚠️ Disclaimer

**Wichtige Hinweise:**
- Diese App dient nur zur Information
- Keine Anlageberatung
- Glücksspiel kann süchtig machen
- Spiele verantwortungsvoll
- Hilfe: www.bzga.de

## 📄 Lizenz

MIT License - Siehe LICENSE Datei

## 🤝 Contributing

Pull Requests sind willkommen!

1. Fork das Projekt
2. Erstelle deinen Feature Branch
3. Commit deine Änderungen
4. Push zum Branch
5. Öffne einen Pull Request

## 📧 Support

Bei Fragen oder Problemen:
- GitHub Issues erstellen
- [api-football.com Dokumentation](https://www.api-football.com/documentation-v3)

---

**Viel Erfolg! 🍀**
