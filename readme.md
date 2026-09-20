
# 🎨 Malen-nach-Zahlen Generator

Ein Python-Tool zum Erstellen von **Malen-nach-Zahlen Vorlagen** aus beliebigen Fotos.  
Die Ausgabe ist eine druckbare **PDF in 60×40 cm @ 300dpi** mit drei Seiten:

1. **Vorlage** mit Flächen, Konturen und Zahlen  
2. **Farblegende** (Amsterdam Standard Series Farben mit Code, Name und Swatch)  
3. **Farbige Vorschau** (ausgemalte Version mit Konturen darüber)

Ideal für eigene Mal-Leinwände mit Amsterdam Acrylfarben.

---

## 🌐 Interaktive Browser-App (Neu!)

Neu: Eine moderne, interaktive Browser-App mit Null-Konfiguration:

```bash
./start.sh
# Alternativ: python3 -m http.server 8080 --directory web
```
Öffne [http://localhost:8080](http://localhost:8080) im Browser.

- **100% Client-Side:** Läuft komplett im Browser via Web Worker (Bilder verlassen nie deinen Rechner).
- **CIELAB K-Means Clustering:** Lebendige, wahrnehmungsgetreue Farben mit Amsterdam Acryl-Zuordnung.
- **5×5 2-Pass Bilateral-Filter & Multi-Pass Island-Cleaner:** Glättet raue Texturen (Asphalt, Putz, etc.) sauber glatt.
- **Polylabel-Algorithmus:** Exakte Zahlenplatzierung im Flächeninneren (keine Zahlen auf Kanten).
- **Interaktiver Malmodus:** Klick auf eine Farbe hebt alle entsprechenden Felder auf der Leinwand hervor.
- **1-Klick-Zuschnitt:** Fokussiert das Bild direkt auf das Hauptmotiv.
- **Export:** 3-seitige Druck-PDF (A4, A3, 60×40 cm), Vektor-SVG und PNG.

---

## ✨ Features

- **Vollständige Amsterdam Standard Series Palette** (alle Farben mit RGB)  
- **Presets** für unterschiedliche Szenarien (`--preset …`)  
- Fortschrittsanzeige mit ASCII-Bar ✅  
- **Vorverarbeitung**: Autokontrast, Gamma-Korrektur, Farbverstärkung, optional Bilateral-Filter  
- **Detailkontrolle**:  
  - Anzahl Farben (`--colors`)  
  - Mindest-Flächengröße (`--min-region`)  
  - Arbeitsauflösung (`--width`, `--height`)  
  - Mergen ähnlicher Farben (`--merge-close`, `--close-thresh`)  
- **Eindeutige Farbzuordnung** mit gewichteter Optimierung (`--force-unique`)  
- **Preview-Konturen** frei einstellbar (Farbe, Alpha, Linienstärke)  

---

## 🔧 Installation

```bash
# Repository klonen
git clone https://github.com/helloworld1987/MalenNachZahlen.git
cd malen-nach-zahlen

# Virtuelle Umgebung anlegen (optional, empfohlen)
python -m venv venv
source venv/bin/activate  # (Linux/Mac)
venv\Scripts\activate     # (Windows)

# Abhängigkeiten installieren
pip install -r requirements.txt
```

**requirements.txt** (Minimal):
```
pillow
numpy
scikit-image
PyYAML
scipy     # optional, für bessere Linienverdickung & Matching
```

---

## 🚀 Verwendung

```bash
python malen_nach_zahlen.py <input.jpg> --out <output.pdf> [Optionen]
```

Beispiele:

```bash
# Meer + Portrait (empfohlen für dein Küstenfoto)
python malen_nach_zahlen.py kinder_an_der_kueste.jpg   --preset sea-portrait   --out kinderanderkueste.pdf

# Portrait mit satteren Hauttönen, manuell 26 Farben
python malen_nach_zahlen.py portrait.jpg   --preset portrait --colors 26 --out portrait26.pdf

# Detaillierte Version mit vielen Farben
python malen_nach_zahlen.py bild.jpg   --colors 34 --width 3200 --height 2134   --min-region 60 --out detailreich.pdf
```

---

## 🎛 Presets

| Preset         | Beschreibung                                    | Default Farben |
|----------------|-------------------------------------------------|----------------|
| `sea-portrait` | Meer + Gesichter, feine Hauttöne, weicher Himmel| 28             |
| `portrait`     | Klassische Portraits, sattere Hautfarben        | 24             |
| `landscape-sky`| Landschaft mit viel Himmel/Wasser               | 26             |
| `high-contrast`| Grafischer Look, weniger Farben, starke Kanten  | 18             |
| `draft-fast`   | Sehr schnelle Skizze, einfache Vorschau         | 16             |

> Alle Optionen lassen sich bei Bedarf **überschreiben** – Presets setzen nur gute Ausgangswerte.

---

## ⚙️ Wichtige Optionen

- `--colors N` → Anzahl Farben (12–36 empfohlen)  
- `--min-region PX` → Mindest-Fläche in Pixeln, kleine Inseln werden gemerged  
- `--force-unique` → Jede Farbe erhält einen einzigartigen Amsterdam-Ton  
- `--merge-close` → Sehr ähnliche Farben zusammenfassen (Schwelle via `--close-thresh`)  
- `--gamma` → Helligkeitsanpassung (0.85 dunkler, 1.0 neutral)  
- `--color-boost` → Farbsättigung verstärken (z. B. 1.05)  
- `--boundary-alpha` → Transparenz der Vorschau-Konturen (0–255)  
- `--line-thickness` → Konturstärke (1–3, mit SciPy >1 möglich)

### ℹ️ Hinweis zu `--merge-close` und `--close-thresh`

Die Option `--merge-close` fasst **sehr ähnliche Farben** zusammen, um die Palette zu vereinfachen.  
`--close-thresh` legt dabei den **RGB-Abstand** fest, ab wann zwei Farben als „ähnlich“ gelten.

- Formel:  
  \[
  d = \sqrt{(R_1-R_2)^2 + (G_1-G_2)^2 + (B_1-B_2)^2}
  \]

- Wenn `d < close-thresh`, werden die Farben gemerged.

#### Empfohlene Werte
| Szenario             | Schwelle | Wirkung |
|----------------------|----------|---------|
| **Portraits**        | 12–14    | Bewahrt feine Hautnuancen, kaum Doppelungen |
| **Meer/Himmel**      | 14–16    | Vermeidet 2–3 fast gleiche Blautöne, Details bleiben sichtbar |
| **Landschaften**     | 16–20    | Vereinfacht viele Grüntöne, ruhigeres Bild |
| **Grafisch/Poster**  | 20–24    | Maximale Vereinfachung, deutliche Reduktion der Farbvielfalt |

#### Merke
- **Kleinere Werte** → mehr Details, aber evtl. sehr ähnliche Farben in der Legende.  
- **Größere Werte** → ruhigeres Bild, aber Gefahr, dass Nuancen verloren gehen.  

---

## 📄 Output

- PDF mit **3 Seiten**:
  1. Malen-nach-Zahlen Vorlage (60×40 cm @ 300dpi)
  2. Farblegende mit Amsterdam Codes
  3. Farbvorschau (ausgemalt + Konturen)

---

## 📝 Hinweise

- `--force-unique` kann Details verfälschen, wenn die Palette keine passenden Töne für Himmel/Meer bietet.  
- Für **mehr Details**:
  - `--colors` erhöhen (28–34)  
  - `--min-region` verkleinern (60–90)  
  - `--width/--height` höher setzen (z. B. 3000×2000)  
  - `--merge-close` weglassen  
- Für **ruhigere Bilder**:
  - Weniger Farben (16–20)  
  - `--min-region` erhöhen (120–150)  
  - `--merge-close` aktivieren  

---
## 📜 Changelog

### v2.0.0 (2026-09-20)

-   **🎨 Authentischer Acryl-Impasto-Shader (Reverse-Engineered):**
    -   Physikalische **Kantenwülste (Boundary Paint Ridges)** per 2-Pass-Chamfer-Distanztransform (Farbe baut sich realistisch am Rand der Farbfelder auf).
    -   **Farbfeld-orientierter Pinselduktus:** Automatische Berechnung der Hauptträgheitsachsen (PCA-Momente 2. Ordnung $\mu_{20}, \mu_{02}, \mu_{11}$) für jedes Farbfeld – Pinselzüge folgen der exakten geometrischen Strömung von Bauteilen, Straßen und Wänden.
    -   **Tiefschwarz-Schutz & Kantenglanz:** Schwarze und dunkle Farbtöne (z. B. Lampenschwarz) behalten ihre volle Tiefe ohne Grauschleier, während seidenglänzende Glanzlichter Kanten akzentuieren.
    -   Echtes Leinwandgewebe (Canvas Linen Weave) mit dezentem Fadenkreuz.
-   **💾 Projekt Speichern & Laden (`.mnz`):**
    -   Projekte können inklusive aller Vektorkonturen, Regionen, Zahlenpositionen, Farblegende, Bildanpassungen und Originalbild als `.mnz`-Datei gespeichert und sofort ohne Neuberechnung wieder geladen werden.
-   **⚡ Kuwahara-Filterung & Schatten-Lifting:**
    -   Malerische Abstraktion zur Auflösung von starren Pixelrastern in organische Kunstflächen.
-   **📐 Glatte Vektor-Outlines & Polylabel:**
    -   Anti-Aliased Vektorkonturen (RDP + Chaikin-Glättung) und Pole of Inaccessibility für perfekte Zahlenplatzierung mit Schutz-Halo.

### v1.1.0 (2025-09-10)

-   **Feature: Perzeptuelle Farbmetrik (CIELAB)**
    -   Neues Argument `--color-metric` (Standard: `cielab`).
    -   Verwendet den CIELAB-Farbraum zur Berechnung von Farbabständen, was die menschliche Wahrnehmung besser widerspiegelt.
    -   **Problembehebung:** Helle Blautöne werden nun deutlich besser von Weiß unterschieden.
    -   Die alte Methode ist weiterhin über `--color-metric rgb` verfügbar.

-   **Verbesserung: Detaillierte Fortschrittsanzeige**
    -   Die Konsole zeigt nun den aktuellen Schritt und den nächsten anstehenden Schritt an.

-   **Verbesserung: Alle Parameter per CLI steuerbar**
    -   Alle Konfigurationsoptionen (z.B. `gamma`, `min_region`) können nun direkt als Kommandozeilen-Argumente übergeben werden und überschreiben die Preset-Werte.

### v1.0.1 (2025-09-10)

-   **Fix:** Behebt einen `ValueError` im Schritt `Flächenbereinigung` durch korrekte Datenübergabe.
-   **Fix:** Löst einen `NameError` in der gewichteten Farbzuordnung.
-   **Verbesserung:** Die Konsolen-Fortschrittsanzeige wird nun sauber ohne Artefakte gerendert.
-   **Wartung:** Veralteter `mode`-Parameter bei `Image.fromarray()` entfernt, um Kompatibilität mit zukünftigen Pillow-Versionen zu gewährleisten.

---

## 📜 Lizenz

MIT License – frei verwendbar für private & kommerzielle Projekte.  
