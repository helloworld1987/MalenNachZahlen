
# 🎨 Malen-nach-Zahlen Generator

Ein Python-Tool zum Erstellen von **Malen-nach-Zahlen Vorlagen** aus beliebigen Fotos.  
Die Ausgabe ist eine druckbare **PDF in 60×40 cm @ 300dpi** mit drei Seiten:

1. **Vorlage** mit Flächen, Konturen und Zahlen  
2. **Farblegende** (Amsterdam Standard Series Farben mit Code, Name und Swatch)  
3. **Farbige Vorschau** (ausgemalte Version mit Konturen darüber)

Ideal für eigene Mal-Leinwände mit Amsterdam Acrylfarben.

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
git clone https://github.com/DEIN-USER/malen-nach-zahlen.git
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

## 📜 Lizenz

MIT License – frei verwendbar für private & kommerzielle Projekte.  
