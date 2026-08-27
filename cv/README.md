# Klassische CV-Kratzerdetektion – finaler eingefrorener Stand

## Zweck

Dieses Projekt enthält den finalen klassischen Computer-Vision-Ansatz zur Detektion von Kratzern auf metallischen Oberflächen.

Der Detektor ist **eingefroren**. Die Parameter in `config.json` entsprechen dem zuletzt abgestimmten Stand und sollten für die finale Auswertung nicht mehr verändert werden.

Die Auswertung wird vollständig reproduzierbar durch `evaluate.py` berechnet. Es werden keine manuell eingetragenen Ergebniswerte verwendet.

## Ordnerstruktur

```text
cv/
├── data/
│   ├── images/              34 Originalbilder
│   └── masks/               zugehörige Ground-Truth-Masken
├── results/                 wird automatisch erzeugt
├── config.json              finale CV- und Evaluationsparameter
├── cv_pipeline.py           eigentliche CV-Pipeline
├── run_detection.py         erzeugt Prediction-Masks und Overlays
├── evaluate.py              vergleicht Predictions mit Ground Truth
├── pyproject.toml
├── uv.lock
└── README.md
```

Nicht enthalten sind virtuelle Umgebungen, Python-Caches, alte Optimierungsläufe oder alte Ergebnisordner.

---

# 1. Installation

Im Projektordner:

```bash
uv sync
```

---

# 2. Finale Detektion ausführen

```bash
uv run python run_detection.py
```

Dabei werden alle Bilder aus `data/images/` verarbeitet.

Erzeugt werden:

```text
results/
├── prediction_masks/        binäre finale Detektionen
├── overlays/                visuelle Kontrolle
├── detection_summary.csv    einfache technische Statistik je Bild
└── used_config.json         tatsächlich verwendete Konfiguration
```

Overlay-Farben:

- **Rot**: finale Kratzerdetektion
- **Blau**: erkannte und ausgeschlossene Kreis-/Bohrungsregion
- **Gelb**: als Außenkantenartefakt verworfene Komponente

## Einzelnes Bild prüfen

```bash
uv run python run_detection.py --image 13_max_flat.png
```

Mit Zwischenschritten:

```bash
uv run python run_detection.py --image 13_max_flat.png --debug
```

Die Debug-Bilder landen dann unter `results/debug/`.

---

# 3. Finale Auswertung ausführen

Zuerst muss einmal die Detektion über alle Bilder gelaufen sein:

```bash
uv run python run_detection.py
```

Danach:

```bash
uv run python evaluate.py
```

Die Auswertung verwendet ausschließlich:

```text
data/masks/                  Ground Truth
results/prediction_masks/    finale CV-Vorhersagen
```

und berechnet die Ergebnisse bei jedem Lauf neu.

Erzeugt werden nur vier Auswertungsdateien:

```text
results/evaluation/
├── summary.json
├── per_image.csv
├── scratch_instances.csv
└── size_bins.csv
```

## `summary.json`

Enthält die zusammengefassten Kennzahlen:

- Pixel-Precision
- Pixel-Recall
- Pixel-F1
- IoU
- Specificity
- toleranzbasierte Pixelmetriken
- Kratzer-Erkennungsrate bei verschiedenen Überdeckungsgrenzen
- Precision der vorhergesagten Komponenten
- automatisch ermittelte Größenschwellen für die gewünschte Erkennungsrate

## `per_image.csv`

Enthält die wichtigsten Kennzahlen separat für jedes Bild. Dadurch können schwierige Bauteile gezielt identifiziert werden.

## `scratch_instances.csv`

Eine Zeile entspricht einem zusammenhängenden Kratzer aus der Ground-Truth-Maske.

Ausgegeben werden unter anderem:

- Fläche in Pixeln
- geschätzte Länge in Pixeln
- geschätzte Breite in Pixeln
- Anzahl überdeckter Pixel
- Überdeckungsanteil
- erkannt / nicht erkannt für mehrere Überdeckungsschwellen

Diese Datei bildet die Grundlage für die spätere Aussage, bis zu welcher Kratzergröße die klassische CV-Methode zuverlässig funktioniert.

## `size_bins.csv`

Fasst die Einzelkratzer in Größenklassen zusammen für:

- Fläche
- geschätzte Länge
- geschätzte Breite

Für jede Größenklasse werden ausgegeben:

```text
n_scratches
n_detected
detection_rate
ci95_low
ci95_high
```

`detection_rate` ist die **empirisch gemessene Erkennungsrate** in dieser Größenklasse.

Das 95-%-Konfidenzintervall wird mit dem Wilson-Verfahren berechnet. Dadurch ist direkt erkennbar, wie belastbar eine Erkennungsrate bei der jeweiligen Anzahl an Kratzern ist.

---

# 4. Definition der Bewertungsmetriken

## Strikte Pixelmetriken

Pixel werden exakt miteinander verglichen.

### Precision

```text
TP / (TP + FP)
```

Welcher Anteil aller als Kratzer markierten Pixel tatsächlich zu einer Ground-Truth-Kratzermaske gehört.

### Recall

```text
TP / (TP + FN)
```

Welcher Anteil der annotierten Kratzerpixel durch die CV-Methode gefunden wird.

### F1

Harmonisches Mittel aus Precision und Recall.

### IoU

```text
TP / (TP + FP + FN)
```

Überdeckung zwischen Prediction und Ground Truth.

Die reine Accuracy ist für diesen Datensatz keine sinnvolle Hauptmetrik, weil Hintergrundpixel zahlenmäßig sehr stark überwiegen.

## Toleranzbasierte Pixelmetriken

Segmentierungsmasken von dünnen Kratzern sind nicht vollständig objektiv. Eine Prediction kann den richtigen Kratzer treffen, aber einige Pixel schmaler oder leicht versetzt sein.

Deshalb berechnet das Programm zusätzlich Precision, Recall und F1 mit räumlichen Toleranzen. Standardmäßig:

```text
0 px
2 px
5 px
10 px
```

Die Werte stehen in `config.json` und sind für die finale Auswertung dokumentiert.

## Kratzerbasierte Detektion

Jede zusammenhängende Komponente der Ground-Truth-Maske wird als einzelner Kratzer behandelt.

Für jeden Kratzer wird berechnet:

```text
overlap_fraction = überdeckte GT-Pixel / alle Pixel dieses GT-Kratzers
```

Standardmäßig werden Ergebnisse für folgende Mindestüberdeckungen ausgewiesen:

```text
5 %
15 %
30 %
50 %
```

Der primäre Wert ist aktuell **15 %**.

Dadurch wird nicht nur bewertet, ob die komplette Segmentierungsfläche pixelgenau getroffen wurde, sondern auch, ob ein Kratzer grundsätzlich lokalisiert werden konnte.

## Größe eines Kratzers

### Fläche

Direkte Anzahl der Pixel des Ground-Truth-Kratzers.

### Geschätzte Länge

Die Pixel des Kratzers werden mittels PCA analysiert. Als Länge wird die Ausdehnung entlang der dominanten Hauptachse verwendet.

Dies ist eine reproduzierbare geometrische Näherung und keine Skelettlänge eines gekrümmten Kratzers.

### Geschätzte Breite

```text
Breite = Fläche / geschätzte Länge
```

Auch dies ist eine effektive mittlere Breite und keine lokale Maximal- oder Minimalbreite.

---

# 5. Finale CV-Pipeline

Die Detektion erfolgt in dieser Reihenfolge:

```text
Originalbild
    ↓
Graustufen
    ↓
Bauteil-ROI bestimmen
    ↓
Außenrandabstand berücksichtigen
    ↓
Grauwertspreizung
    ↓
Bohrungen / Kreisstrukturen erkennen und ausschließen
    ↓
leichter Gaussian Blur
    ↓
Local Residual
    ↓
Perzentil-Schwellwert
    ↓
Closing
    ↓
kleine Komponenten entfernen
    ↓
randnahe + parallele Komponenten entfernen
    ↓
finale Prediction-Maske
```

## Zentrale Kratzererkennung

Der eigentliche Detektor basiert auf:

```text
response = |Bild - lokal geglätteter Hintergrund|
```

Dadurch können sowohl helle als auch dunkle lokale Kratzerstrukturen hervorgehoben werden.

## Bohrungen

Bohrungen werden separat über `cv2.HoughCircles` gesucht und anschließend über zusätzliche Helligkeitsbedingungen plausibilisiert. Akzeptierte Bereiche werden aus der Scratch-ROI ausgeschlossen.

## Außenkanten

Zunächst wird ein kleiner Randbereich der äußeren Bauteilsilhouette ausgeschlossen.

Zusätzlich werden verbleibende vorhergesagte Komponenten verworfen, wenn sie:

1. überwiegend nahe an der äußeren Kontur liegen,
2. ausreichend lang und länglich sind und
3. annähernd parallel zur lokalen Bauteilkontur verlaufen.

Damit wird nicht jede randnahe Struktur automatisch gelöscht.

---

# 6. Wichtiger Hinweis zur Interpretation

Die Auswertung beschreibt die Leistung des **eingefrorenen Verfahrens auf dem vorliegenden Datensatz**.

Da Teile dieses Datensatzes während der Entwicklung und Parameterwahl betrachtet wurden, sind die resultierenden Kennzahlen keine vollständig unabhängige Test-Set-Schätzung für beliebige zukünftige Bauteile.

Für einen streng unabhängigen Generalisierungstest müsste später ein zusätzlicher, bei der Entwicklung vollständig unangetasteter Bilddatensatz verwendet werden.

Für den Vergleich der untersuchten Verfahren auf demselben definierten Datensatz ist die hier implementierte Auswertung jedoch vollständig reproduzierbar, sofern für alle Verfahren dieselben Ground-Truth-Masken und Bewertungsdefinitionen verwendet werden.
