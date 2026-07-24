# Scratch Annotation Tool

Lokales Annotationstool für die pixelgenaue Segmentierung von Kratzern auf
metallischen Oberflächen. Das Tool nutzt ein vollständig beleuchtetes `_all`-Bild
als Master und überträgt dessen Annotation als bearbeitbare Ausgangskopie auf
alle zugehörigen Slave-Beleuchtungen.

Die abgestimmte fachliche Anforderungsdefinition befindet sich in
[`TOOL_REQUIREMENTS.md`](TOOL_REQUIREMENTS.md).

## Ordnerstruktur

```text
annotation/
├── images/                 # Master- und Slavebilder
├── annotations/            # automatisch erzeugte JSON-Projektdateien
├── masks/                  # binäre PNG-Masken
├── config.yaml             # Leitlinien und Anzeigeeinstellungen
├── scratch_core.py         # Datenmodell und Maskenerzeugung
├── scratch_annotator.py    # Desktopoberfläche
└── TOOL_REQUIREMENTS.md
```

## Dateinamen

Das Masterbild muss auf `_all` enden. Die Anzahl der Slaves ist variabel.

```text
01_all.bmp
01_blue.bmp
01_bottom.bmp
...
```

Alle Bilder einer Reihe müssen dieselbe Auflösung besitzen.

## Installation mit `uv`

Vom Repository-Root aus:

```bash
uv sync
```

Unter Ubuntu muss gegebenenfalls Tkinter als Systempaket installiert werden:

```bash
sudo apt update
sudo apt install python3-tk
```

## Start

```bash
uv run python annotation/scratch_annotator.py
```

Alternativ unter Linux/Ubuntu:

```bash
./start_annotator.sh
```

Beim Start wird standardmäßig `annotation/images/` eingelesen. Über
**„Bilderordner öffnen“** kann ein anderer Ordner gewählt werden.

## Grundbedienung

### Master

1. `_all`-Bild auswählen.
2. Werkzeug **Polyline zeichnen** aktivieren.
3. Punkte entlang der Kratzermittellinie setzen.
4. Mit `Enter` oder Rechtsklick abschließen.
5. Für jeden Kratzer die finale Gesamtbreite individuell einstellen.
6. Orange Ausnahmen prüfen und bei fachlicher Berechtigung akzeptieren.
7. **Bild als fertig markieren**.

### Slave

1. Slave auswählen. Die aktuelle Masterannotation wird automatisch kopiert.
2. Links Slave mit Maske, rechts unverändertes Slavebild vergleichen.
3. Nicht sichtbare Kratzer auswählen und vollständig löschen.
4. Fehlende Kratzer ergänzen.
5. Für technische Korrekturen den rechteckigen Bereichslöscher verwenden.
6. Slave explizit als fertig markieren.

## Wichtige Tastenkürzel

```text
N           Polyline zeichnen
V           Auswählen / Punkte verschieben
E           Bereichslöscher
Enter       aktuelle Polyline abschließen
Rechtsklick aktuelle Polyline abschließen
Esc         aktuelle Aktion abbrechen
Delete      ausgewählten Kratzer löschen
Ctrl + S    JSON und aktuelle Maske speichern
Ctrl + Z    Rückgängig
Ctrl + Y    Wiederholen
A / D       vorheriges / nächstes Bild
Leertaste   Overlay vorübergehend ausblenden
Mausrad     Zoom
Mittlere Maustaste oder Shift+Ziehen  Bildausschnitt verschieben
```

## Leitlinien

Die Standardwerte stehen in `config.yaml`:

```yaml
max_annotation_zoom: 5.0
min_scratch_length_px: 35.0
```

Eine orange Warnung entsteht nur, wenn bei mehr als 5× tatsächlich annotiert
oder verändert wird. Kurze Kratzer werden anhand der Mittellinienlänge in
Originalpixeln geprüft.

## Pixelgenauigkeit

- Originalbilder werden nur gelesen und niemals überschrieben.
- Anzeigezoom verändert die gespeicherten Bilddaten nicht.
- Alle Punkte werden in Originalbildkoordinaten gespeichert.
- Exportmasken besitzen exakt die Originalauflösung.
- PNG-Masken enthalten ausschließlich `0` und `255`.

## Ausgabe

Bearbeitbare Projektdaten:

```text
annotation/annotations/01.json
```

Binäre Masken:

```text
annotation/masks/01_all.png
annotation/masks/01_blue.png
...
```
