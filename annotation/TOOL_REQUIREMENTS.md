# Anforderungsdefinition – Scratch Annotation Tool

## 1. Ziel

Das lokale Python-Tool dient zur pixelgenauen Segmentierungsannotation von
Kratzern auf metallischen Oberflächen. Es ersetzt für diesen spezialisierten
Workflow die allgemeine CVAT-Pipeline.

Die Originalbilder dürfen weder überschrieben noch in ihrer Auflösung
verändert werden. Bildschirmzoom und Anzeige-Skalierung sind ausschließlich
Darstellungsfunktionen. Annotationen und exportierte Masken verwenden immer das
Koordinatensystem und die Auflösung des Originalbildes.

## 2. Aufbau einer Bildreihe

Eine Bildreihe besteht aus:

- genau einem vollständig beleuchteten Masterbild mit dem Suffix `_all`,
- einer variablen Anzahl an Slave-Bildern,
- typischerweise 13 oder 14 Slaves,
- pixelgenau ausgerichteten Aufnahmen desselben Werkstücks mit verschiedenen
  Beleuchtungsszenarien.

Beispiel:

```text
01_all.bmp
01_blue.bmp
01_bottom.bmp
01_left.bmp
...
```

Das Tool erkennt Bildreihen automatisch anhand des Präfixes. Es darf keine
feste Anzahl an Slaves voraussetzen. Alle Bilder einer Reihe müssen exakt
identische Pixelabmessungen besitzen.

## 3. Masterannotation

Das `_all`-Bild ist die Mastervorlage.

- Kratzer werden durch mehrere miteinander verbundene Punkte als Polyline
  entlang ihrer Mittellinie angezeichnet.
- Jeder Kratzer ist ein eigenes Objekt.
- Jeder Kratzer besitzt eine individuell einstellbare Breite.
- Der Breitenwert beschreibt die finale Gesamtbreite der binären Maske in
  Pixeln.
- Die Breite kann vor dem Zeichnen eingestellt und bei ausgewählten Kratzern
  nachträglich geändert werden.
- Die Annotation wird halbtransparent direkt auf dem Originalbild dargestellt.
- Stützpunkte eines ausgewählten Kratzers können verschoben werden.

## 4. Slaveannotation

Beim ersten Öffnen eines Slaves wird die aktuelle Masterannotation als
unabhängige Ausgangskopie übernommen.

Die Slaveansicht besteht aus zwei synchronisierten Darstellungen:

- links: Slavebild mit bearbeitbarer Slaveannotation,
- rechts: unverändertes Slavebild als visuelle Referenz.

Zoom und Bildausschnitt beider Darstellungen sind gekoppelt. Bearbeitet wird nur
die linke Darstellung.

Für jeden Slave können:

- nicht sichtbare übernommene Masterkratzer vollständig deaktiviert werden,
- zusätzliche Kratzer als neue Polylines ergänzt werden,
- Maskenpixel in einem aufgezogenen Rechteck gelöscht werden,
- alle Annotationen des Slaves vollständig geleert werden,
- alle Slaveänderungen verworfen und der aktuelle Masterzustand neu übernommen
  werden.

Der Rechtecklöscher entfernt ausschließlich die Maskenpixel innerhalb des
Rechtecks. Laut Annotationsleitlinie soll ein zusammenhängender Kratzer jedoch
nicht allein wegen lokal schwächerer Sichtbarkeit in Teilstücke aufgeteilt
werden. Der Bereichslöscher dient vor allem der technischen Korrektur.

Jeder neue, noch unbearbeitete Slave startet vom Master. Bereits bearbeitete
Slaves laden ihre eigene gespeicherte Version.

## 5. Master-Sperre

Sobald ein Slave erstmals verändert wird, wird der Master gesperrt. Dadurch
können bereits bearbeitete und später bearbeitete Slaves nicht unbemerkt auf
unterschiedlichen Masterständen basieren.

Der Master kann bewusst entsperrt werden. Bereits bearbeitete Slavekopien
bleiben dabei unverändert. Unbearbeitete Slaves übernehmen beim nächsten Öffnen
den aktualisierten Masterstand.

## 6. Annotationsleitlinien

### 6.1 Maximale Zoomstufe

Die maximal erlaubte Zoomstufe während einer Annotationsänderung beträgt
**50-fach**.

Reines Hineinzoomen zur Kontrolle erzeugt keine Ausnahme. Eine Ausnahme wird
registriert, wenn oberhalb von 5× beispielsweise:

- ein Punkt gesetzt,
- ein Punkt verschoben,
- eine Breite geändert,
- ein Kratzer gelöscht oder
- der Bereichslöscher verwendet wird.

Betroffene Kratzer beziehungsweise das Bild werden orange markiert. Die
Ausnahme muss vor Abschluss des Bildes explizit akzeptiert werden.

### 6.2 Mindestgröße

Die Mindestlänge eines Kratzers beträgt **1 Originalpixel**. Maßgeblich ist die
geometrische Länge der Mittellinie, nicht die Fläche der aufgedickten Maske.

Ein kürzerer Kratzer wird orange markiert und muss explizit als gültiger Kratzer
bestätigt werden. Nach einer Geometrieänderung wird die Bestätigung erneut
geprüft.

## 7. Abschlussstatus

Ein Bild gilt erst nach dem expliziten Befehl **„Bild als fertig markieren“** als
abgeschlossen.

Mögliche Statuswerte:

```text
Nicht begonnen
In Bearbeitung
Fertig
Fertig mit Ausnahmen
```

Ein Bild kann nur abgeschlossen werden, wenn alle Zoom- und
Mindestgrößenausnahmen bewusst akzeptiert wurden.

## 8. Speicherung

### 8.1 Bearbeitbare Annotationen

Pro Bildreihe wird eine JSON-Datei gespeichert:

```text
annotation/annotations/01.json
```

Sie enthält unter anderem:

- Master- und Slave-Polylines,
- Originalpixelkoordinaten,
- individuelle Linienbreiten,
- deaktivierte Masterkratzer,
- Rechtecklöschungen,
- Warnungen und akzeptierte Ausnahmen,
- Bearbeitungsstatus.

Die JSON-Datei wird automatisch und atomar gespeichert, damit die Bearbeitung
nach einem Abbruch fortgesetzt werden kann.

### 8.2 Binäre Masken

Masken werden verlustfrei als PNG gespeichert:

```text
annotation/masks/01_all.png
annotation/masks/01_blue.png
...
```

Es gelten ausschließlich die Werte:

```text
0   = Hintergrund
255 = Kratzer
```

Vor dem Schreiben wird geprüft, dass die Maske exakt dieselbe Breite und Höhe
wie das Originalbild besitzt und keine Zwischenwerte enthält.

## 9. Bedienfunktionen der ersten Version

- automatische Erkennung mehrerer Bildreihen,
- Master-/Slave-Navigation mit Fortschrittsstatus,
- Polyline-Zeichnung und individuelle Kratzerbreite,
- Auswahl und Verschieben vorhandener Stützpunkte,
- vollständiges Löschen ausgewählter Kratzer,
- rechteckiger Bereichslöscher für Slaves,
- Slave leeren und auf Master zurücksetzen,
- synchroner Zoom und synchrones Verschieben,
- Undo/Redo,
- einstellbare Overlay-Deckkraft,
- temporäres Ausblenden des Overlays mit der Leertaste,
- Autosave der bearbeitbaren Daten,
- expliziter Abschluss mit Leitlinienprüfung,
- pixelgenauer binärer Maskenexport.
