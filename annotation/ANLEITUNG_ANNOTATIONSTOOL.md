# Anleitung – Scratch Annotation Tool

## Zweck des Tools

- Lokale, pixelgenaue Segmentierung von Kratzern auf metallischen Oberflächen.
- Verwendung eines vollständig beleuchteten Masterbildes als Vorlage für die zugehörigen Slavebilder.
- Speicherung der bearbeitbaren Annotationen als JSON-Dateien.
- Export der fertigen Segmentierungsmasken als binäre PNG-Dateien.
- Keine Veränderung oder Überschreibung der Originalbilder.
- Keine Herunterskalierung der Bild- oder Maskenauflösung.

## Benötigte Ordner und Dateien

- Repository-Hauptordner:
  - `start_annotator.sh`
  - `pyproject.toml`
  - `uv.lock`
- Ordner `annotation/`:
  - `scratch_annotator.py` – grafische Benutzeroberfläche.
  - `scratch_core.py` – Datenmodell und Maskenerzeugung.
  - `config.yaml` – Grenzwerte, Breitenbereich und Anzeigeeinstellungen.
  - `images/` – Eingabebilder.
  - `annotations/` – automatisch erzeugte, bearbeitbare JSON-Dateien.
  - `masks/` – exportierte binäre PNG-Masken.
  - `TOOL_REQUIREMENTS.md` – fachliche Anforderungsdefinition.

## Einmalige Einrichtung

- Terminal im Repository-Hauptordner öffnen.
- Abhängigkeiten installieren beziehungsweise synchronisieren:

  ```bash
  uv sync
  ```

- Unter Ubuntu bei fehlendem Tkinter zusätzlich ausführen:

  ```bash
  sudo apt update
  sudo apt install python3-tk
  ```

- Die virtuelle Umgebung muss nicht manuell aktiviert werden, wenn das Tool über `uv run` oder `start_annotator.sh` gestartet wird.

## Ablage und Benennung der Bilder

- Standardmäßiger Eingabeordner:
  - `annotation/images/`
- Alternativ kann nach dem Start über **„Bilderordner öffnen“** ein anderer Ordner ausgewählt werden.
- Alle Bilder einer Bildreihe müssen direkt im selben Ordner liegen.
- Unterordner innerhalb des Bilderordners werden nicht durchsucht.
- Unterstützte Bildformate:
  - `.bmp`
  - `.png`
  - `.jpg`
  - `.jpeg`
  - `.tif`
  - `.tiff`
- Jede Bildreihe benötigt genau ein Masterbild mit dem Suffix `_all`.
- Alle Slavebilder derselben Reihe benötigen dasselbe Präfix wie das Masterbild.
- Beispiel:

  ```text
  01_all.bmp
  01_blue.bmp
  01_bottom.bmp
  01_left.bmp
  01_right.bmp
  ```

- Bedeutung der Dateinamen:
  - `01` – Kennung der Bildreihe.
  - `_all` – vollständig beleuchtetes Masterbild.
  - alle weiteren Endungen – Slavebilder mit anderen Beleuchtungsszenarien.
- Die Anzahl der Slavebilder ist variabel.
- Alle Bilder einer Reihe müssen exakt dieselbe Pixelauflösung besitzen.
- Bei mehreren Bildreihen können alle Dateien gemeinsam im Bilderordner abgelegt werden.
- Die Auswahl der erkannten Bildreihe erfolgt anschließend über das Feld **„Bildreihe“**.

## Start des Tools

- Terminal im Repository-Hauptordner öffnen.
- Empfohlener Start unter Linux/Ubuntu:

  ```bash
  ./start_annotator.sh
  ```

- Alternativer Start:

  ```bash
  uv run python annotation/scratch_annotator.py
  ```

- Ein Start aus dem Unterordner `annotation/` ist mit folgendem Befehl möglich:

  ```bash
  uv run python scratch_annotator.py
  ```

- Beim Start wird standardmäßig der Ordner `annotation/images/` eingelesen.
- Das erste erkannte Masterbild wird automatisch geöffnet.

## Aufbau der Benutzeroberfläche

- Obere Werkzeugleiste:
  - **„Bilderordner öffnen“** – Auswahl eines anderen Eingabeordners.
  - **„Bildreihe“** – Wechsel zwischen erkannten Bildreihen.
  - **„Speichern“** – Speicherung der JSON-Projektdaten und Export der Maske des aktuellen Bildes.
  - **„Bild als fertig markieren“** – Abschlussprüfung, Speicherung und Festlegung des Bildstatus.
  - **„Rückgängig“** – Rücknahme des letzten Bearbeitungsschritts im aktuellen Bild.
  - **„Wiederholen“** – Wiederherstellung eines rückgängig gemachten Bearbeitungsschritts.
  - Anzeige des Sperrstatus des Masters.
- Linke Seitenleiste:
  - Liste aller Bilder der ausgewählten Bildreihe.
  - Kennzeichnung als `MASTER` oder `SLAVE`.
  - Anzeige des Bearbeitungsstatus.
- Mittlere linke Bildansicht:
  - aktuelles Master- oder Slavebild.
  - eingeblendete und bearbeitbare Segmentierungsmaske.
  - einzige Ansicht, in der Annotationen verändert werden.
- Mittlere rechte Bildansicht:
  - unverändertes aktuelles Bild ohne Maske.
  - reine Referenzansicht.
  - Zoom und Bildausschnitt sind mit der linken Ansicht gekoppelt.
- Rechter Bedienbereich:
  - Werkzeugauswahl.
  - Kratzerbreite.
  - Overlay-Deckkraft.
  - Informationen zum ausgewählten Kratzer.
  - Lösch- und Ausnahmefunktionen.
  - Slave-Funktionen.
  - Anzeige der aktiven Leitlinien.
  - Entsperrfunktion für den Master.
- Untere Statusleiste:
  - Dateiname des aktuellen Bildes.
  - Bearbeitungsstatus.
  - Anzahl der sichtbaren Kratzer.
  - Anzahl der offenen Warnungen.
  - Hinweise zur Mausbedienung.

## Statusanzeige der Bilder

- `○` – nicht begonnen.
- `◐` – in Bearbeitung.
- `✓` – fertig.
- `⚠` – fertig mit bestätigten Ausnahmen.
- Ein Bild erhält den Status **„Fertig“** erst nach Betätigung von **„Bild als fertig markieren“**.

## Grundlegende Navigation

- Bildauswahl:
  - Klick auf den gewünschten Eintrag in der linken Bildliste.
  - `A` – vorheriges Bild.
  - `D` – nächstes Bild.
- Wechsel der Bildreihe:
  - Auswahl im Feld **„Bildreihe“**.
- Zoom:
  - Mausrad über der linken oder rechten Bildansicht.
  - Beide Ansichten werden synchron gezoomt.
- Verschieben des Bildausschnitts:
  - mittlere Maustaste gedrückt halten und ziehen.
  - alternativ `Shift` gedrückt halten und mit der linken Maustaste ziehen.
  - alternativ Werkzeug **„Ansicht verschieben“** verwenden.
- Overlay kurzzeitig ausblenden:
  - Leertaste gedrückt halten.
  - Beim Loslassen wird das Overlay wieder eingeblendet.
- Overlay dauerhaft transparenter oder deckender darstellen:
  - Regler **„Overlay-Deckkraft“** verwenden.
- Der Anzeigezoom verändert weder Originalbild noch Exportmaske.

## Masterannotation

- Masterbild in der linken Bildliste auswählen.
- Masterbilder sind mit `MASTER` gekennzeichnet und besitzen den Dateinamenszusatz `_all`.
- Werkzeug **„Polyline zeichnen“** auswählen oder Taste `N` verwenden.
- Gewünschte finale Gesamtbreite über den Regler **„Finale Kratzerbreite“** einstellen.
- Punkte nacheinander entlang der sichtbaren Kratzermittellinie setzen.
- Zwischen den gesetzten Punkten wird automatisch eine zusammenhängende Linie erzeugt.
- Zum Abschließen der aktuellen Polyline:
  - Rechtsklick in die linke Bildansicht oder
  - Taste `Enter`.
- Eine Polyline benötigt mindestens zwei Punkte.
- Zum Abbrechen einer noch nicht abgeschlossenen Polyline:
  - Taste `Esc`.
- Jeder abgeschlossene Kratzer wird als separates Objekt mit eigener Breite gespeichert.
- Der Breitenwert entspricht der finalen Gesamtbreite der binären Maske in Originalpixeln.
- Nach Abschluss der Masterannotation:
  - offene Warnungen bearbeiten oder bestätigen.
  - **„Bild als fertig markieren“** auswählen.

## Auswahl und Bearbeitung eines vorhandenen Kratzers

- Werkzeug **„Auswählen / Punkte verschieben“** auswählen oder Taste `V` verwenden.
- Auf den gewünschten Kratzer klicken.
- Im rechten Bereich werden Typ, Länge, Breite und offene Warnungen angezeigt.
- Die Stützpunkte des ausgewählten Kratzers werden eingeblendet.
- Stützpunkt verschieben:
  - gewünschten Stützpunkt mit der linken Maustaste greifen.
  - Punkt an die gewünschte Position ziehen.
  - Maustaste loslassen.
- Breite eines vorhandenen Kratzers ändern:
  - Kratzer auswählen.
  - gewünschten Wert am Breitenregler einstellen.
  - **„Breite auf Auswahl anwenden“** auswählen.
- Vollständigen Kratzer löschen:
  - Kratzer auswählen.
  - **„Ausgewählten Kratzer löschen“** auswählen oder Taste `Delete` verwenden.
- Das Löschen eines übernommenen Masterkratzers in einem Slave betrifft ausschließlich den aktuell geöffneten Slave.
- Das Masterobjekt und die übrigen Slaves bleiben unverändert.

## Slaveannotation

- Gewünschtes Slavebild in der linken Bildliste auswählen.
- Beim ersten Öffnen wird die aktuelle Masterannotation als Ausgangskopie übernommen.
- Linke Ansicht:
  - Slavebild mit bearbeitbarer Slaveannotation.
- Rechte Ansicht:
  - unverändertes Slavebild ohne Annotation.
- Nicht sichtbaren übernommenen Kratzer entfernen:
  - Werkzeug **„Auswählen / Punkte verschieben“** aktivieren.
  - Kratzer auswählen.
  - **„Ausgewählten Kratzer löschen“** oder `Delete` verwenden.
- Zusätzlichen Kratzer ergänzen:
  - Werkzeug **„Polyline zeichnen“** aktivieren.
  - Breite einstellen.
  - Punkte entlang der Mittellinie setzen.
  - Polyline mit Rechtsklick oder `Enter` abschließen.
- Bereits bearbeitete Slaves laden beim erneuten Öffnen ihren eigenen gespeicherten Stand.
- Neue und noch unveränderte Slaves übernehmen beim Öffnen den aktuellen Masterstand.

## Bereichslöscher

- Nur für Slavebilder verfügbar.
- Werkzeug **„Bereichslöscher“** auswählen oder Taste `E` verwenden.
- Mit gedrückter linker Maustaste ein Rechteck über dem zu löschenden Bereich aufziehen.
- Beim Loslassen der Maustaste werden ausschließlich die Maskenpixel innerhalb des Rechtecks entfernt.
- Kratzerbereiche außerhalb des Rechtecks bleiben erhalten.
- Der zugrunde liegende Masterkratzer wird durch den Bereichslöscher nicht aus dem Master gelöscht.
- Der Bereichslöscher ist für lokale Maskenkorrekturen vorgesehen.

## Slave auf Master zurücksetzen

- Nur für Slavebilder verfügbar.
- Funktion **„Slave auf Master zurücksetzen“** auswählen.
- Sicherheitsabfrage bestätigen.
- Wirkung:
  - alle Änderungen des aktuellen Slaves werden verworfen.
  - ergänzte Slavekratzer werden entfernt.
  - gelöschte Masterkratzer werden wiederhergestellt.
  - Rechtecklöschungen werden entfernt.
  - der aktuelle Masterstand wird erneut als Ausgangsbasis verwendet.
- Der Master selbst bleibt unverändert.

## Slave-Maske vollständig leeren

- Nur für Slavebilder verfügbar.
- Funktion **„Slave-Maske vollständig leeren“** auswählen.
- Sicherheitsabfrage bestätigen.
- Wirkung:
  - alle übernommenen Masterkratzer werden für diesen Slave ausgeblendet.
  - alle zusätzlich eingezeichneten Slavekratzer werden entfernt.
  - alle bisherigen Rechtecklöschungen werden zurückgesetzt.
  - die aktuelle Slave-Maske ist vollständig leer.
- Anschließend können bei Bedarf neue Kratzer eingezeichnet werden.
- Der Master und alle übrigen Slaves bleiben unverändert.

## Master-Sperre

- Der Master wird automatisch gesperrt, sobald erstmals eine Änderung an einem Slave vorgenommen wird.
- Die Sperre verhindert unbemerkte Änderungen der gemeinsamen Ausgangsvorlage.
- Ein gesperrter Master kann angezeigt, jedoch nicht bearbeitet werden.
- Zum bewussten Bearbeiten des Masters:
  - **„Master entsperren“** auswählen.
  - Sicherheitsabfrage bestätigen.
- Nach einer Entsperrung:
  - bereits bearbeitete Slavekopien bleiben unverändert.
  - noch unbearbeitete Slaves übernehmen beim nächsten Öffnen den aktualisierten Masterstand.

## Leitlinien und orange Warnungen

- Die wirksamen Grenzwerte werden aus `annotation/config.yaml` geladen.
- Die aktuell verwendeten Werte werden im rechten Bereich der Benutzeroberfläche angezeigt.
- Relevante Einstellungen:
  - `max_annotation_zoom` – maximal zulässiger Zoom während einer Bearbeitung.
  - `min_scratch_length_px` – Mindestlänge der Kratzermittellinie in Originalpixeln.
  - `default_width_px` – voreingestellte Gesamtbreite neuer Kratzer.
  - `min_width_px` – kleinste einstellbare Kratzerbreite.
  - `max_width_px` – größte einstellbare Kratzerbreite.
- Reines Hineinzoomen erzeugt noch keine Ausnahme.
- Eine Zoomwarnung entsteht, wenn oberhalb der eingestellten Zoomgrenze tatsächlich annotiert oder verändert wird.
- Eine Mindestlängenwarnung entsteht, wenn die Länge der Mittellinie unter dem eingestellten Grenzwert liegt.
- Offene Warnungen werden orange dargestellt.
- Ein Bild kann mit offenen Warnungen nicht als fertig markiert werden.

## Ausnahme akzeptieren

- Ausnahme eines einzelnen Kratzers:
  - Werkzeug **„Auswählen / Punkte verschieben“** aktivieren.
  - orange markierten Kratzer auswählen.
  - **„Offene Ausnahme akzeptieren“** auswählen.
- Bildbezogene Ausnahme, beispielsweise nach einer Löschaktion oberhalb der Zoomgrenze:
  - aktive Kratzerauswahl aufheben, beispielsweise durch Klick auf einen freien Bildbereich im Auswahlmodus.
  - **„Offene Ausnahme akzeptieren“** auswählen.
- Nach Bestätigung bleibt die Ausnahme dokumentiert.
- Ein fertiggestelltes Bild mit bestätigter Ausnahme erhält den Status **„Fertig mit Ausnahmen“**.

## Speichern und Abschließen

- Bearbeitbare Projektdaten werden nach Änderungen automatisch als JSON gespeichert.
- Pro Bildreihe wird eine JSON-Datei erzeugt:

  ```text
  annotation/annotations/<Reihen-ID>.json
  ```

- Die JSON-Datei enthält unter anderem:
  - Masterpolylines.
  - Slavekopien.
  - individuelle Breiten.
  - gelöschte Masterkratzer pro Slave.
  - ergänzte Slavekratzer.
  - Rechtecklöschungen.
  - Warnungen und akzeptierte Ausnahmen.
  - Bearbeitungsstatus.
- **„Speichern“** bewirkt:
  - Speicherung der aktuellen JSON-Projektdaten.
  - Erzeugung beziehungsweise Aktualisierung der Maske des aktuell geöffneten Bildes.
- **„Bild als fertig markieren“** bewirkt:
  - Prüfung auf eine noch offene Polyline.
  - Prüfung auf nicht akzeptierte Warnungen.
  - Festlegung des Abschlussstatus.
  - Speicherung der JSON-Projektdaten.
  - Export der Maske des aktuellen Bildes.
- Ein Bildwechsel speichert den bearbeitbaren Projektstand, ersetzt jedoch nicht den bewussten Abschluss des Bildes.
- Vor dem Beenden der Arbeit sollte das aktuelle Bild gespeichert oder als fertig markiert werden.

## Exportierte Masken

- Speicherort:

  ```text
  annotation/masks/
  ```

- Dateiname:
  - identischer Grundname wie das Originalbild.
  - Dateiendung `.png`.
- Beispiel:

  ```text
  Originalbild: annotation/images/01_blue.bmp
  Maske:        annotation/masks/01_blue.png
  ```

- Maskenwerte:
  - `0` – Hintergrund.
  - `255` – Kratzer.
- Die Masken besitzen exakt dieselbe Breite und Höhe wie die Originalbilder.
- Es werden keine Zwischenwerte oder JPEG-Kompressionsartefakte erzeugt.

## Rückgängig und Wiederholen

- **„Rückgängig“** oder `Ctrl + Z`:
  - letzter Bearbeitungsschritt des aktuellen Bildes wird zurückgenommen.
- **„Wiederholen“** oder `Ctrl + Y`:
  - zuletzt rückgängig gemachter Schritt wird erneut angewendet.
- Die Historie gilt für das aktuell geöffnete Bild.
- Beim Wechsel zu einem anderen Bild wird die lokale Undo-/Redo-Historie zurückgesetzt.

## Tastenkürzel

- `N` – Werkzeug **„Polyline zeichnen“**.
- `V` – Werkzeug **„Auswählen / Punkte verschieben“**.
- `E` – Werkzeug **„Bereichslöscher“**.
- `Enter` – aktuelle Polyline abschließen.
- Rechtsklick – aktuelle Polyline abschließen.
- `Esc` – aktuelle Polyline oder Aktion abbrechen.
- `Delete` – ausgewählten Kratzer löschen.
- `Ctrl + S` – aktuelle Projektdaten und Maske speichern.
- `Ctrl + Z` – rückgängig.
- `Ctrl + Y` – wiederholen.
- `A` – vorheriges Bild.
- `D` – nächstes Bild.
- Leertaste gedrückt halten – Overlay temporär ausblenden.
- Mausrad – zoomen.
- Mittlere Maustaste und Ziehen – Bildausschnitt verschieben.
- `Shift` und linke Maustaste ziehen – Bildausschnitt verschieben.

## Empfohlener Arbeitsablauf pro Bildreihe

- Masterbild öffnen.
- Alle sichtbaren Kratzer als einzelne Polylines annotieren.
- Für jeden Kratzer eine passende individuelle Gesamtbreite einstellen.
- Master kontrollieren.
- Offene Warnungen bearbeiten oder fachlich bestätigen.
- Master als fertig markieren.
- Erstes Slavebild öffnen.
- Übernommene Masterannotation mit dem unveränderten Slavebild vergleichen.
- Nicht sichtbare Kratzer vollständig entfernen.
- Fehlende Kratzer ergänzen.
- Lokale Maskenfehler bei Bedarf mit dem Bereichslöscher korrigieren.
- Slave kontrollieren und als fertig markieren.
- Vorgang für alle weiteren Slaves wiederholen.
- Nach Abschluss prüfen, ob alle Bilder der Reihe den Status `✓` oder `⚠` besitzen.
- Vor Weitergabe der Ergebnisse die zugehörige JSON-Datei und alle Maskendateien gemeinsam sichern.

## Wichtige Hinweise zur Datensicherheit

- Originalbilder werden durch das Tool nur gelesen.
- Originalbilder werden nicht überschrieben.
- Bearbeitbare JSON-Dateien dürfen nicht gelöscht werden, solange spätere Änderungen an den Annotationen möglich bleiben sollen.
- PNG-Masken können aus der zugehörigen JSON-Datei und den Originalbildern erneut erzeugt werden, sollten für den Datensatzexport dennoch gemeinsam gesichert werden.
- Die Dateien einer begonnenen Bildreihe sollten nicht umbenannt werden.
- Master- und Slavebilder einer begonnenen Bildreihe sollten nicht zwischen verschiedenen Bilderordnern verschoben werden.
- Die Werte in `config.yaml` sollten innerhalb eines laufenden Annotationsprojekts nicht ohne abgestimmte Entscheidung geändert werden.
