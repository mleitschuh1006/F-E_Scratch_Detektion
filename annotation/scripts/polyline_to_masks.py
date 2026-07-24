ChatGPT Plus

heute 12:31
01_right(2).bmp
Datei
01_left(2).bmp
Datei
01_bottom(2).bmp
Datei
01_top(2).bmp
Datei
01_all(2).bmp
Datei
Ich habe viele von diesen aufnahmen von metallischen oberflächen, auf welchen ich kratzer detektieren möchte. Um diese Kratzer zu detektieren muss ich nun natürlich erstmal die daten annotieren und alle kratzer semantisch segmentieren bzw für jedes bild eine segmentierungsmasker erstellen. 
Daher ist nun die frage mit welchem tool ich dies am bestne und effizientesten tue um nicht extrem viel zeit damit zu versschwendne. Es sind insgesamt 110 bildreihen und jede bildreihe enthält 14 unterschiedliche Belichtungsszenarien und zu jedem Belichtungsszenario wurde ein  bild gemacht. Es sind also 110x14 Bilder welche zu annotieren sind.
Wir sind eine Vierergruppe auf die wir die bilder uns aufteilen möchten. Wir benötigen daher bevor wir mit der annotation starten eine genaue Definition/Leitlinie wie und was als kratzer annotiert wird.
Außerdem benötigen wir eine möglichst effiziente pipeline für die datenannotation. Meine idee ist es, dass wir die bildreihen durch vier teilen und jeder eine gesamte Bildreihe komplett immer macht. Da die bildreihen ja die selbe oberfläche darstellen nur mit unterschiedlichen belichtungsszenarien sollte man doch einfach das belichtungsszenario all händisch nur annotieren müssen udn die anderen könnte man dann doch mit hilfe eines tresholds aus der annotation von all ableiten. Also das dort wo der pixelwert beispielsweise kleiner 10 ist (Schwarz) die annotationen verworfen werden. Dann annotiert man nicht jeden kratzer 14 mal in einer Bilderreihe.
DAs sind grob mein aktueller standpunkt und ideen. Bitte gebe mir nun einen passenden sehr guten Vorschlag, für die Annotationspipeline, also wie welches tool, mit welchen leitlinien etc. ICh hatte schonmla von dem Tool Computer Vision annotation tool gehört, da hugging face vermutlich zu klein ist bei der kostenlosen variante, da es 18GB an daten sind und wir sie auch nicht unbedingt vollständig öffentlich machen wollen

Meine klare Empfehlung

Für euren Datensatz würde ich CVAT selbst hosten und nicht 1.540 Bilder einzeln annotieren. Die eigentliche Annotationseinheit sollte eine physische Oberfläche beziehungsweise eine Bildreihe sein. Damit benötigt ihr zunächst nur 110 Ground-Truth-Masken.

CVAT passt hier sehr gut, weil es lokal per Docker betrieben werden kann, Masken mit einem Brush-Werkzeug unterstützt, Aufgaben verschiedenen Personen zugewiesen und anschließend geprüft werden können und sich semantische Segmentierungsmasken als PNG-basierte Formate exportieren lassen.

Die 18 GB Rohdaten müssen dabei nicht vollständig in irgendeinen Cloud-Dienst oder zu Hugging Face hochgeladen werden. Der CVAT-Server kann auf einem Hochschulrechner, einem lokalen Server oder einem leistungsfähigen Rechner aus eurer Gruppe laufen. Die anderen greifen beispielsweise über Hochschul-VPN, Tailscale oder WireGuard darauf zu.
Wichtigste Korrektur an deiner bisherigen Idee
Einmal annotieren: ja
Über einen Helligkeitsschwellwert übertragen: eher nein

Bei den fünf hochgeladenen Beispielbildern sieht es so aus, als wären Kamera und Metallplatte nahezu unverändert positioniert. Die Bohrungen, Plattenkanten und Strukturen liegen weitgehend an denselben Stellen. Deshalb ist es grundsätzlich sinnvoll, eine Maske nur einmal zu erstellen und auf die anderen Belichtungen zu übertragen.

Ein Schwellwert wie

Pixelwert < 10 → Scratch-Annotation löschen

wäre aber problematisch. In euren Bildern bedeutet ein dunkler Pixel meistens:

    geringe Beleuchtung,

    Schatten,

    ungünstiger Reflexionswinkel,

    aber nicht „an dieser Stelle befindet sich kein Kratzer“.

Gerade die Richtungsbeleuchtungen sind in großen Bildbereichen sehr dunkel. Mit dem Schwellwert würdet ihr deshalb echte Kratzer aus der Ground Truth entfernen. Derselbe physische Kratzer hätte dann je nach Beleuchtung unterschiedliche Masken. Das erzeugt inkonsistente Labels.

Besser ist die Trennung zwischen:

scratch_mask        = physischer Kratzer
validity_mask       = Bereich, der in diesem Bild sinnvoll auswertbar ist

Ein schwarzer oder vollständig gesättigter Bereich kann über die validity_mask beim Training ignoriert werden. Die eigentliche Kratzermaske bleibt aber geometrisch gleich.
Zuerst müsst ihr entscheiden, was euer späteres Modell als Eingang erhält
Variante A: Das Modell erhält alle 14 Belichtungen

Falls die spätere Anlage ebenfalls 14 Bilder derselben Oberfläche aufnehmen kann, würde ich die Reihe als einen einzigen Datenpunkt behandeln:

Input: 14 Belichtungsbilder derselben Oberfläche
Target: eine gemeinsame Kratzermaske

Dann habt ihr nicht 1.540 unabhängige Trainingsbeispiele, sondern:

110 Oberflächen
110 Masken
jeweils 14 Eingangsbilder

Das Modell kann beispielsweise mit einem 14-kanaligen Eingang, mehreren parallelen Encodern oder einem vorher erzeugten Beleuchtungs-Composite arbeiten.

Das ist aus meiner Sicht die sauberste Lösung, weil die verschiedenen Beleuchtungen gerade dazu dienen, dieselben Kratzer besser sichtbar zu machen.
Variante B: Das Modell erhält später nur ein einzelnes Bild

Dann wird dieselbe Maske nach der Annotation auf alle 14 Belichtungsbilder übertragen:

S001_light_01.png → S001_mask.png
S001_light_02.png → S001_mask.png
...
S001_light_14.png → S001_mask.png

Vorher müsst ihr allerdings sicherstellen, dass die Bilder wirklich pixelgenau ausgerichtet sind. Bei Verschiebungen wird die Maske mit derselben geometrischen Transformation auf das jeweilige Bild übertragen. Für Masken muss dabei immer Nearest-Neighbor-Interpolation verwendet werden, damit keine Zwischenwerte entstehen.

Wichtig: Beim späteren Train-/Validation-/Test-Split müssen alle 14 Bilder einer Oberfläche im selben Split bleiben. Sonst sieht das Modell im Training nahezu dieselbe Oberfläche wie im Test und eure Ergebnisse wären künstlich zu gut.
Empfohlene Annotationspipeline
1. Daten standardisieren und Ausrichtung prüfen

Die Ordner und Dateien sollten eindeutig benannt werden:

series_001/
    light_01.bmp
    light_02.bmp
    ...
    light_14.bmp

series_002/
    light_01.bmp
    ...

Anschließend prüft ein Skript für jede Reihe:

    Haben alle Bilder dieselbe Auflösung?

    Sind Bohrungen und Plattenkanten an derselben Position?

    Gibt es Verschiebungen, Drehungen oder Perspektivänderungen?

    Sind Bilder beschädigt, schwarz oder überbelichtet?

    Fehlt eine Beleuchtung?

Bei dünnen Kratzern kann bereits eine Verschiebung von ein bis zwei Pixeln relevant sein. Falls nötig, werden alle Bilder einer Reihe auf ein Referenzbild registriert, beispielsweise mit Translation, ECC oder Homografie.

Zusätzlich erstellt ihr eine feste ROI-Maske für den gültigen Metallbereich. Ausgeschlossen werden:

    schwarzer Bildrand,

    Bereiche außerhalb der Platte,

    Bohrungen,

    Halterungen,

    dauerhaft nicht zu untersuchende Randzonen.

Für diese ROI-Maske kann Bildverarbeitung beziehungsweise Thresholding sinnvoll sein. Für das Entfernen von Kratzern aus der Maske dagegen nicht.
2. Ein „Annotation Master Image“ pro Reihe erzeugen

Ich würde nicht ausschließlich das Bild all annotieren. In euren Beispielen zeigt all zwar viele Strukturen, bestimmte Kratzer treten aber unter gerichteter Beleuchtung deutlich stärker hervor.

Deshalb erzeugt ihr für jede Reihe automatisiert ein optimiertes Referenzbild. Sinnvolle Bestandteile wären:

Beleuchtungsdifferenz:
range(x,y) = max(I1 ... I14) - min(I1 ... I14)

Lokaler Kontrast:
local(x,y) = max |Ii - GaussianBlur(Ii)|

All-Beleuchtung:
normalisierte Version des all-Bildes

Daraus kann ein dreikanaliges Composite erzeugt werden:

Rot:   Beleuchtungsdifferenz
Grün:  maximaler lokaler Kontrast
Blau:  All-Beleuchtung

Dieses Bild dient nur zur Annotation. Trainiert wird später weiterhin mit den Originalbildern oder dem definierten Multi-Light-Input.
Originalbelichtungen als Kontext anzeigen

CVAT unterstützt sogenannte Contextual Images. Damit kann ein primäres Bild annotiert werden, während zusätzliche Ansichten derselben Szene daneben angezeigt werden. CVAT unterstützt dabei bis zu zwölf Kontextbilder.

Da ihr 14 Belichtungen habt, könnt ihr beispielsweise:

    das Composite als Hauptbild verwenden,

    sieben Kontextbilder erzeugen,

    wobei jedes Kontextbild zwei Belichtungen nebeneinander zeigt.

So kann der Annotator jederzeit prüfen, ob eine Linie tatsächlich ein Kratzer oder lediglich eine Reflexion ist.
3. CVAT-Projekt einrichten

Ich würde ein Projekt mit diesen Labels anlegen:
Label	Bedeutung
scratch	sicherer Kratzer
uncertain	möglicher Kratzer, nicht sicher entscheidbar
abrasion	optional: breite Scheuer- oder Schleifstelle

uncertain wird beim späteren Export nicht als eigene Trainingsklasse verwendet, sondern in einen Ignore-Wert umgewandelt, beispielsweise:

0   = Hintergrund
1   = Kratzer
255 = Ignore / unsicher / ungültig

Für semantische Segmentierung würde ich primär das Brush-Maskenwerkzeug verwenden. Polygone sind für lange, sehr dünne und unregelmäßige Kratzer meistens umständlicher. CVAT kann Masken direkt bearbeiten und als Segmentation-Mask- beziehungsweise CamVid-Format exportieren.

Die vollständige Leitlinie solltet ihr direkt als CVAT-Guide hinterlegen. CVAT besitzt dafür einen Markdown-Editor, dessen Anleitung im Projekt oder in einer einzelnen Aufgabe angezeigt werden kann.
Konkrete Annotationsleitlinie

Folgende Definition würde ich als erste Version verwenden.
Was wird als Kratzer annotiert?

Ein Kratzer ist eine permanente, überwiegend lineare oder kurvenförmige Oberflächenveränderung, die sich geometrisch von der regulären Oberflächenstruktur unterscheidet.

Ein Bereich wird als Kratzer annotiert, wenn mindestens eines gilt:

    Die Struktur ist in mindestens zwei unterschiedlichen Beleuchtungen an derselben Position erkennbar.

    Die Struktur ist in einer Beleuchtung eindeutig sichtbar und besitzt einen klaren, kontinuierlichen Verlauf.

    Die Struktur ist im Composite eindeutig als lokale, linienförmige Oberflächenveränderung erkennbar.

Annotiert wird die sichtbare beschädigte Fläche, nicht nur die ungefähre Mittellinie.
Nicht als Kratzer annotieren

Nicht annotiert werden:

    Reflexionen und Beleuchtungsgradienten,

    Schatten,

    Fingerabdrücke,

    Staub und lose Verschmutzungen,

    Bohrungen und deren Ränder,

    Plattenkanten,

    gleichmäßige Fertigungs- oder Bürststrukturen,

    großflächige Verfärbungen,

    Kamerarauschen,

    Strukturen, die nur durch Über- oder Unterbelichtung entstehen.

Breite Scheuerstellen müssen vor Projektbeginn eindeutig behandelt werden: entweder als eigene Klasse abrasion oder bewusst gar nicht. Sie dürfen nicht von manchen Annotatoren als Kratzer und von anderen als Hintergrund behandelt werden.
Grenzen eines Kratzers

    Die Maske soll dem tatsächlichen sichtbaren Verlauf folgen.

    Bei dünnen Kratzern wird stark hineingezoomt.

    Der Pinsel wird an die sichtbare Breite angepasst.

    Die Maske soll nicht pauschal breiter gemacht werden, nur damit sie leichter zu zeichnen ist.

    Endpunkte werden dort gesetzt, wo der Kratzer nicht mehr verlässlich erkennbar ist.

Unterbrochene Kratzer

Kurze Lücken dürfen verbunden werden, wenn:

    der Verlauf geometrisch eindeutig ist und

    mindestens eine andere Beleuchtung die Verbindung bestätigt.

Ist die Verbindung nicht sicher, werden zwei getrennte Bereiche annotiert oder die Lücke als uncertain markiert.
Kreuzungen

Da ihr semantische und keine Instanzsegmentierung benötigt, werden sich kreuzende Kratzer einfach als gemeinsame Kratzermaske annotiert. Es muss nicht nachvollzogen werden, welcher Pixel zu welchem einzelnen Kratzer gehört.
Sehr kleine Kratzer

Die Mindestgröße sollte nicht willkürlich in Pixeln bestimmt werden, sondern aus eurer technischen Anforderung:

Mindestlänge in Pixeln
= relevante Mindestlänge in mm / mm pro Pixel

Falls es noch keine technische Mindestgröße gibt, würde ich zunächst alle Kratzer annotieren, die bei nativer Auflösung verlässlich identifizierbar sind. Nach dem Pilotdurchlauf könnt ihr eine objektive Mindestgröße festlegen.
Unsichere Bereiche

Unsichere Stellen dürfen niemals einfach nach Bauchgefühl als Hintergrund markiert werden. Sie werden als uncertain beziehungsweise ignore gekennzeichnet und beim Training aus der Loss-Berechnung ausgeschlossen.
Aufteilung in eurer Vierergruppe

Deine Idee, vollständige Reihen einer Person zuzuweisen, ist richtig:

Person A: 28 Bildreihen
Person B: 28 Bildreihen
Person C: 27 Bildreihen
Person D: 27 Bildreihen

Aber ihr solltet nicht sofort mit dieser Aufteilung starten.
Pilotphase

Zuerst annotieren alle vier unabhängig dieselben fünf bis acht repräsentativen Reihen:

    wenige Kratzer,

    viele Kratzer,

    sehr dünne Kratzer,

    breite Beschädigungen,

    starke Reflexionen,

    schwierige dunkle Bereiche.

Danach vergleicht ihr die Masken gemeinsam. Gerade bei dünnen Linien ist normales IoU sehr streng, weil eine Verschiebung um einen Pixel bereits stark bestraft wird. Sinnvoll sind zusätzlich:

    Dice Score,

    Boundary-F1 mit beispielsweise zwei Pixeln Toleranz,

    Anteil übereinstimmend gefundener Kratzer,

    Vergleich der markierten Endpunkte.

Alle strittigen Fälle werden als Beispielbilder direkt in die Leitlinie aufgenommen. Erst wenn diese Pilotmasken ausreichend konsistent sind, beginnt die eigentliche Aufteilung.
Produktions- und Reviewphase

CVAT besitzt Job-Stufen wie Annotation, Validation und Acceptance und erlaubt die Zuweisung an Annotatoren beziehungsweise Reviewer. Außerdem können Reviewer konkrete Probleme markieren und zur Korrektur zurückgeben.

Eine sinnvolle Rotation wäre:

A prüft B
B prüft C
C prüft D
D prüft A

Ich würde:

    alle 110 Masken kurz gegenprüfen,

    mindestens 20 % detailliert prüfen,

    alle uncertain-Fälle detailliert prüfen,

    nach jeweils etwa zehn Reihen eine Zwischenbesprechung durchführen.

Wird die Leitlinie später geändert, müssen bereits annotierte Reihen auf die neue Regel hin kontrolliert werden. Deshalb sollte jede Maske eine guideline_version erhalten.
Automatisierte Übertragung der Masken

Nach dem Export aus CVAT gibt es pro Bildreihe eine Maske:

S001_mask.png
S002_mask.png
...
S110_mask.png

Bei einem Einzelbildmodell erzeugt ein Skript daraus:

S001_light_01_mask.png
S001_light_02_mask.png
...
S001_light_14_mask.png

Falls Registrierungen notwendig waren:

mask_i = warp(mask_master, transformation_i)

Dabei gilt:

    nur Nearest-Neighbor-Interpolation,

    anschließend nur Werte 0, 1 und 255,

    visuelle Stichprobenkontrolle,

    niemals Multiplikation mit einem einfachen Helligkeitsschwellwert.

Für vollständig schwarze, gesättigte oder außerhalb der ROI liegende Bereiche wird separat eine Ignore-Maske erstellt.
Modellgestützte zweite Annotationsphase

Nach den ersten ungefähr 20 bis 30 sauber annotierten Reihen könnt ihr einen kleinen U-Net-, DeepLab- oder YOLO-Segmentierungsbaseline trainieren. Dessen Vorhersagen werden für die restlichen Reihen als Vorannotation verwendet und anschließend nur noch korrigiert.

CVAT unterstützt automatische beziehungsweise modellgestützte Annotationen und kann auch selbst gehostete Modelle einbinden oder Vorhersagen als vorhandene Annotationen importieren.

SAM würde ich bei euren Daten lediglich testen, aber nicht als zentrale Lösung einplanen. Die Kratzer sind teilweise extrem dünn, kontrastarm und nicht wie klar abgegrenzte Alltagsobjekte aufgebaut. Ein eigenes Modell, das bereits auf 20 bis 30 eurer Oberflächen trainiert wurde, dürfte für die weitere Vorannotation passender sein.
Empfohlene Gesamtarchitektur

14 Rohbilder einer Oberfläche
          │
          ├── Registrierungsprüfung
          ├── ROI-Maske
          └── Erzeugung Annotation-Composite
                         │
                         ▼
                 CVAT-Annotation
                eine Maske pro Reihe
                         │
                 Review und Korrektur
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
   Multi-Light-Modell          Einzelbildmodell
  14 Bilder + 1 Maske       Maske auf 14 Bilder

Damit reduziert ihr den manuellen Aufwand von 1.540 Einzelannotationen auf 110 Annotationen, ohne inkonsistente belichtungsabhängige Ground Truth zu erzeugen. CVAT selbst gehostet, ein Composite als Hauptbild, die Belichtungen als Kontextansichten und ein klar definierter Review-Prozess wäre für euren Fall die sinnvollste Pipeline.

okay ich habe mir nun cvat heruntergeladen und mir die folgende ordnerstruktur in einem github repro aufgebaut. Die idee ist es dass wir uns dann die bildreihen dann halt einfach aufteilen udn jeder auf seinem eigenen branch seine bilder annotiert. Ich brauche nun das essentielle dokument für die leitlinien der annotation und insgesamt der vorgang der annotationspipeline..
Ich würde es dabei gerne so versuchen zunächst erstmal nur das all bild händisch perfekt zu annotieren und dann auf die anderen bildreihen zu adaptierne. Also das ich dann mittels eines programms schonmal für alle anderen bidlreihen die segmenteirungsmaske erstelle aufgrund der schwarzen bereiche welceh weggelassenw erden o.ä. und dann geht man halt nochmal schneller über die bilder drüber und schaut ob es so passt

Bitt erstelle mir die entsprechenden DOkumente, programme für unser Repro

Fertiges Paket für euer Repository

Ich habe die vollständige Annotationspipeline passend zu deiner vorhandenen annotation/-Struktur erstellt:

Die wichtigsten Dateien einzeln:

Die Skripte wurden syntaktisch geprüft und einmal vollständig mit deinen hochgeladenen Beispielbildern durchlaufen: Validierung → Maskenübertragung → Overlays → CVAT-ZIP → erneutes Extrahieren.
Enthaltene Pipeline

all-Bild in CVAT annotieren
          ↓
CVAT Segmentation-Mask-Export
          ↓
Mastermasken extrahieren
          ↓
Maske auf alle 14 Belichtungen übertragen
          ↓
Validitäts- und Ignore-Masken erzeugen
          ↓
Review-Overlays erzeugen
          ↓
Vorannotationen als CVAT-ZIP importieren
          ↓
13 Bilder kontrollieren und korrigieren
          ↓
finale Masken exportieren

CVAT unterstützt den Import und Export von PNG-Segmentierungsmasken über ein ZIP-Archiv mit labelmap.txt, ImageSets und den Maskenordnern. Wichtig ist, dass beim Hochladen von Annotationen bereits vorhandene Annotationen im Task ersetzt werden. Das erzeugte Import-ZIP enthält deshalb auch die manuelle all-Maske.
Umgang mit schwarzen Bereichen

Ich habe deine Idee eingebaut, aber sicherer umgesetzt.

Bei deinen Beispielaufnahmen besitzen top und bottom jeweils über 50 % Pixel mit einem Grauwert von höchstens 10. Ein einfaches

mask[image < 10] = 0

würde daher große Teile realer Kratzer entfernen.

Standardmäßig entstehen deshalb drei unterschiedliche Dateien:

annotations_auto/
    binäre Scratch-Vorannotation: 0 oder 255

validity_masks/
    255 = Bildbereich auswertbar
    0   = dunkel/hell und lokal strukturlos

training_masks/
    0   = Hintergrund
    1   = Kratzer
    255 = Ignore

Die physische Kratzermaske bleibt somit erhalten, aber das Modell wird in vollständig unbrauchbaren Bildbereichen nicht bestraft.

Deine ursprüngliche Variante lässt sich dennoch testen. In config.yaml:

processing:
  clip_auto_masks_to_validity: true

Dann werden Scratch-Pixel in automatisch erkannten ungültigen Bereichen aus der Vorannotation gelöscht. Diese Variante muss anschließend vollständig in CVAT kontrolliert werden.
Zusätzliche Ordner

Gegenüber deiner bisherigen Struktur kommen diese Ordner hinzu:

data/
├── annotations_manual/   # manuell annotierte all-Masken
├── annotations_auto/     # automatisch übertragene Masken
├── annotations_final/    # nach Review freigegebene Masken
├── validity_masks/       # gültige Bildbereiche
├── training_masks/       # fertige 0/1/255-Trainingsmasken
├── review_overlays/      # schnelle visuelle Kontrolle
├── cvat_exports/
└── cvat_imports/

Damit werden die manuelle Annotation, die automatisch erzeugte Vorannotation und die tatsächlich freigegebene Ground Truth nicht miteinander vermischt.
Erste Schritte

Entpacke den Inhalt in dein Repository und verwende möglichst einheitliche Bildnamen:

S001_all.bmp
S001_top.bmp
S001_bottom.bmp
S001_left.bmp
S001_right.bmp
S001_light_05.bmp
...

Danach:

uv pip install -r annotation/requirements.txt

Datensatz prüfen:

uv run python annotation/scripts/validate_dataset.py \
  --config annotation/config.yaml \
  --stage images \
  --strict

Die 110 Serien auf vier Personen verteilen:

uv run python annotation/scripts/create_assignments.py \
  --config annotation/config.yaml \
  --annotators Marlon Person_B Person_C Person_D \
  --output annotation/data/assignments.csv

Nach der manuellen CVAT-Annotation der all-Bilder:

uv run python annotation/scripts/extract_cvat_masks.py \
  --config annotation/config.yaml \
  --zip annotation/data/cvat_exports/manual_export.zip \
  --destination manual \
  --reference-only

Anschließend propagieren:

uv run python annotation/scripts/mask_propagation.py \
  --config annotation/config.yaml

Review-Overlays erzeugen:

uv run python annotation/scripts/create_review_overlays.py \
  --config annotation/config.yaml \
  --mask-source auto

CVAT-Import-ZIP erstellen:

uv run python annotation/scripts/build_cvat_import.py \
  --config annotation/config.yaml \
  --mask-source auto \
  --output annotation/data/cvat_imports/auto_masks.zip

Git-Branches

Eure geplante Branch-Aufteilung ist sinnvoll, solange jede Person ausschließlich ihre zugeordneten Serien bearbeitet:

annotation/marlon
annotation/person-b
annotation/person-c
annotation/person-d

Die 18 GB Rohbilder sollten allerdings nicht als normale Git-Dateien in das Repository aufgenommen werden. Die beigefügte .gitignore ignoriert deshalb Rohbilder, Overlays und automatisch regenerierbare Masken. Für große Binärdateien empfiehlt GitHub Git LFS; alternativ könnt ihr die Rohbilder über einen gemeinsamen Drive-/Netzwerkordner bereitstellen und nur Code, Leitlinien sowie manuelle und finale Masken versionieren.

Die eigentliche Leitlinie enthält bereits Definitionen für Kratzer, Ausschlusskriterien, Grenzfälle, Pilotphase, Zweitreview, Abnahmekriterien und die wichtige Regel, dass alle 14 Bilder einer Reihe immer im selben Train-/Validation-/Test-Split bleiben.

okay ich habe jetzt mal den ersten versuch zum annotrierne versucht. 
Aber ich bin mir unsicher ob das wirklich so gut ist, da es ja schon sehr ungenau so ist

puh aber ist das dann wirklich genau genug und ich brauche doch jetzt ewigkeiten dafür oder geht das? Wenn du dir jetzt mal das beispielbild mit schon paar segmentierten kratzern anschaust

Da deine Testannotation aktuell aus Polylines besteht, brauchst du zunächst einen zusätzlichen Schritt:

CVAT-Polyline
    ↓
Polyline-Koordinaten exportieren
    ↓
automatisch auf feste Breite bringen
    ↓
binäre PNG-Maske erzeugen
    ↓
Maske auf die übrigen Belichtungen übertragen

Die bisherige Pipeline mit dem direkten Export als Segmentation Mask funktioniert nicht direkt für Polylines, weil CVAT in diesem Format nur Masken, Polygone, Ellipsen und Bounding Boxes rasterisiert – keine Polylines. Polylines solltest du deshalb zunächst im nativen Format CVAT for images exportieren.
1. Zuerst die Dateinamen prüfen

Alle 14 Bilder einer Reihe sollten einheitlich benannt sein, beispielsweise:

S001_all.bmp
S001_top.bmp
S001_bottom.bmp
S001_left.bmp
S001_right.bmp
S001_light_05.bmp
...

Wichtig ist:

    S001 bezeichnet immer dieselbe Metalloberfläche.

    all ist das manuell annotierte Masterbild.

    Alle Bilder haben dieselbe Auflösung.

    Kamera und Metallplatte haben sich zwischen den Aufnahmen nicht bewegt.

Die Bilder gehören nach:

annotation/data/images/

2. Testannotation in CVAT speichern

Speichere deinen CVAT-Job zunächst ganz normal über Save.

Kontrolliere dabei:

    Alle Polylines haben exakt das Label scratch.

    Jede Polyline liegt ungefähr in der Mitte des Kratzers.

    Abzweigungen sind eigene Polylines.

    Es gibt keine versehentlich gesetzten kurzen Linien oder Punkte.

    Du hast möglichst wenige, aber ausreichend viele Stützpunkte benutzt.

3. Annotation aus CVAT exportieren

Öffne den entsprechenden Task und wähle:

Actions
→ Export task dataset
→ CVAT for images 1.1

Die Bilder musst du nicht noch einmal mit exportieren.

CVAT erzeugt ein ZIP-Archiv, zum Beispiel:

scratch_test_cvat_for_images.zip

Lege dieses Archiv hier ab:

annotation/data/cvat_exports/

Also beispielsweise:

annotation/data/cvat_exports/scratch_test_cvat_for_images.zip

Das Archiv enthält eine Datei namens:

annotations.xml

Darin stehen die Punktkoordinaten deiner Polylines.
4. Polyline-Breite festlegen

Bevor die Maske erzeugt wird, musst du definieren, wie breit eine Polyline rasterisiert werden soll.

Für den ersten Test würde ich folgende Werte ausprobieren:

3 Pixel
5 Pixel
7 Pixel

Als Ausgangswert empfehle ich:

5 Pixel Gesamtbreite

Das bedeutet ungefähr:

2 Pixel links
1 Pixel Mittellinie
2 Pixel rechts

Die Breite sollte zunächst bei allen dünnen Kratzern gleich sein. So verhindert ihr, dass jede Person nach eigenem Gefühl unterschiedlich breite Masken erzeugt.
5. Polyline in PNG-Maske umwandeln

Für deinen derzeitigen Workflow fehlt im bisherigen Paket noch ein Polyline-Konverter. Er sollte beispielsweise so aufgerufen werden:

uv run python annotation/scripts/polylines_to_masks.py \
  --config annotation/config.yaml \
  --zip annotation/data/cvat_exports/scratch_test_cvat_for_images.zip \
  --line-width 5

Das Skript muss:

    annotations.xml aus dem CVAT-ZIP lesen,

    das Bild S001_all.bmp suchen,

    alle Polylines mit dem Label scratch auslesen,

    die Punkte mit Linien verbinden,

    eine binäre PNG-Maske erzeugen,

    diese als S001_all.png speichern.

Das Ergebnis gehört nach:

annotation/data/annotations_manual/S001_all.png

Die Maske enthält nur:

0   = Hintergrund
255 = Kratzer

So sollte die erzeugte Maske aussehen

Öffnest du die PNG-Datei separat, sollte sie folgendermaßen aussehen:

    schwarzer Hintergrund,

    weiße Linien entlang der annotierten Kratzer,

    exakt dieselbe Bildgröße wie das Originalbild,

    keine grauen Zwischenwerte.

Nicht korrekt wären beispielsweise:

0, 13, 128, 200, 255

Die Maske darf nur 0 und 255 enthalten.
6. Mastermaske kontrollieren

Bevor du die Maske auf 13 weitere Bilder kopierst, solltest du ein Overlay erzeugen.

Prinzipiell:

overlay = originalbild + halbtransparente_maske

Dadurch siehst du direkt:

    liegt die Linie mittig auf dem Kratzer?

    ist die Maskenbreite passend?

    wurden Kratzer vergessen?

    wurden Reflexionen fälschlich annotiert?

    enden die Linien an der richtigen Stelle?

Für diesen Test solltest du drei Varianten erzeugen:

S001_all_width_3.png
S001_all_width_5.png
S001_all_width_7.png

Danach vergleichst du die Overlays.

Meine Erwartung bei deinem gezeigten Bild:

    3 Pixel könnten für deutlich sichtbare breite Kratzer zu schmal sein,

    7 Pixel könnten bei den feinen Linien zu viel Hintergrund einschließen,

    5 Pixel ist ein vernünftiger Startwert.

7. Datensatzstruktur prüfen

Sobald die Mastermaske vorhanden ist:

uv run python annotation/scripts/validate_dataset.py \
  --config annotation/config.yaml \
  --stage manual

Die Prüfung sollte bestätigen:

    Originalbild wurde gefunden,

    Maske wurde gefunden,

    Abmessungen stimmen überein,

    Maske ist binär,

    Zuordnung zu S001_all funktioniert.

8. Maske auf alle Belichtungen übertragen

Danach startest du die vorhandene Übertragung:

uv run python annotation/scripts/mask_propagation.py \
  --config annotation/config.yaml \
  --series S001

Das Skript liest:

annotation/data/annotations_manual/S001_all.png

und erzeugt Masken für alle Bilder der Reihe:

annotation/data/annotations_auto/S001_all.png
annotation/data/annotations_auto/S001_top.png
annotation/data/annotations_auto/S001_bottom.png
annotation/data/annotations_auto/S001_left.png
annotation/data/annotations_auto/S001_right.png
...

Da alle Bilder dieselbe Oberfläche darstellen, wird zunächst dieselbe geometrische Kratzermaske verwendet.
9. Schwarze Bildbereiche behandeln

Hier würde ich vorerst bei der sicheren Einstellung bleiben:

processing:
  clip_auto_masks_to_validity: false

Das bedeutet:

    Ein Kratzer wird nicht gelöscht, nur weil er in einer Beleuchtung dunkel liegt.

    Zusätzlich wird eine Validitätsmaske erzeugt.

    Vollständig schwarze oder strukturarme Bildbereiche können beim Training später ignoriert werden.

Das Skript erzeugt daher drei verschiedene Ergebnisse:

annotations_auto/

Binäre Kratzermasken:

0   = Hintergrund
255 = Kratzer

validity_masks/

Gültigkeit der Bildinformation:

0   = unbrauchbarer oder strukturarmer Bereich
255 = auswertbarer Bereich

training_masks/

Trainingsmaske:

0   = Hintergrund
1   = Kratzer
255 = Ignore

Bitte lösche Kratzer nicht direkt nur aufgrund eines Schwellwerts wie Pixel < 10. Ein dunkler Pixel bedeutet bei euren gerichteten Beleuchtungen nicht automatisch, dass dort kein Kratzer vorhanden ist.
10. Ausrichtung kontrollieren

Nach der Übertragung öffnest du für dieselbe Reihe beispielsweise:

S001_all.bmp
S001_top.bmp
S001_bottom.bmp
S001_left.bmp
S001_right.bmp

und legst jeweils die erzeugte Maske darüber.

Prüfe besonders:

    Bohrungen liegen in allen Bildern an derselben Stelle.

    Die Maske liegt in allen Bildern auf demselben physischen Kratzer.

    Es gibt keine Verschiebung um mehrere Pixel.

    Die Bilder wurden nicht anders beschnitten.

    Es gibt keine Rotation.

Bei einer Verschiebung darf die Maske nicht einfach kopiert werden. Dann muss zuerst eine Bildregistrierung durchgeführt und dieselbe Transformation auf die Maske angewendet werden.

Das Skript enthält bereits einen Alignment-Check. Im Bericht:

annotation/data/annotations_auto/propagation_report.csv

solltest du auf Warnungen und die geschätzten Verschiebungen achten.

In deiner Konfiguration gilt aktuell:

max_shift_px: 1.5

Bei mehr als etwa 1,5 Pixeln sollte die entsprechende Reihe manuell geprüft werden.
11. Review-Overlays erzeugen

Für die schnelle Sichtkontrolle:

uv run python annotation/scripts/create_review_overlays.py \
  --config annotation/config.yaml \
  --mask-source auto

Danach findest du die Bilder unter:

annotation/data/review_overlays/auto/

Öffne für deine Testreihe alle 14 Overlays nacheinander.

Bewertung:

Grün:
Maske liegt korrekt auf dem Kratzer.

Gelb:
Maske ist leicht verschoben oder etwas zu breit.

Rot:
Maske liegt auf Hintergrund, Kratzer fehlt oder Bild ist nicht ausgerichtet.

12. Automatische Masken wieder in CVAT importieren

Damit du nicht jede PNG einzeln öffnen musst, erzeugst du ein CVAT-Importarchiv:

uv run python annotation/scripts/build_cvat_import.py \
  --config annotation/config.yaml \
  --mask-source auto \
  --output annotation/data/cvat_imports/S001_auto_masks.zip \
  --series S001

Danach im CVAT-Task:

Actions
→ Upload annotations
→ Segmentation Mask 1.1
→ S001_auto_masks.zip auswählen

Das Segmentation-Mask-Format unterstützt beim Import echte Masken.

Wichtig: Der Upload ersetzt die bestehenden Annotationen des Tasks. Das Importarchiv muss daher auch die Maske des all-Bilds enthalten.
13. Andere Belichtungen in CVAT kontrollieren

Jetzt gehst du die 13 übrigen Bilder durch.

Pro Bild prüfst du:

    Liegt die Maske geometrisch auf dem Kratzer?

    Ist dieser Bereich in der Beleuchtung überhaupt auswertbar?

    Gibt es einen deutlich sichtbaren Kratzerabschnitt, der in der Mastermaske fehlt?

    Gibt es in der Mastermaske eine Struktur, die sich in keiner anderen Beleuchtung bestätigt?

    Ist eine manuelle Korrektur nötig?

Dabei gilt:

    Geometrisch gleicher Kratzer: Maske beibehalten.

    Dunkler Bereich: nicht automatisch löschen.

    Bildverschiebung: Maske registrieren beziehungsweise korrigieren.

    Fehler in der Masterannotation: zuerst auch das all-Bild korrigieren.

    Nur in einer Beleuchtung sichtbarer Reflex: nicht automatisch als neuer Kratzer aufnehmen.

14. Finale Masken exportieren

Nach der Kontrolle exportierst du den vollständigen Task als:

Segmentation Mask 1.1

Speichere beispielsweise:

annotation/data/cvat_exports/S001_final.zip

Danach extrahierst du die finalen Masken:

uv run python annotation/scripts/extract_cvat_masks.py \
  --config annotation/config.yaml \
  --zip annotation/data/cvat_exports/S001_final.zip \
  --destination final \
  --no-reference-only

Die fertigen Masken liegen danach unter:

annotation/data/annotations_final/

Zum Schluss:

uv run python annotation/scripts/validate_dataset.py \
  --config annotation/config.yaml \
  --stage final \
  --strict

Der konkrete Ablauf für deinen ersten Test

Für genau eine Bildreihe S001:

# 1. CVAT-Polylines exportieren:
#    CVAT for images 1.1

# 2. Polylines in eine 5-Pixel-Maske umwandeln
uv run python annotation/scripts/polylines_to_masks.py \
  --config annotation/config.yaml \
  --zip annotation/data/cvat_exports/S001_cvat.zip \
  --line-width 5

# 3. Mastermaske prüfen
uv run python annotation/scripts/validate_dataset.py \
  --config annotation/config.yaml \
  --stage manual

# 4. Auf alle 14 Beleuchtungen übertragen
uv run python annotation/scripts/mask_propagation.py \
  --config annotation/config.yaml \
  --series S001

# 5. Overlays erzeugen
uv run python annotation/scripts/create_review_overlays.py \
  --config annotation/config.yaml \
  --mask-source auto

# 6. Importarchiv für CVAT erstellen
uv run python annotation/scripts/build_cvat_import.py \
  --config annotation/config.yaml \
  --mask-source auto \
  --output annotation/data/cvat_imports/S001_auto_masks.zip \
  --series S001

Ein wichtiger Punkt: Das erwähnte Skript polylines_to_masks.py ist in deinem bisherigen Paket noch nicht enthalten. Die vorhandenen Skripte gehen von einer bereits rasterisierten Brush-/Polygon-Maske aus. Für deine jetzige Polyline-Methode muss dieser Konverter noch ergänzt werden.

welches format?

Für deinen aktuellen Fall mit Polylines:
Erste Annotation exportieren

In CVAT:

Actions
→ Export task dataset
→ CVAT for images 1.1

Dieses Format enthält die Polyline-Koordinaten in einer annotations.xml. Daraus erzeugt das Skript anschließend die binäre PNG-Maske.
Später die erzeugten Masken wieder in CVAT laden

Actions
→ Upload annotations
→ Segmentation Mask 1.1

Also:

Polyline-Export: CVAT for images 1.1
Masken-Import/Export: Segmentation Mask 1.1

Nicht direkt Segmentation Mask 1.1 für den ersten Export verwenden, weil deine Polylines dort nicht zuverlässig als flächige Masken übernommen werden.

ich habe noch nicht das polyline to masks skript udn acuh in der config finde ich die parameter nciht

Stimmt — das Skript und die Parameter fehlten in dem bisherigen Paket. Ich habe beides ergänzt und außerdem die ebenfalls benötigte common.py hinzugefügt.
Aktualisierte Dateien

Einzeln:

1. Dateien in dein Repository kopieren

Deine Struktur sollte danach so aussehen:

annotation/
├── config.yaml
├── README.md
├── requirements.txt
├── docs/
│   └── annotation_guidelines.md
├── scripts/
│   ├── common.py
│   ├── polylines_to_masks.py
│   ├── mask_propagation.py
│   ├── build_cvat_import.py
│   ├── extract_cvat_masks.py
│   └── validate_dataset.py
└── data/
    ├── images/
    ├── annotations_manual/
    ├── annotations_auto/
    ├── validity_masks/
    ├── training_masks/
    ├── cvat_exports/
    └── cvat_imports/

2. Neuer Abschnitt in der config.yaml

Dort findest du jetzt:

polyline_rasterization:
  # Gesamtbreite der aus CVAT-Polylines erzeugten Scratch-Maske in Pixeln.
  # Für den ersten Test sind 5 Pixel ein sinnvoller Startwert.
  line_width_px: 5

  # Zeichnet Verbindungen und Enden geglättet.
  line_type: "antialiased"

  # Nach dem Zeichnen wird wieder hart binarisiert.
  binary_threshold: 1

  # Runde Enden an den Polylines ergänzen.
  round_end_caps: true

  # Nur Polylines werden verarbeitet.
  accepted_shapes: ["polyline"]

Der wichtigste Parameter ist:

line_width_px: 5

Damit werden deine Centerlines als fünf Pixel breite Masken ausgegeben.
3. CVAT-Annotation exportieren

In CVAT:

Actions
→ Export task dataset
→ CVAT for images 1.1

Das ZIP legst du beispielsweise hier ab:

annotation/data/cvat_exports/S001_manual.zip

Wichtig: Im ZIP muss eine annotations.xml enthalten sein.
4. Polyline-Maske erzeugen

Vom Root deines Repositories:

uv run python annotation/scripts/polylines_to_masks.py \
  --config annotation/config.yaml \
  --zip annotation/data/cvat_exports/S001_manual.zip

Das Skript liest die Breite standardmäßig aus der config.yaml.

Die Maske wird hier erzeugt:

annotation/data/annotations_manual/S001_all.png

Sie enthält ausschließlich:

0   = Hintergrund
255 = Kratzer

5. Breite testweise überschreiben

Du kannst auch direkt verschiedene Breiten ausprobieren, ohne jedes Mal die Config zu ändern:

uv run python annotation/scripts/polylines_to_masks.py \
  --config annotation/config.yaml \
  --zip annotation/data/cvat_exports/S001_manual.zip \
  --line-width 3

Oder:

uv run python annotation/scripts/polylines_to_masks.py \
  --config annotation/config.yaml \
  --zip annotation/data/cvat_exports/S001_manual.zip \
  --line-width 5

Oder:

uv run python annotation/scripts/polylines_to_masks.py \
  --config annotation/config.yaml \
  --zip annotation/data/cvat_exports/S001_manual.zip \
  --line-width 7

Beachte dabei, dass die erzeugte Datei jeweils überschrieben wird. Für einen visuellen Vergleich solltest du sie danach jeweils separat kopieren:

S001_all_width_3.png
S001_all_width_5.png
S001_all_width_7.png

Für die eigentliche Pipeline muss die finale Datei aber wieder exakt heißen:

S001_all.png

6. Nur eine bestimmte Serie verarbeiten

Falls ein CVAT-Export mehrere Serien enthält:

uv run python annotation/scripts/polylines_to_masks.py \
  --config annotation/config.yaml \
  --zip annotation/data/cvat_exports/person_a_manual.zip \
  --series S001

Mehrere ausgewählte Serien:

uv run python annotation/scripts/polylines_to_masks.py \
  --config annotation/config.yaml \
  --zip annotation/data/cvat_exports/person_a_manual.zip \
  --series S001 S002 S003

Standardmäßig werden nur die Bilder mit der Referenzbeleuchtung verarbeitet:

*_all.bmp

Das entspricht:

--reference-only

7. Erzeugte Mastermaske prüfen

uv run python annotation/scripts/validate_dataset.py \
  --config annotation/config.yaml \
  --stage manual

Dabei werden unter anderem geprüft:

    Maske vorhanden,

    gleiche Auflösung wie das Bild,

    nur Pixelwerte 0 und 255,

    korrektes Referenzbild all.

8. Anschließend auf alle Belichtungen übertragen

Sobald beispielsweise diese Datei existiert:

annotation/data/annotations_manual/S001_all.png

führst du aus:

uv run python annotation/scripts/mask_propagation.py \
  --config annotation/config.yaml \
  --series S001

Danach entstehen beispielsweise:

annotation/data/annotations_auto/
├── S001_all.png
├── S001_top.png
├── S001_bottom.png
├── S001_left.png
├── S001_right.png
└── ...

Zusätzlich:

annotation/data/validity_masks/
annotation/data/training_masks/

Kompletter Ablauf für deinen aktuellen Test

# 1. Abhängigkeiten installieren
uv pip install -r annotation/requirements.txt

# 2. Polylines aus CVAT in eine 5-Pixel-Maske umwandeln
uv run python annotation/scripts/polylines_to_masks.py \
  --config annotation/config.yaml \
  --zip annotation/data/cvat_exports/S001_manual.zip \
  --line-width 5

# 3. Mastermaske prüfen
uv run python annotation/scripts/validate_dataset.py \
  --config annotation/config.yaml \
  --stage manual

# 4. Maske auf alle Bilder der Reihe übertragen
uv run python annotation/scripts/mask_propagation.py \
  --config annotation/config.yaml \
  --series S001

Das neue Skript habe ich mit einem beispielhaften CVAT-XML-Export getestet. Es erzeugt eine Maske in der korrekten Auflösung mit ausschließlich den Werten 0 und 255.


polylines_to_masks.py

#!/usr/bin/env python3
"""Rasterisiert CVAT-Polylines aus 'CVAT for images 1.1' zu binären PNG-Masken.

Beispiel:
uv run python annotation/scripts/polylines_to_masks.py \
  --config annotation/config.yaml \
  --zip annotation/data/cvat_exports/S001_cvat.zip

Der Standardwert für die Linienbreite wird aus config.yaml gelesen und kann mit
--line-width überschrieben werden.
"""

from __future__ import annotations

import argparse
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from common import load_config, resolve_path, write_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("annotation/config.yaml"))
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument(
        "--line-width",
        type=int,
        default=None,
        help="Gesamtbreite der Linienmaske in Pixeln; überschreibt config.yaml.",
    )
    parser.add_argument(
        "--destination",
        choices=["manual", "auto", "final"],
        default="manual",
        help="Zielordner für die erzeugten Masken.",
    )
    parser.add_argument(
        "--reference-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Standardmäßig nur *_all-Bilder rasterisieren.",
    )
    parser.add_argument("--series", nargs="*", help="Optional nur bestimmte Serien verarbeiten.")
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Vorhandene Masken überschreiben.",
    )
    return parser.parse_args()


def parse_points(raw: str) -> np.ndarray:
    points: list[tuple[int, int]] = []
    for pair in raw.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        x_raw, y_raw = pair.split(",", maxsplit=1)
        points.append((int(round(float(x_raw))), int(round(float(y_raw)))))
    if len(points) < 2:
        raise ValueError(f"Polyline benötigt mindestens zwei Punkte: {raw!r}")
    return np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))


def infer_series_and_lighting(stem: str, regex: str) -> tuple[str, str]:
    import re

    match = re.match(regex, Path(stem).name)
    if not match or len(match.groups()) < 2:
        raise ValueError(f"Bildname passt nicht zu filename_regex: {stem}")
    return match.group(1), match.group(2)


def locate_annotations_xml(root: Path) -> Path:
    direct = root / "annotations.xml"
    if direct.is_file():
        return direct
    matches = list(root.rglob("annotations.xml"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Im CVAT-ZIP wurde keine eindeutige annotations.xml gefunden ({len(matches)} Treffer)."
        )
    return matches[0]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    zip_path = args.zip_path.resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)

    poly_cfg = config.get("polyline_rasterization", {})
    line_width = args.line_width if args.line_width is not None else int(poly_cfg.get("line_width_px", 5))
    if line_width < 1:
        raise ValueError("line_width_px muss mindestens 1 sein.")
    threshold = int(poly_cfg.get("binary_threshold", 1))
    round_caps = bool(poly_cfg.get("round_end_caps", True))
    line_type_name = str(poly_cfg.get("line_type", "antialiased")).lower()
    line_type = cv2.LINE_AA if line_type_name in {"antialiased", "aa", "line_aa"} else cv2.LINE_8

    label = str(config["cvat"]["scratch_label"])
    reference = str(config["dataset"]["reference_lighting"]).lower()
    filename_regex = str(config["dataset"]["filename_regex"])
    selected_series = set(args.series or [])
    destination_key = {
        "manual": "manual_masks",
        "auto": "auto_masks",
        "final": "final_masks",
    }[args.destination]
    destination_root = resolve_path(config, destination_key)

    written = 0
    skipped = 0
    total_polylines = 0

    with tempfile.TemporaryDirectory(prefix="cvat_polyline_") as temp_dir:
        root = Path(temp_dir)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(root)

        xml_path = locate_annotations_xml(root)
        tree = ET.parse(xml_path)
        annotations = tree.getroot()

        for image_node in annotations.findall("image"):
            name = image_node.get("name")
            width_raw = image_node.get("width")
            height_raw = image_node.get("height")
            if not name or not width_raw or not height_raw:
                raise ValueError("CVAT-XML enthält ein Bild ohne name/width/height.")

            relative = PurePosixPath(name)
            stem = relative.stem
            series_id, lighting = infer_series_and_lighting(stem, filename_regex)
            if selected_series and series_id not in selected_series:
                skipped += 1
                continue
            if args.reference_only and lighting.lower() != reference:
                skipped += 1
                continue

            width = int(width_raw)
            height = int(height_raw)
            mask = np.zeros((height, width), dtype=np.uint8)
            image_polylines = 0

            for polyline in image_node.findall("polyline"):
                if polyline.get("label") != label:
                    continue
                if polyline.get("outside") == "1":
                    continue
                points_raw = polyline.get("points")
                if not points_raw:
                    continue
                points = parse_points(points_raw)
                cv2.polylines(
                    mask,
                    [points],
                    isClosed=False,
                    color=255,
                    thickness=line_width,
                    lineType=line_type,
                )
                if round_caps:
                    radius = max(1, line_width // 2)
                    cv2.circle(mask, tuple(points[0, 0]), radius, 255, thickness=-1, lineType=line_type)
                    cv2.circle(mask, tuple(points[-1, 0]), radius, 255, thickness=-1, lineType=line_type)
                image_polylines += 1

            # Garantiert binär: ausschließlich 0 und 255.
            mask = np.where(mask >= threshold, 255, 0).astype(np.uint8)
            output_relative = Path(*relative.parts).with_suffix(str(config["dataset"].get("mask_extension", ".png")))
            output_path = destination_root / output_relative
            if output_path.exists() and not args.overwrite:
                skipped += 1
                continue
            write_png(output_path, mask)
            written += 1
            total_polylines += image_polylines
            print(f"[OK] {name}: {image_polylines} Polylines -> {output_path}")

    print(f"Erzeugte Masken: {written}")
    print(f"Rasterisierte Polylines: {total_polylines}")
    print(f"Übersprungene Bilder: {skipped}")
    print(f"Linienbreite: {line_width} px")


if __name__ == "__main__":
    main()

