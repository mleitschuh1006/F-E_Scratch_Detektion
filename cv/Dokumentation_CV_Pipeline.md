# Dokumentation der klassischen CV-Pipeline zur Kratzererkennung

## 1. Grundidee

Ziel der entwickelten Pipeline ist es, Kratzer auf metallischen Bauteilen mit **klassischer Computer Vision (CV)** zu erkennen.

Der Ansatz besitzt kein semantisches Verständnis davon, was ein Kratzer ist. Stattdessen wird nach Bildmerkmalen gesucht, die für Kratzer typisch sind:

- lokale Helligkeits- bzw. Kontrastunterschiede,
- längliche Strukturen,
- zusammenhängende auffällige Bereiche.

Das Grundprinzip lautet:

> **Auffällige Strukturen zunächst erkennen und anschließend bekannte Störstrukturen gezielt herausfiltern.**

Dadurch entsteht eine binäre Maske, in der mögliche Kratzer markiert werden.

---

# 2. Verarbeitungsschritte der Pipeline

## 2.1 Umwandlung in Graustufen

Das Eingangsbild wird zunächst in ein Graustufenbild umgewandelt.

**Warum?**
- Für die Kratzererkennung ist hauptsächlich die Helligkeitsinformation relevant.
- Die weitere Verarbeitung wird dadurch einfacher.

**Folge:**  
Jeder Pixel besitzt nur noch einen Intensitätswert zwischen dunkel und hell.

---

## 2.2 Bestimmung des Bauteilbereichs

Der schwarze Hintergrund wird vom eigentlichen Bauteil getrennt.

Dazu werden unter anderem

- ein Grauwert-Schwellwert,
- morphologisches Closing,
- und die größte zusammenhängende Fläche

verwendet.

**Warum?**  
Die Kratzererkennung soll nur auf dem Bauteil und nicht im Bildhintergrund durchgeführt werden.

**Folge:**  
Es entsteht eine Region of Interest (ROI), welche die zulässige Suchfläche definiert.

---

## 2.3 Ausschluss des direkten Außenrandes

Der Bereich unmittelbar an der Bauteilaußenkante wird teilweise ausgeschlossen.

**Warum?**  
Der Übergang zwischen hellem Bauteil und schwarzem Hintergrund erzeugt einen sehr starken Kontrast und würde ansonsten häufig als Kratzer erkannt.

**Folge:**  
False Positives an der Außenkontur werden reduziert.

**Nachteil:**  
Kratzer direkt an der Außenkante können dadurch schwerer oder gar nicht erkannt werden.

---

## 2.4 Grauwertspreizung

Die Grauwerte des Bauteils werden auf einen größeren Wertebereich verteilt.

**Warum?**  
Schwache Helligkeitsunterschiede zwischen Kratzer und Oberfläche werden dadurch deutlicher.

**Folge:**  
Kratzer können stärker hervortreten, allerdings werden auch andere kontrastreiche Strukturen verstärkt.

---

## 2.5 Glättung

Das Bild wird leicht mit einem Gaussian Blur geglättet.

**Warum?**
- kleine Pixelstörungen reduzieren,
- hochfrequentes Bildrauschen unterdrücken.

**Folge:**  
Die nachfolgende Kratzererkennung reagiert weniger stark auf einzelne Störpixel.

---

## 2.6 Local Residual als zentrales Kratzermerkmal

Der wichtigste Schritt der Pipeline ist die Berechnung eines **Local Residuals**.

Dazu wird zunächst eine stärker geglättete Version des Bildes als lokaler Hintergrund erzeugt. Anschließend wird die lokale Abweichung bestimmt:

\[
R(x,y) = |I(x,y) - I_{lokal}(x,y)|
\]

**Interpretation:**
- gleichmäßige Oberfläche → geringe Abweichung,
- auffällige Struktur → hohe Abweichung.

**Warum ist das für Kratzer geeignet?**  
Kratzer unterscheiden sich häufig lokal von ihrer direkten Umgebung und erzeugen dadurch eine starke Residual-Antwort.

**Zentrale Einschränkung:**  
Der Algorithmus erkennt an dieser Stelle **keine Kratzer**, sondern lediglich starke lokale Bildabweichungen. Auch Bohrungen, Kanten oder andere Strukturen können deshalb eine starke Antwort erzeugen.

---

## 2.7 Binarisierung

Aus dem kontinuierlichen Residualbild wird über einen Schwellwert eine binäre Maske erzeugt.

Dabei wird ein Perzentil-Schwellwert verwendet, sodass nur besonders starke lokale Abweichungen als Kandidaten übernommen werden.

**Folge:**

- `0` → kein Kratzerkandidat
- `1` → auffällige Struktur / möglicher Kratzer

---

## 2.8 Morphologische Nachbearbeitung

Mit einem morphologischen **Closing** werden kleine Lücken innerhalb erkannter Strukturen geschlossen.

**Warum?**  
Ein Kratzer kann aufgrund von Reflexionen oder lokalen Helligkeitsunterschieden unterbrochen erscheinen.

**Folge:**  
Nahe beieinanderliegende Kratzerbereiche werden stärker zu zusammenhängenden Strukturen verbunden.

---

## 2.9 Connected Components

Die binäre Maske wird in einzelne zusammenhängende Komponenten zerlegt.

Dadurch können einzelne Kandidaten anhand ihrer Eigenschaften untersucht werden.

Sehr kleine Komponenten werden entfernt.

**Warum?**  
Viele kleine isolierte Bereiche entstehen durch Oberflächenstrukturen oder Bildrauschen und sind keine relevanten Kratzer.

---

# 3. Filterung von Bohrungen

## Warum ist ein zusätzlicher Bohrungsfilter notwendig?

Eine Bohrung besitzt typischerweise einen starken Übergang zwischen

- dunklem Bohrungsinneren
- und heller Metalloberfläche.

Dieser Übergang erzeugt einen sehr hohen Local Residual und kann daher fälschlicherweise als Kratzer erkannt werden.

Da Bohrungen bekannte konstruktive Elemente des Bauteils sind, werden sie gezielt aus dem Suchbereich entfernt.

## Funktionsweise

Die Erkennung erfolgt über eine **Hough-Kreisdetektion**.

Gefundene Kreiskandidaten werden zusätzlich geprüft:

- Mittelpunkt liegt innerhalb des Bauteils,
- Zentrum des Kreises ist ausreichend dunkel,
- Umgebung des Kreises ist deutlich heller.

Wird ein Kandidat als Bohrung akzeptiert, wird zusätzlich ein kleiner Sicherheitsbereich um die Bohrung aus der Kratzer-ROI entfernt.

**Folge:**  
Insbesondere die kontrastreichen Lochränder sollen nicht mehr als Kratzer erscheinen.

## Grenzen

Die Bohrungserkennung funktioniert nicht für jede Geometrie gleich gut. Probleme können beispielsweise auftreten bei:

- sehr kleinen Bohrungen,
- vielen dicht nebeneinanderliegenden Löchern,
- stark unterschiedlichen Lochgrößen,
- nicht kreisförmigen Aussparungen,
- unvollständig sichtbaren oder optisch schwachen Lochkonturen.

Nicht erkannte Lochränder können weiterhin False Positives erzeugen.

---

# 4. Filterung von Bauteilkanten

## Warum ist ein zusätzlicher Kantenfilter notwendig?

Bauteilkanten besitzen ebenfalls typische Eigenschaften eines möglichen Kratzers:

- hoher Kontrast,
- längliche Form,
- zusammenhängende Struktur.

Ein reiner Kontrastdetektor kann daher nicht sicher zwischen Kratzer und konstruktiver Außenkante unterscheiden.

## Funktionsweise

Für erkannte Komponenten wird geprüft:

- befindet sich die Struktur nahe an der Bauteilaußenkante?
- liegt ein großer Anteil ihrer Pixel im Randbereich?
- ist die Struktur ausreichend lang und länglich?
- verläuft sie ungefähr parallel zur lokalen Bauteilkante?

Die Orientierung wird dabei über eine **PCA-basierte Hauptachsenbestimmung** abgeschätzt.

Eine Komponente wird nur entfernt, wenn mehrere dieser Bedingungen gleichzeitig erfüllt sind.

**Folge:**  
Typische entlang der Bauteilkontur verlaufende False Positives können reduziert werden.

**Grenze:**  
Ein echter Kratzer, der ebenfalls randnah und parallel zur Bauteilkante verläuft, kann dadurch fälschlicherweise entfernt werden.

---

# 5. Zentrale Probleme des klassischen CV-Ansatzes

Die wesentliche Grenze der Methode besteht darin, dass sie nur **Bildmerkmale**, aber nicht die Bedeutung einer Struktur erkennt.

Dadurch entstehen insbesondere folgende Probleme:

- nicht alle Kratzer besitzen einen ausreichend starken lokalen Kontrast,
- sehr kleine oder dünne Kratzer werden häufiger übersehen,
- Bohrungs- und Aussparungskanten können wie Kratzer erscheinen,
- Bauteilkanten können wie lange Kratzer erscheinen,
- konstruktive Oberflächenstrukturen können False Positives erzeugen,
- Filter zur Entfernung von Störungen können gleichzeitig echte Kratzer entfernen,
- stark unterschiedliche Bauteilgeometrien erschweren die Verwendung derselben Regeln und Parameter.

Ein Kratzer und eine konstruktive Linie können aus Sicht der klassischen Bildverarbeitung teilweise nahezu identische Merkmale besitzen.

---

# 6. Für welche Bauteile ist der Ansatz sinnvoll?

Der Ansatz ist insbesondere für Bauteile geeignet, die sich geometrisch und strukturell stark ähneln.

### Eher geeignet

- ähnliche bzw. wiederkehrende Bauteilgeometrien,
- wenige unterschiedliche konstruktive Strukturen,
- bekannte und wiederkehrende Bohrungen oder Aussparungen,
- relativ homogene Oberflächenstrukturen,
- Kratzer mit ähnlicher visueller Ausprägung,
- wenige kratzerähnliche konstruktive Elemente,
- Kratzer mit ausreichend deutlichem lokalen Kontrast,
- Bauteile, deren Außenkonturen und Störstrukturen mit festen Regeln gut beschrieben werden können.

### Eher ungeeignet

- stark unterschiedliche Bauteilgeometrien,
- viele Bohrungen, Aussparungen, Rippen oder andere konstruktive Strukturen,
- stark strukturierte oder inhomogene Oberflächen,
- konstruktive Strukturen mit sehr ähnlichem Erscheinungsbild wie Kratzer,
- sehr unterschiedliche Kratzerformen, -breiten und -kontraste,
- sehr kleine oder kontrastarme Kratzer,
- viele randnahe Kratzer,
- viele unterschiedliche Lochgrößen oder nicht kreisförmige Aussparungen.

Die Evaluation auf dem homogenen Teil des Datensatzes zeigt entsprechend, dass die Pipeline auf geometrisch ähnlichen Standardbauteilen besser funktioniert als auf dem vollständigen heterogenen Bauteilspektrum.

---

# 7. Vor- und Nachteile

## Vorteile

- bekannte Bauteilgeometrien können gezielt berücksichtigt werden,
- bekannte Störstrukturen wie Bohrungen und Außenkanten können explizit ausgeschlossen werden,
- bei ähnlichen Bauteilen können feste geometrische Regeln gut funktionieren,
- lokale Kratzermerkmale lassen sich direkt über Bildoperationen hervorheben,
- die einzelnen Verarbeitungsschritte sind klar voneinander getrennt und technisch nachvollziehbar.

## Nachteile

- konstruktive Strukturen können dieselben visuellen Merkmale wie Kratzer besitzen,
- Störfilter können gleichzeitig echte Kratzer entfernen,
- geringe Übertragbarkeit auf deutlich andere Bauteilgeometrien,
- Schwierigkeiten bei sehr kleinen oder kontrastarmen Kratzern,
- Bohrungsfilter funktioniert nicht zuverlässig für jede Loch- oder Aussparungsgeometrie,
- Randfilter erschwert die Erkennung echter randnaher Kratzer,
- Local Residual unterscheidet ausschließlich anhand von Bildmerkmalen und nicht anhand der semantischen Bedeutung einer Struktur.

---

# 8. Fazit

Die entwickelte CV-Pipeline zeigt, dass Kratzer auf **geometrisch ähnlichen und vergleichsweise homogenen Bauteilen** mit klassischen Bildverarbeitungsmethoden grundsätzlich detektiert werden können.

Die zentrale Stärke liegt darin, typische Kratzer zunächst anhand lokaler Helligkeitsabweichungen zu erkennen und bekannte geometrische Störstrukturen anschließend gezielt auszuschließen.

Gleichzeitig liegt genau darin die wesentliche Grenze des Ansatzes: Die Methode besitzt kein semantisches Verständnis eines Kratzers. Sobald konstruktive Strukturen ähnliche Bildmerkmale besitzen oder die Bauteilgeometrien stark variieren, nimmt die Zuverlässigkeit der festen CV-Regeln ab.

Der Ansatz ist daher insbesondere für **klar definierte und wiederkehrende Bauteiltypen** sinnvoll. Für stark heterogene Bauteile mit vielen unterschiedlichen geometrischen Strukturen ist eine rein klassische CV-Lösung deutlich schwieriger zuverlässig umzusetzen.
