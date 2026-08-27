# Finale Evaluation der klassischen CV-Kratzerdetektion

## 1. Ziel und Einordnung

Dieses Dokument fasst die Ergebnisse des **eingefrorenen klassischen Computer-Vision-Ansatzes** zur Kratzerdetektion zusammen. Grundlage sind die automatisch erzeugten Dateien aus dem Ordner `evaluation/`.

Ausgewertet wurden **34 Bild-Masken-Paare**.

Die Evaluation betrachtet zwei unterschiedliche Fragestellungen:

1. **Wie genau stimmt die vorhergesagte Segmentierungsmaske mit der Ground-Truth-Maske überein?**
2. **Wie zuverlässig wird ein einzelner annotierter Kratzer grundsätzlich erkannt?**

Diese beiden Fragestellungen müssen getrennt betrachtet werden. Ein Kratzer kann beispielsweise an der richtigen Stelle erkannt werden, obwohl die vorhergesagte Linie schmaler als die manuell annotierte Maske ist. In diesem Fall ist die reine Pixel-F1 relativ niedrig, obwohl die Lokalisierung des Kratzers grundsätzlich korrekt ist.

Zusätzlich ist bei der Interpretation der Kennzahlen die verwendete **Annotationsdefinition** zu berücksichtigen. Bei der manuellen Annotation wurden bewusst nur Strukturen als Kratzer gekennzeichnet, die:

- **hell beziehungsweise weiß** erscheinen,
- eine **linienförmige Struktur** besitzen und
- **ohne extremes Hineinzoomen** im Bild erkennbar sind.

Die klassische CV-Methode arbeitet dagegen nicht mit dieser semantischen Definition, sondern reagiert auf **lokale Kontrastabweichungen**. Dadurch können neben den annotierten hellen Kratzern auch **dunkle Kratzer oder andere dunkle Oberflächenfehler** erkannt werden. Solche Strukturen fehlen gegebenenfalls in der Ground Truth und werden in der rein quantitativen Auswertung deshalb formal als False Positives gezählt, obwohl sie visuell durchaus reale Oberflächenabweichungen darstellen können.

> **Wichtiger Hinweis:** Die Ergebnisse sind eine deskriptive Evaluation auf dem vorhandenen Datensatz. Falls dieselben Bilder während der manuellen oder automatischen Parametrierung verwendet wurden, stellen die Kennzahlen **keine vollständig unabhängige Test-Set-Genauigkeit** dar. Für den Vergleich verschiedener Verfahren auf demselben Datensatz sind sie dennoch gut geeignet. Sämtliche Kennzahlen beziehen sich außerdem auf die festgelegte Annotationsdefinition und nicht auf eine vollständige Erfassung aller visuell vorhandenen Oberflächenfehler.

---

# 2. Gesamtbewertung auf Pixelebene

## 2.1 Strikte Pixelbewertung

Bei der strikten Bewertung muss ein vorhergesagtes Pixel exakt mit einem Ground-Truth-Pixel übereinstimmen.

| Metrik | Ergebnis |
|---|---:|
| True Positives | 438,922 |
| False Positives | 1,214,685 |
| False Negatives | 709,349 |
| Precision | **26.5 %** |
| Recall | **38.2 %** |
| F1-Score | **31.3 %** |
| IoU | **18.6 %** |
| Specificity | 98.9 % |

### Interpretation

Die **Precision von 26.5 %** bedeutet, dass 26.5 % der von der CV-Methode markierten Pixel exakt mit den nach der festgelegten Annotationsdefinition gekennzeichneten Ground-Truth-Pixeln übereinstimmen. Der verbleibende Anteil darf jedoch **nicht vollständig mit tatsächlichen Fehlklassifikationen gleichgesetzt werden**.

Ein wesentlicher Einflussfaktor ist die verwendete Annotationsdefinition. Manuell annotiert wurden ausschließlich Strukturen, die **hell beziehungsweise weiß, linienförmig und ohne extremes Hineinzoomen erkennbar** sind. Die CV-Methode basiert dagegen auf lokalen Kontrastunterschieden und unterscheidet grundsätzlich nicht danach, ob eine Struktur heller oder dunkler als ihre Umgebung ist. Dadurch werden teilweise auch **dunkle Kratzer beziehungsweise andere kontrastreiche Oberflächenfehler** erkannt, die nicht Bestandteil der Ground-Truth-Masken sind. In der quantitativen Auswertung werden solche Detektionen formal als False Positives gezählt, obwohl sie visuell reale Oberflächenabweichungen darstellen können.

Weitere typische Ursachen für False Positives sind:

- Bauteilaußenkanten,
- Kanten von Bohrungen und Aussparungen,
- nicht vollständig erkannte Bohrungen,
- Schrift und Beschriftungen,
- stark strukturierte beziehungsweise perforierte Bauteile,
- vereinzelte lokale Oberflächenstrukturen.

Der **Recall von 38.2 %** bedeutet, dass 38.2 % der gesamten manuell annotierten Kratzerfläche pixelgenau durch die CV-Methode abgedeckt werden.

Auch diese Bewertung ist sehr streng, da bereits wenige Pixel Versatz oder eine abweichende Segmentierungsbreite als False-Negative- beziehungsweise False-Positive-Pixel gewertet werden. Die strikte Pixelbewertung beschreibt daher primär die **Übereinstimmung der CV-Segmentierung mit der festgelegten Ground Truth** und nicht uneingeschränkt die Fähigkeit der Methode, sämtliche tatsächlich vorhandenen Oberflächenfehler zu erkennen.

Die Precision sollte deshalb nicht als Aussage

> „Nur 26.5 % der CV-Erkennungen sind echte Kratzer.“

interpretiert werden. Korrekt ist vielmehr:

> **26.5 % der von der CV-Methode markierten Pixel stimmen exakt mit den nach der festgelegten Annotationsdefinition gekennzeichneten Kratzerpixeln überein.**

---

## 2.2 Toleranzbasierte Pixelbewertung

Um zu berücksichtigen, dass die genaue Breite und Lage einer manuellen Kratzermaske nicht vollständig objektiv ist, wurde zusätzlich eine räumliche Toleranz verwendet.

| Toleranz | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0 px | 26.5 % | 38.2 % | **31.3 %** |
| ±2 px | 34.4 % | 50.3 % | **40.8 %** |
| ±5 px | 36.7 % | 57.5 % | **44.8 %** |
| ±10 px | 37.9 % | 62.9 % | **47.3 %** |

Der F1-Score steigt bereits bei einer Toleranz von **±5 px von 31.3 % auf 44.8 %**.

Das zeigt, dass ein relevanter Teil der Abweichung aus **leicht versetzten oder unterschiedlich breiten Segmentierungen** resultiert. Die klassische CV findet die Kratzer somit häufiger in der richtigen räumlichen Umgebung, als es der vollständig pixelgenaue F1-Score allein vermuten lässt.

Für die Beurteilung dieses Ansatzes sollte deshalb neben dem strikten F1 insbesondere der **F1 bei ±5 px Toleranz** berücksichtigt werden.

---

# 3. Erkennung einzelner Kratzer

Jede zusammenhängende Komponente der Ground-Truth-Maske wird als ein Kratzer behandelt.

Ein Kratzer gilt als erkannt, wenn ein definierter Mindestanteil seiner Ground-Truth-Pixel von der CV-Prediction überdeckt wird.

| notwendige GT-Überdeckung | erkannte Kratzer | Gesamtzahl | Erkennungsrate | 95-%-Konfidenzintervall |
|---:|---:|---:|---:|---:|
| ≥5 % | 882 | 1303 | **67.7 %** | 65.1 % – 70.2 % |
| ≥15 % | 829 | 1303 | **63.6 %** | 61.0 % – 66.2 % |
| ≥30 % | 700 | 1303 | **53.7 %** | 51.0 % – 56.4 % |
| ≥50 % | 502 | 1303 | **38.5 %** | 35.9 % – 41.2 % |

Für die weitere Auswertung wird **15 % Überdeckung** als primäre Definition verwendet.

Damit werden:

**829 von 1303 Kratzern = 63.6 %**

erkannt.

### Interpretation

Die Erkennungsrate sinkt deutlich, wenn nicht nur irgendein Teil, sondern ein größerer Anteil des Kratzers korrekt segmentiert werden muss.

Das zeigt:

- Die CV-Methode lokalisiert viele Kratzer grundsätzlich.
- Bei einem Teil davon wird jedoch nur ein Ausschnitt des Kratzers erkannt.
- Eine vollständige Segmentierung von Form und Breite ist deutlich schwieriger als eine reine Lokalisierung.

---

# 4. False Positives auf Komponentenebene

Die Prediction enthält insgesamt **3670 zusammenhängende vorhergesagte Komponenten**.

Davon besitzen **1211** mindestens 15 % eigene Überdeckung mit einer Ground-Truth-Kratzerregion.

Damit ergibt sich eine komponentenbasierte Precision von:

**33.0 %**  
95-%-Konfidenzintervall: 31.5 % – 34.5 %

Diese Kennzahl ist **keine strikte Eins-zu-eins-Objekterkennungs-Precision**. Außerdem ist sie – genau wie die Pixel-Precision – von der festgelegten Ground Truth abhängig. Eine vorhergesagte Komponente kann beispielsweise einen dunklen, visuell plausiblen Oberflächenfehler markieren und dennoch als nicht durch die Ground Truth gestützt gelten, wenn dieser nach der Annotationsdefinition nicht annotiert wurde.

Trotzdem zeigt die Kennzahl zusammen mit der visuellen Auswertung der Overlays, dass weiterhin viele zusätzliche Strukturen detektiert werden. Die größte verbleibende Schwäche des klassischen CV-Ansatzes ist damit die **Trennung der annotierten Kratzer von anderen kontrastreichen Strukturen und nicht annotierten Oberflächenabweichungen**.

---

# 5. Einfluss der Kratzergröße

Die folgende Größenanalyse verwendet ebenfalls die primäre Definition:

> Ein Kratzer gilt als erkannt, wenn mindestens 15 % seiner Ground-Truth-Pixel von der Prediction überdeckt werden.

Die folgenden Erkennungsraten beziehen sich ausschließlich auf die **nach der festgelegten Definition annotierten Kratzer**. Dunkle oder anderweitig nicht annotierte Oberflächenfehler gehen nicht als zusätzliche Ground-Truth-Objekte in diese Größenanalyse ein.

## 5.1 Abhängigkeit von der Kratzerfläche

| Fläche | Kratzer | erkannt | Erkennungsrate | 95-%-Konfidenzintervall |
|---|---:|---:|---:|---:|
| 0–<100 px | 177 | 90 | **50.8 %** | 43.5 % – 58.1 % |
| 100–<200 px | 283 | 168 | **59.4 %** | 53.6 % – 64.9 % |
| 200–<400 px | 308 | 189 | **61.4 %** | 55.8 % – 66.6 % |
| 400–<800 px | 254 | 170 | **66.9 %** | 60.9 % – 72.4 % |
| 800–<1600 px | 152 | 112 | **73.7 %** | 66.2 % – 80.0 % |
| 1600–<3200 px | 69 | 56 | **81.2 %** | 70.4 % – 88.6 % |
| ≥ 3200 px | 60 | 44 | **73.3 %** | 61.0 % – 82.9 % |

### Bewertung

Die Fläche zeigt einen deutlichen Zusammenhang mit der Detektionswahrscheinlichkeit.

- Sehr kleine Kratzer unter 100 px werden nur zu **50.8 %** erkannt.
- Zwischen 800 und 1600 px steigt die Erkennungsrate bereits auf **73.7 %**.
- Im Bereich 1600–3200 px werden **81.2 %** erkannt.

Für extrem große Komponenten oberhalb von 3200 px fällt die Rate wieder etwas ab. Große Fläche allein garantiert daher keine Detektion. Sehr große Ground-Truth-Komponenten können komplexe, verzweigte oder auf schwierigen Bauteilstrukturen liegende Kratzer enthalten.

---

## 5.2 Abhängigkeit von der geschätzten Kratzerlänge

Die Länge wird über die PCA-Hauptachse der Ground-Truth-Komponente abgeschätzt.

| geschätzte Länge | Kratzer | erkannt | Erkennungsrate | 95-%-Konfidenzintervall |
|---|---:|---:|---:|---:|
| 0–<25 px | 219 | 110 | **50.2 %** | 43.7 % – 56.8 % |
| 25–<50 px | 397 | 261 | **65.7 %** | 60.9 % – 70.2 % |
| 50–<100 px | 362 | 217 | **59.9 %** | 54.8 % – 64.9 % |
| 100–<150 px | 136 | 106 | **77.9 %** | 70.3 % – 84.1 % |
| 150–<250 px | 83 | 55 | **66.3 %** | 55.6 % – 75.5 % |
| 250–<400 px | 62 | 45 | **72.6 %** | 60.4 % – 82.1 % |
| 400–<800 px | 29 | 22 | **75.9 %** | 57.9 % – 87.8 % |
| ≥ 800 px | 15 | 13 | **86.7 %** | 62.1 % – 96.3 % |

### Bewertung

Grundsätzlich werden lange Kratzer häufiger erkannt als sehr kurze Kratzer. Der Zusammenhang ist jedoch weniger gleichmäßig als bei der Fläche.

Dies zeigt, dass **Länge allein nicht entscheidet**, ob ein Kratzer gut erkennbar ist. Ebenfalls relevant sind unter anderem:

- Breite,
- lokaler Kontrast,
- Orientierung,
- Unterbrechungen im Kratzer,
- Lage an problematischen Bauteilstrukturen.

Sehr lange Kratzer ab 800 px erreichen in diesem Datensatz eine hohe Erkennungsrate, die Stichprobe ist mit nur 15 Kratzern jedoch klein.

---

## 5.3 Abhängigkeit von der geschätzten Kratzerbreite

Die Breite wird als Ground-Truth-Fläche geteilt durch die geschätzte PCA-Länge bestimmt.

| geschätzte Breite | Kratzer | erkannt | Erkennungsrate | 95-%-Konfidenzintervall |
|---|---:|---:|---:|---:|
| 0–<2 px | 12 | 3 | **25.0 %** | 8.9 % – 53.2 % |
| 2–<4 px | 263 | 155 | **58.9 %** | 52.9 % – 64.7 % |
| 4–<6 px | 356 | 212 | **59.6 %** | 54.4 % – 64.5 % |
| 6–<8 px | 354 | 207 | **58.5 %** | 53.3 % – 63.5 % |
| 8–<12 px | 240 | 188 | **78.3 %** | 72.7 % – 83.1 % |
| 12–<20 px | 64 | 53 | **82.8 %** | 71.8 % – 90.1 % |
| 20–<40 px | 12 | 9 | **75.0 %** | 46.8 % – 91.1 % |
| ≥ 40 px | 2 | 2 | **100.0 %** | 34.2 % – 100.0 % |

### Bewertung

Die Breite besitzt einen besonders deutlichen Einfluss:

- Unter 2 px Breite werden nur **25,0 %** erkannt. Aufgrund von nur 12 Kratzern ist dieser Wert allerdings unsicher.
- Zwischen 2 und 8 px liegt die Erkennungsrate ungefähr im Bereich von 58–60 %.
- Ab 8–12 px steigt sie auf **78,3 %**.
- Zwischen 12 und 20 px werden **82,8 %** erkannt.

Sehr breite Kategorien oberhalb 20 px enthalten nur wenige Beispiele und sollten deshalb nicht isoliert als zuverlässige Grenzwerte interpretiert werden.

---

# 6. Größenbereiche für mindestens 80 % empirische Erkennungsrate

Das Evaluationsprogramm sucht zusätzlich die kleinste beobachtete Größe, ab der alle mindestens ebenso großen Kratzer zusammen eine empirische Erkennungsrate von mindestens 80 % erreichen. Es werden mindestens 20 Beispiele verlangt.

| Größenmerkmal | Schwelle | Kratzer ≥ Schwelle | erkannt | Erkennungsrate | 95-%-Konfidenzintervall |
|---|---:|---:|---:|---:|---:|
| Fläche | **≥ 4744 px** | 37 | 30 | **81.1 %** | 65.8 % – 90.5 % |
| Länge | **≥ 395,0 px** | 45 | 36 | **80.0 %** | 66.2 % – 89.1 % |
| Breite | **≥ 11,5 px** | 96 | 77 | **80.2 %** | 71.1 % – 86.9 % |

Diese Werte dürfen nicht als harte physikalische Detektionsgrenze verstanden werden. Sie bedeuten:

> Unter den im Datensatz vorhandenen Kratzern, die mindestens diese Größe besitzen, wurden mindestens 80 % erkannt.

Besonders bei Fläche und Länge sind die Konfidenzintervalle aufgrund der begrenzten Anzahl großer Kratzer relativ breit.

---

# 7. Unterschiede zwischen einzelnen Bildern

Die Leistung variiert stark zwischen verschiedenen Bauteilen.

## Bilder mit vergleichsweise hoher Pixel-F1

| Bild | Precision | Recall | F1 | erkannte GT-Kratzer | Prediction-Komponenten-Precision |
|---|---:|---:|---:|---:|---:|
| `98_max_flat.png` | 58.6 % | 60.4 % | 59.5 % | 4/4 (100.0 %) | 43.5 % |
| `71_max_flat.png` | 46.8 % | 69.6 % | 56.0 % | 31/40 (77.5 %) | 57.8 % |
| `81_max_flat.png` | 49.4 % | 59.4 % | 53.9 % | 28/38 (73.7 %) | 55.0 % |
| `83_max_flat.png` | 42.2 % | 66.6 % | 51.7 % | 20/22 (90.9 %) | 24.7 % |
| `06_max_flat.png` | 52.7 % | 48.3 % | 50.4 % | 126/180 (70.0 %) | 64.2 % |
| `97_max_flat.png` | 40.7 % | 63.2 % | 49.5 % | 2/2 (100.0 %) | 17.8 % |

## Besonders schwierige Bilder

| Bild | Precision | Recall | F1 | erkannte GT-Kratzer | Prediction-Komponenten-Precision |
|---|---:|---:|---:|---:|---:|
| `102_max_flat.png` | 0.0 % | 0.0 % | 0.0 % | 0/1 (0.0 %) | 0.0 % |
| `103_max_flat.png` | 0.0 % | 0.0 % | 0.0 % | 0/0 (0.0 %) | 0.0 % |
| `100_max_flat.png` | 0.0 % | 0.0 % | 0.0 % | 0/5 (0.0 %) | 0.0 % |
| `95_max_flat.png` | 0.0 % | 0.0 % | 0.0 % | 0/1 (0.0 %) | 0.0 % |
| `92_max_flat.png` | 0.3 % | 6.0 % | 0.5 % | 0/1 (0.0 %) | 1.7 % |
| `107_max_flat.png` | 0.3 % | 16.4 % | 0.6 % | 1/2 (50.0 %) | 2.6 % |
| `105_max_flat.png` | 0.9 % | 15.9 % | 1.7 % | 3/11 (27.3 %) | 9.8 % |
| `75_max_flat.png` | 1.0 % | 17.8 % | 2.0 % | 1/3 (33.3 %) | 1.7 % |

Die großen Unterschiede zwischen den Bildern sind ein wichtiger Befund. Die Methode besitzt **keine über alle Bauteilgeometrien konstante Leistung**.

Beispielsweise zeigt `13_max_flat.png` einen interessanten Fall:

- Scratch Detection Rate: **90.7 %**
- Pixel-F1: **31.1 %**
- Prediction-Komponenten-Precision: **47.4 %**

Damit werden auf diesem Bild sehr viele echte Kratzer grundsätzlich getroffen, gleichzeitig entstehen jedoch zusätzliche Falschdetektionen und die vorhergesagten Masken stimmen nicht vollständig mit der Ground-Truth-Fläche überein.

---

# 8. Was mit dem klassischen CV-Ansatz gut funktioniert

## 8.1 Deutliche, lokal kontrastreiche Kratzer

Der Local-Residual-Ansatz reagiert auf lokale Helligkeitsabweichungen. Dadurch können sowohl helle als auch dunkle Kratzer beziehungsweise kontrastreiche Oberflächenabweichungen gegenüber ihrer unmittelbaren Umgebung erkannt werden.

Diese Eigenschaft ist technisch grundsätzlich ein Vorteil, führt im vorliegenden Evaluationsaufbau jedoch teilweise zu einer scheinbar schlechteren Precision: Dunkle Kratzer oder Oberflächenfehler können von der CV-Methode sinnvoll erkannt werden, obwohl sie aufgrund der bewusst engeren Annotationsdefinition nicht in der Ground Truth enthalten sind.

Besonders gut funktioniert die Detektion bei:

- klar abgegrenzten linearen Strukturen,
- längeren Kratzern,
- ausreichend breiten Kratzern,
- Kratzern auf relativ homogener Metalloberfläche,
- Kratzern mit deutlichem lokalem Kontrast.

Die Größenanalyse bestätigt insbesondere den Vorteil bei größeren und breiteren Kratzern.

## 8.2 Grundsätzliche Lokalisierung ist besser als exakte Segmentierung

Die toleranzbasierte Bewertung zeigt eine deutliche Verbesserung gegenüber der vollständig pixelgenauen Bewertung.

Dies spricht dafür, dass der Ansatz Kratzer häufig **räumlich korrekt lokalisiert**, aber:

- nicht immer die komplette Kratzerlänge erfasst,
- teilweise nur die stärksten Abschnitte erkennt,
- eine andere Linienbreite als die manuelle Annotation erzeugt.

## 8.3 Nachvollziehbarkeit

Ein Vorteil der klassischen CV-Lösung ist, dass alle Verarbeitungsschritte und Fehlermechanismen nachvollziehbar bleiben. Es ist klar, warum eine starke lokale Helligkeitsänderung detektiert wird und an welcher Stelle Filter oder geometrische Ausschlüsse wirken.

---

# 9. Was schlecht funktioniert bzw. typische Fehlerquellen

## 9.1 Bauteilkanten

Bauteilkanten gehören zu den wichtigsten verbleibenden Fehlerquellen.

Aus Sicht des Local-Residual-Filters besitzen sie genau die Eigenschaften, auf die auch bei einem Kratzer reagiert wird:

- starke lokale Helligkeitsänderung,
- häufig lang und schmal,
- teilweise sehr geradlinig.

Der zusätzliche Rand- und Parallelitätsfilter reduziert einen Teil dieser Detektionen. Eine vollständige Trennung ist jedoch schwierig, da echte Kratzer ebenfalls in Kantennähe liegen oder parallel zu einer Kante verlaufen können.

Eine zu aggressive Kantenunterdrückung würde deshalb echte Defekte entfernen.

## 9.2 Bohrungen und kreisförmige Strukturen

Bohrungen erzeugen durch den Übergang zwischen dunklem Loch und hellem Metall sehr starke lokale Kontraste.

Die integrierte Kreiserkennung kann viele dieser Bereiche vorab ausschließen. Die Bohrungen im Datensatz sind jedoch heterogen:

- unterschiedliche Radien,
- unterschiedliche Helligkeiten,
- teilweise nur unvollständige Kreisränder,
- unterschiedliche Randreflexionen,
- dichte Lochmuster.

Daher kann eine einzige Hough-Circle-Konfiguration nicht alle Bohrungen zuverlässig erkennen, ohne gleichzeitig zusätzliche falsche Kreise zu erzeugen.

## 9.3 Schrift und Beschriftungen

Schrift ist für einen rein kontrastbasierten klassischen CV-Ansatz problematisch.

Buchstaben bestehen ebenfalls aus:

- schmalen Linien,
- starken Helligkeitsunterschieden,
- teilweise länglichen Segmenten.

Der Algorithmus besitzt kein semantisches Wissen darüber, dass diese Linien eine Beschriftung und keinen Oberflächendefekt darstellen.

Da Schrift nur auf einem kleinen Teil des Datensatzes vorkommt, wäre eine starke Spezialisierung der gesamten Pipeline auf diesen Sonderfall nicht sinnvoll.

## 9.4 Stark strukturierte Bauteile

Besonders problematisch sind Bauteile mit vielen konstruktiven Kanten, Aussparungen oder Lochmustern.

Bei einzelnen schwierigen Bildern werden zahlreiche Prediction-Komponenten erzeugt, obwohl nur sehr wenige oder gar keine Ground-Truth-Kratzer vorhanden sind.

Dies ist eine grundsätzliche Grenze des gewählten Ansatzes:

> Der Local-Residual-Filter erkennt lokale Bildstrukturen, besitzt aber kein Wissen darüber, ob eine Struktur konstruktiv gewollt oder ein Defekt ist.

## 9.5 Kleine und sehr dünne Kratzer

Die Größenanalyse zeigt eine deutlich geringere Erkennungsrate bei kleinen Kratzern.

Insbesondere sehr dünne Defekte konkurrieren direkt mit:

- Bildrauschen,
- feiner Oberflächentextur,
- kleinen Reflexionen.

Eine empfindlichere Einstellung könnte mehr dieser Kratzer erkennen, würde gleichzeitig aber die False-Positive-Rate erhöhen.

Dies stellt einen zentralen Trade-off der klassischen CV-Pipeline dar.

---

# 10. Gesamtbewertung des eingefrorenen Ansatzes

Der klassische CV-Ansatz ist als **Baseline für Kratzerdetektion gut geeignet**, erreicht jedoch keine robuste, universelle Segmentierung für alle im Datensatz vorkommenden Bauteile.

### Stärken

- keine Trainingsdaten notwendig,
- vollständig nachvollziehbare Verarbeitung,
- gute Erkennung klarer und größerer Kratzer,
- Erkennung heller und dunkler lokaler Defekte,
- gute Grundlage für einen späteren Vergleich mit KI-basierten Methoden,
- mit zunehmender Kratzergröße steigt die Detektionswahrscheinlichkeit deutlich.

### Schwächen und Einschränkungen der Bewertung

- die Ground Truth enthält gemäß Annotationsdefinition nicht sämtliche sichtbaren Oberflächenfehler; dadurch können technisch plausible CV-Detektionen formal als False Positives gewertet werden,
- geringe Trennfähigkeit zwischen Kratzern und konstruktiven Kanten,
- Bohrungen müssen über zusätzliche geometrische Regeln behandelt werden,
- Schrift wird teilweise als Defekt interpretiert,
- stark strukturierte Bauteile erzeugen viele False Positives,
- kleine beziehungsweise dünne Kratzer sind nur begrenzt zuverlässig erkennbar,
- exakte Segmentierungsbreite und -form weichen häufig von der manuellen Ground Truth ab.

### Zentrale Kennzahlen des finalen Standes

| Kennzahl | Wert |
|---|---:|
| strikter Pixel-F1 | **31.3 %** |
| Pixel-F1 bei ±5 px | **44.8 %** |
| Scratch Detection Rate bei ≥15 % Überdeckung | **63.6 %** |
| Prediction-Komponenten-Precision | **33.0 %** |
| 80-%-Schwelle Fläche | **≥ 4744 px** |
| 80-%-Schwelle Länge | **≥ 395,0 px** |
| 80-%-Schwelle Breite | **≥ 11,5 px** |

---

# 11. Fazit

Die finale klassische CV-Pipeline kann einen relevanten Anteil der nach der festgelegten Definition annotierten Kratzer ohne trainiertes Modell erkennen. Ihre Leistungsfähigkeit hängt jedoch stark von Kratzergröße, Kratzerkontrast und Bauteilgeometrie ab.

Bei der Interpretation der absoluten Precision-Werte ist zusätzlich zu beachten, dass die Ground Truth bewusst nur **helle beziehungsweise weiße, linienförmige und ohne extremes Hineinzoomen erkennbare Kratzer** enthält. Der kontrastbasierte CV-Ansatz kann dagegen auch dunkle Kratzer oder andere reale Oberflächenabweichungen hervorheben. Ein Teil der formal berechneten False Positives kann deshalb aus Strukturen bestehen, die visuell plausibel sind, aber außerhalb der Annotationsdefinition liegen.

Die Ergebnisse zeigen insbesondere zwei unterschiedliche technische Grenzen:

1. **Detektionsgrenze kleiner Defekte:** Kleine beziehungsweise sehr dünne Kratzer sind deutlich schwieriger zuverlässig von Rauschen und Oberflächentextur zu unterscheiden.
2. **Semantische Grenze klassischer CV:** Kontrastreiche konstruktive Strukturen wie Bauteilkanten, Bohrungen und Schrift besitzen teilweise dieselben einfachen Bildeigenschaften wie Kratzer. Ohne höheres semantisches Verständnis können sie nur über zusätzliche heuristische Regeln unterschieden werden.

Zusätzlich besteht eine **Grenze der quantitativen Ground-Truth-Bewertung**: Die Kennzahlen messen die Übereinstimmung mit der gewählten Annotationsdefinition und nicht vollständig die Erkennung jedes tatsächlich vorhandenen Oberflächenfehlers.

Der eingefrorene Stand sollte deshalb nicht durch immer weitere Sonderregeln auf die vorhandenen Bilder optimiert werden. Für den Methodenvergleich ist gerade diese verbleibende Grenze relevant: Sie zeigt, welche Detektionsleistung mit einer relativ einfachen, nachvollziehbaren klassischen CV-Pipeline erreichbar ist und an welchen Stellen komplexere beziehungsweise lernende Verfahren einen Vorteil bieten können.
