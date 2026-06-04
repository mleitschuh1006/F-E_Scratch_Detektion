# Baseline: Kratzerdetektion mit einem leichtgewichtigen YOLO-Modell

Als erste Detektionsbaseline wird ein einstufiges Objektdetektionsmodell der YOLO-Familie verwendet. YOLO-Modelle führen Klassifikation und Lokalisierung in einem einzigen Inferenzschritt durch. Dadurch eignen sie sich besonders für Anwendungen, bei denen eine schnelle Auswertung erforderlich ist. In der Literatur werden YOLO und SSD als typische einstufige Detektoren beschrieben, während R-CNN-, Fast-R-CNN- und Faster-R-CNN-Verfahren zweistufig arbeiten. Einstufige Verfahren sind im Allgemeinen schneller, zweistufige Verfahren erreichen häufig eine höhere Lokalisierungsgenauigkeit.

Für die Baseline wird ein leichtgewichtiges YOLO-Modell mit MobileNet-basiertem Backbone vorgesehen. Die Wahl eines leichten Backbones ist sinnvoll, da das Modell möglichst klein bleiben und später effizient auf industrieller Hardware einsetzbar sein soll. Im Review wird eine angepasste YOLOv4-Variante für Aluminiumstreifen beschrieben, bei der der ursprüngliche Backbone auf Basis von MobileNet neu entworfen wurde. Zusätzlich wurden Channel-Attention- und Spatial-Attention-Mechanismen integriert. Dadurch wurde die Parameterzahl von etwa 64,4 Mio. auf etwa 11,7 Mio. reduziert, während weiterhin eine hohe Detektionsleistung erreicht wurde.

Die Baseline wird daher wie folgt definiert:

| Bestandteil | Festlegung |
|---|---|
| Modellfamilie | YOLO |
| Konkrete Orientierung | leichtgewichtiges YOLOv4-basiertes Detektionsmodell |
| Backbone | MobileNet-basierter Backbone |
| Neck | Feature-Pyramid- bzw. Path-Aggregation-Struktur zur Multi-Scale-Merkmalsfusion |
| Head | YOLO-Detektionskopf |
| Ausgabedaten | Bounding Boxes, Objektkonfidenzen und Klassenwahrscheinlichkeiten |
| Zielklasse | `scratch` |

Die Multi-Scale-Verarbeitung ist für Kratzer besonders wichtig, da diese häufig sehr schmal, länglich und kontrastarm sind. Feature Pyramid Networks kombinieren flache räumliche Informationen mit tieferen semantischen Merkmalen und können dadurch Defekte unterschiedlicher Größe besser erfassen.

## Variante 1: Detektion auf dem Gesamtbild

Im ersten Schritt wird das vollständige Kamerabild direkt mit dem YOLO-Modell verarbeitet. Dieses Vorgehen dient als einfache Referenzvariante.

Ablauf:

1. Das vollständige Kamerabild wird geladen.
2. Das Bild wird auf die Eingangsgröße des YOLO-Modells skaliert.
3. Das YOLO-Modell sagt Bounding Boxes, Konfidenzwerte und die Klasse `scratch` vorher.
4. Die Detektionen werden direkt im Gesamtbild visualisiert.

Der Vorteil dieser Variante liegt in der einfachen Verarbeitungspipeline. Pro Aufnahme ist nur eine einzige Modellinferenz notwendig. Außerdem müssen die vorhergesagten Bounding Boxes nicht aus lokalen Patch-Koordinaten in das Gesamtbild zurückgerechnet werden.

Der Nachteil besteht im möglichen Informationsverlust durch die Skalierung des Gesamtbildes. Feine Kratzer können nur wenige Pixel breit sein. Wenn ein hochauflösendes Bild stark verkleinert wird, können kleine oder kontrastarme Kratzer teilweise verschwinden. Dieses Problem ist besonders relevant bei metallischen Oberflächen, da Reflexionen, Bearbeitungsspuren und geringe Kontraste die Erkennung zusätzlich erschweren.

## Variante 2: Patchbasierte Detektion mit Overlap

Um den Informationsverlust bei kleinen Kratzern zu reduzieren, wird im zweiten Schritt eine patchbasierte Detektion untersucht. Das Originalbild wird dazu in kleinere, gleich große Bildbereiche zerlegt. Jeder Patch wird separat durch dasselbe YOLO-Modell verarbeitet.

Dadurch bleibt die lokale Detailauflösung besser erhalten. Kleine Kratzer erscheinen im Verhältnis zur Eingangsgröße des Modells größer und können dadurch zuverlässiger erkannt werden.

Die Zerlegung erfolgt zunächst über ein festes Raster mit Überlappung. Die Überlappung ist notwendig, damit Kratzer an Patch-Grenzen nicht abgeschnitten oder nur teilweise erkannt werden.

Beispielhafte Parameter:

| Parameter | Beispielwert |
|---|---|
| Patchgröße | `512 x 512 px` |
| Schrittweite | `256 px` |
| Overlap | `50 %` |
| Modell | YOLO mit MobileNet-Backbone |
| Klasse | `scratch` |

Die relative Überlappung ergibt sich aus:

```text
Overlap = 1 - Schrittweite / Patchgröße
```

Beispiel:

```text
Patchgröße = 512 px
Schrittweite = 256 px

Overlap = 1 - 256 / 512 = 0,5 = 50 %
```

Für jeden Patch liefert das YOLO-Modell lokale Bounding Boxes. Diese Koordinaten beziehen sich zunächst nur auf den jeweiligen Patch. Nach der Inferenz werden die lokalen Koordinaten in das Koordinatensystem des ursprünglichen Gesamtbildes zurücktransformiert.

Wenn der Ursprung eines Patches im Gesamtbild bei `(x0, y0)` liegt, dann gilt:

```text
x_global = x_patch + x0
y_global = y_patch + y0
```

Da sich benachbarte Patches überlappen, kann derselbe Kratzer mehrfach detektiert werden. Nach der Rückprojektion aller Bounding Boxes wird deshalb eine Non-Maximum Suppression auf Gesamtbildebene durchgeführt. Dadurch werden Mehrfachdetektionen zusammengeführt und nur die plausibelste Detektion beibehalten.

## Vergleich der beiden Varianten

Die Baseline besteht aus zwei Auswertungsvarianten:

| Variante | Beschreibung | Vorteil | Nachteil |
|---|---|---|---|
| Gesamtbild | YOLO wird direkt auf das vollständige Bild angewendet | Sehr einfache und schnelle Pipeline | Kleine Kratzer können durch Skalierung verloren gehen |
| Patches mit Overlap | Das Bild wird in überlappende Patches zerlegt, jeder Patch wird separat ausgewertet | Bessere Detailauflösung für kleine Kratzer | Mehr Inferenzläufe und Rückprojektion notwendig |

Beide Varianten verwenden dasselbe YOLO-Modell mit MobileNet-basiertem Backbone. Dadurch kann untersucht werden, ob eine mögliche Verbesserung der Detektionsleistung durch die Patch-Verarbeitung entsteht und nicht durch eine veränderte Modellarchitektur.

## Bewertung

Die Bewertung der Baseline erfolgt anhand folgender Metriken:

| Metrik | Bedeutung |
|---|---|
| Precision | Anteil der vorhergesagten Kratzer, die tatsächlich Kratzer sind |
| Recall | Anteil der echten Kratzer, die vom Modell gefunden werden |
| mAP | Detektionsgüte unter Berücksichtigung von Klassifikation und Lokalisierung |
| Inferenzzeit | Laufzeit pro Bild bzw. pro Patch |
| FPS | Bilder bzw. Patches pro Sekunde |
| minimale Kratzergröße | kleinste Kratzergröße, die zuverlässig erkannt wird |

Besonders wichtig ist der Recall, da echte Kratzer im industriellen Prüfprozess möglichst nicht übersehen werden sollen. Gleichzeitig muss die Precision betrachtet werden, damit Reflexionen, Staub oder Bearbeitungsspuren nicht zu vielen Fehlalarmen führen.

## Zusammenfassung der Baseline

Die Baseline untersucht zunächst, ob ein leichtgewichtiges YOLO-Modell mit MobileNet-Backbone Kratzer auf dem vollständigen Bild zuverlässig erkennen kann. Anschließend wird dasselbe Modell auf überlappenden Patches eingesetzt, um kleine und feine Kratzer besser sichtbar zu machen. Die Ergebnisse beider Varianten werden anhand von Precision, Recall, mAP, Inferenzzeit und minimal zuverlässig detektierbarer Kratzergröße verglichen.

## Quellenhinweise

- Liu et al. (2023): *A survey of real-time surface defect inspection methods based on deep learning*.
- Ma et al. (2022): Leichtgewichtige YOLOv4-Variante mit MobileNet-basiertem Backbone, Attention-Mechanismen und reduzierter Parameterzahl für Oberflächendefekte auf Aluminiumstreifen.
- Informationszusammenstellung zum Projekt: Hinweise zu hochauflösender Bildaufnahme, Patch-Verarbeitung und Herausforderungen metallischer Oberflächen.