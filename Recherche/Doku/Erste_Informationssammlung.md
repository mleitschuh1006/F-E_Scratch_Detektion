https://rhengineering.de/innovative-oberflaechenkontrolle-shape-from-shading-ki-in-der-praxis/


- besonders in Metallverarbeitung Unterscheidung zwischen echten Defekten und porzessbedingten Spuren wie Flecken oder Bearbeitungsspuren der vorangegangenen Bearbeitung eine anspruchsvolle Aufgabe
- Projekt zeigt, dass Kombination aus KI und optimalen physikalischen Gegebenheiten eine deutlich präzisere Oberflächenkontrolle ermöglicht
        -> Shape from Shading (SFS) Verfahren


Shape from Shading Verfahren (SFS)
- Rekonstruktionder dreidimensionalen Form von Oberflächen basierend auf Licht- und Schatteninformationen
    - ermöglicht es Fehler, Poren oder Kratzer klar und deutlich sichtbar zu machen

- bildgebende Analyse, die nicht nur reflektierte Licht einer Oberfläche betrachtet, sondern auch feine Krümmungsunterschiede sichtbar macht
- Störungen verschwanden im Krümmungsbild des SFS fast vollständig, echte Defekte bleiben deutlich sichtbar

- Verfahren:
    - 4 Bilder, die aus untersschiedlichen Beleuchtungsrichtungen aufgenommen wurden, werden nach speziellen Algorithmus zu einem neuenn Bild verrechnet


Hochauflösende Kameratechnik und Patch-Verarbeitung:
- Herausforderung:
    - aufgrund Bauteilgröße (250x250 mm) nicht möglich KI Fehlerdetektion oder Anomalieerkennung über gesamten Bildausschnitt zu kontrollieren
    Bildausschnitt im Verhältnis zu detektierenden Fehlern (kleiner als 1mm) zu groß und Erkennungsgenauigkeit dadurch zu grob geworden wäre
- Lösung:
    - hochauflösende Kameratechnik
    - Aufteilung des erfassten Bildausschnitts in mehrere kleine Patches 
    - einzelne Analyse der einzelnen Patches
    ![Bild der Beleuchtungsrichtungen](image.png)





## Paper: Surface inspection system for industrial components based on shape from shading minimization approach

https://www.spiedigitallibrary.org/journals/optical-engineering/volume-56/issue-12/123105/Surface-inspection-system-for-industrial-components-based-on-shape-from/10.1117/1.OE.56.12.123105.full

Shape from Shading (SFS) eines der grundlegenden und klassischen 3D-Foermwiederherstellungsprobleme in der CV

Erlangung von Tiefeninformationen der Oberflächen mit Formrückgewinnungsmethoden hängt von einigen Faktoren ab, wie z.B. Richtung, spektrale Energieverteilung und räumliche Position der Beleuchtung, Oberflächenreflektivität, Geometrie der Oberfläche, projiziertes System usw.

SFS Beschäftigt sich mit Wiederherstellung der Form von Schattierungsvariationen
Durch Verwendung bekannter oder geschätzter Szeneninformationen sind SFS-Methoden in der Lage zufriedenstellende geometrische Details des Objekts wiederherzustellen



### Paper Applications of Deep Learning to Metal Surface Defect Detection: Status and Challenges
siehe papers/Applications_2026.pdf
Komplette Abbildung des aktuellen State of the Art, aktuelle Challenges, existierende datensätze etc



### Paper SCS-YOLOv8: steel surface defect detection algorithm based on improved YOLOv8n
https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13993/139932R/SCS-YOLOv8--steel-surface-defect-detection-algorithm-based-on/10.1117/12.3092194.full
eher basierend auf Stahlprodukte und denke eher nicht auf Aluminium
Stahloberflächendefekterkennung mittels Algorithmen fallen hauptsächlich in zwei Kategorien:
- zweistufe Algorithmen:
    - R-CNN und Faster R-CNN
    - erzeugen interessante Regionen und klassifizieren und verfeineren sie dann
    - bietet hohe Genauigkeit
    - komplex und weniger für Echtzeitanforderungen geeignet
    
- einstufige Algorithmen:
    - YOLO und SSD
    - Erkennen Objekte in einem einzigen Durchgang
    - priorisieren Geschwindigkeit und Einfachheit
    - Genauigkeitsabstriche


Versuchsdurchführung des optimierten YOLO Modells mittels NEU-DET Stahloberflächendefektdatensatz
 - erhält 6 gängige Oberflächendefekttypen, nämlich: Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches




## Datensätze:
- NEU Surface Defect Database (Northeastern University):
    - Standard-Datensatz für metallische Defekte
    - erhält 1800 Bilder von sechs Defektklassen

- KolektorSDD / KolektorSDD2:
    - erhält sehr feine Kratzer auf industriellen Bauteilen
    - bietet pixelgenaue Masken für Segmentierung

- DAGM 2007:
    - klassicher Datensatz für Defekte auf verschiedenen Texturen


    ### Paper Surface weak scratch detection for optical elements based on a multimodal imaging system and a depp encoder-decoder network
    https://pubmed.ncbi.nlm.nih.gov/37706778/