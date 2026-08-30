# Datenvorverarbeitung

## Übersicht

Für jedes Bauteil werden mehrere Aufnahmen unter unterschiedlichen Beleuchtungsrichtungen zu einem einzelnen Graustufenbild zusammengeführt. Ziel ist es, Kratzer unabhängig von ihrer Orientierung möglichst deutlich hervorzuheben.

## Verarbeitung

Die ausgewählten Aufnahmen werden zunächst in Graustufenbilder umgewandelt. Anschließend wird der großflächige Beleuchtungsverlauf mithilfe eines Gauß-Tiefpasses bestimmt und aus dem jeweiligen Bild herausgerechnet (`flatten`).

Danach erfolgt eine **pixelweise Maximum-Fusion**, vergleichbar mit einem **Max-Pooling über die verschiedenen Beleuchtungsrichtungen**. Für jeden Pixel wird somit der höchste Intensitätswert aller Aufnahmen übernommen. Kratzer, die unter mindestens einer Beleuchtungsrichtung stark reflektieren, werden dadurch hervorgehoben.

Aufnahmen mit starken Spiegelreflexionen bzw. gesättigten Bildbereichen können automatisch verworfen werden. Abschließend wird das fusionierte Bild normalisiert.

## Benennungsschema

Das fusionierte Ergebnis wird als

```text
[prefix]_max_flat.png
```

gespeichert.

Zusätzlich wird mit

```text
[prefix]_max_flat.config.yaml
```

die verwendete Konfiguration sowie die Auswahl der verwendeten und verworfenen Aufnahmen dokumentiert.
Für die Verarbeitung des gesamten Datensatzes wird `generate_max_flat_batch.py` verwendet.