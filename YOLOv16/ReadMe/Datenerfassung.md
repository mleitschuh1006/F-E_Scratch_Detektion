# Datenaufnahme

## Übersicht

Die Bilddaten werden mit einem **Engineering PC**, einem **Raspberry Pi 3**, einer **Baumer Industriekamera** und einer **WS2812B-LED-Kette** aufgenommen.

Der Engineering PC steuert den gesamten Aufnahmeablauf. Die Baumer Kamera ist direkt mit dem Engineering PC verbunden. Die Beleuchtung wird über den Raspberry Pi angesteuert.

## Hardwareaufbau

Der Versuchsaufbau besteht aus:

- Engineering PC
- Baumer Industriekamera
- Raspberry Pi 3
- Logikpegelwandler von 3,3 V auf 5 V
- WS2812B-LED-Kette
- höhenverstellbarer Beleuchtung

Der Raspberry Pi liefert ein 3,3-V-Steuersignal. Über einen Logikpegelwandler wird dieses auf den für die WS2812B-LEDs benötigten 5-V-Pegel umgesetzt.

## Ablauf der Datenaufnahme

Das Aufnahmeskript wird auf dem Engineering PC ausgeführt. Dieser steuert sowohl die Kamera als auch den Ablauf der Beleuchtung.

Für jede Aufnahmeserie wird folgender Ablauf durchgeführt:

1. Der Engineering PC verbindet sich mit der Baumer Kamera und übernimmt die in `config.yaml` definierten Kameraparameter.
2. Der Engineering PC sendet die gewünschte Beleuchtungskonfiguration über TCP an den Raspberry Pi.
3. Der Raspberry Pi aktiviert die entsprechenden LEDs der WS2812B-LED-Kette.
4. Nach einer kurzen Einschwingzeit wird ein Bild mit der Baumer Kamera aufgenommen.
5. Der Vorgang wird für alle definierten Beleuchtungsrichtungen wiederholt.
6. Nach Abschluss der Aufnahmeserie werden alle LEDs ausgeschaltet.

Die Bilder werden automatisch anhand der jeweiligen Beleuchtungsrichtung benannt, beispielsweise:

```text
01_top.bmp
01_right.bmp
01_bottom.bmp
01_left.bmp
```

Die einzelnen Beleuchtungsrichtungen und die dafür verwendeten LEDs sind in `led_config.yaml` definiert.

## Konfiguration

Die Kamera- und Versuchsparameter werden in `config.yaml` gespeichert.

Dazu gehören unter anderem:

- Beleuchtungshöhe
- LED-Helligkeit
- Kamerahöhe
- Blende
- Fokusdistanz
- Belichtungszeit
- Gain
- Pixelformat

Die Verbindung zum Raspberry Pi sowie die Beleuchtungspositionen und Aufnahmeparameter werden in `led_config.yaml` definiert.

## Automatische Dokumentation der Parameter

Für jede vollständige Aufnahmeserie werden die verwendeten Kamera- und Versuchsparameter automatisch in

```text
overview_images_parameters.csv
```

gespeichert.

Dadurch kann jede Aufnahmeserie eindeutig der verwendeten Konfiguration zugeordnet werden.

## Bestimmung der finalen Aufnahmeparameter

Für die Erstellung des Testdatensatzes wurden unterschiedliche Kamera- und Beleuchtungsparameter untersucht. Auf Grundlage dieser Versuche wurde folgende Konfiguration für die weitere Datenaufnahme ausgewählt:

```text
Beleuchtungshöhe:       60,0 mm
LED-Helligkeit:         1,0
Kamerahöhe:             375,0 mm
Blende:                 f/5,5
Fokusdistanz:           0,37 m
Acquisition Mode:       Continuous
Trigger Mode:           Off
Pixelformat:            BayerRG8
Exposure Auto:          Off
Belichtungszeit:        30000 µs
Gain Auto:              Off
Gain:                   0,0
Balance White Auto:     Off
```

Diese Konfiguration wird für die Aufnahme des finalen Datensatzes verwendet.