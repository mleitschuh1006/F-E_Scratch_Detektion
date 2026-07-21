from pathlib import Path

import neoapi
import yaml


with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

camera = neoapi.Cam()

try:
    camera.Connect()
    print("Camera connected")

    for feature_name, value in config["camera"].items():
        feature = getattr(camera.f, feature_name)

        if isinstance(value, str):
            feature.SetString(value)
        else:
            feature.Set(value)

        print(f"{feature_name} = {value}")

    image = camera.GetImage()

    output_path = Path("images/test_image")
    output_path.parent.mkdir(exist_ok=True)

    image.Save(str(output_path))
    print(f"Image saved: {output_path}")

finally:
    if camera.IsConnected():
        camera.Disconnect()