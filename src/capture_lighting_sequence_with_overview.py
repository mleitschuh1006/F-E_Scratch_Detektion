import csv
import json
import re
import socket
import time
from datetime import datetime
from pathlib import Path

import neoapi
import yaml


CAMERA_CONFIG_FILE = Path("config.yaml")
LED_CONFIG_FILE = Path("led_config.yaml")
OVERVIEW_FILE_NAME = "overview_images_parameters.csv"


def load_yaml(file_path: Path) -> dict:
    """Load a YAML configuration file."""
    with file_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration: {file_path}")

    return config


def receive_json_line(
    connection: socket.socket,
    receive_buffer: str,
) -> tuple[dict, str]:
    """Receive exactly one newline-terminated JSON message."""
    while "\n" not in receive_buffer:
        received_data = connection.recv(4096)

        if not received_data:
            raise ConnectionError("Connection to Raspberry Pi was closed")

        receive_buffer += received_data.decode("utf-8")

    message, receive_buffer = receive_buffer.split("\n", 1)

    return json.loads(message), receive_buffer


def send_command(
    connection: socket.socket,
    command: dict,
    receive_buffer: str,
) -> str:
    """Send a command and wait for the Raspberry Pi response."""
    message = json.dumps(command) + "\n"
    connection.sendall(message.encode("utf-8"))

    response, receive_buffer = receive_json_line(
        connection,
        receive_buffer,
    )

    if response.get("status") != "ready":
        error_message = response.get(
            "message",
            "Unknown Raspberry Pi error",
        )
        raise RuntimeError(error_message)

    return receive_buffer


def configure_camera(
    camera: neoapi.Cam,
    config: dict,
) -> None:
    """Apply all camera settings from config.yaml."""
    camera_settings = config.get("camera")

    if not isinstance(camera_settings, dict):
        raise ValueError("Section 'camera' missing in config.yaml")

    for feature_name, value in camera_settings.items():
        feature = getattr(camera.f, feature_name)

        if isinstance(value, str):
            feature.SetString(value)
        else:
            feature.Set(value)

        print(f"{feature_name} = {value}")


def create_output_directory(directory: str) -> Path:
    """Create and return the output directory."""
    output_directory = Path(directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    return output_directory


def get_next_series_number(output_directory: Path) -> int:
    """
    Determine the next capture-series number from existing image files.

    Examples:
        15_top.bmp
        16_left.bmp
        16_right.bmp

    The next series number is 17.
    """
    existing_series_numbers = []
    filename_pattern = re.compile(r"^(\d+)_.*\.bmp$", re.IGNORECASE)

    for file_path in output_directory.iterdir():
        if not file_path.is_file():
            continue

        match = filename_pattern.match(file_path.name)

        if match:
            existing_series_numbers.append(int(match.group(1)))

    if not existing_series_numbers:
        return 1

    return max(existing_series_numbers) + 1


def validate_brightness(brightness: float) -> float:
    """Validate that LED brightness is between 0.0 and 1.0."""
    if not 0.0 <= brightness <= 1.0:
        raise ValueError(
            "experiment.led_brightness must be between 0.0 and 1.0"
        )

    return brightness


def scale_color(color: list[int], brightness: float) -> list[int]:
    """Scale an RGB color using a brightness factor from 0.0 to 1.0."""
    if len(color) != 3:
        raise ValueError(f"Invalid RGB color: {color}")

    return [
        max(0, min(255, round(int(channel) * brightness)))
        for channel in color
    ]


def format_csv_value(value: object) -> object:
    """Convert lists and dictionaries to compact JSON strings for CSV."""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    return value


def append_overview_row(csv_path: Path, row: dict) -> None:
    """
    Append one image-parameter row to the overview CSV.

    If new parameter columns are introduced later, the existing CSV is
    automatically extended while preserving its previous rows.
    """
    normalized_row = {
        key: format_csv_value(value)
        for key, value in row.items()
    }

    existing_rows = []
    fieldnames = list(normalized_row.keys())

    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file, delimiter=";")
            existing_fieldnames = reader.fieldnames or []
            existing_rows = list(reader)

        fieldnames = existing_fieldnames + [
            key for key in normalized_row if key not in existing_fieldnames
        ]

    file_mode = "w" if existing_rows or not csv_path.exists() else "a"

    with csv_path.open(file_mode, encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            delimiter=";",
            extrasaction="ignore",
        )

        if file_mode == "w":
            writer.writeheader()
            writer.writerows(existing_rows)
        elif csv_path.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(normalized_row)


def create_overview_row(
    dataset_name: str,
    series_number: int,
    experiment_config: dict,
    camera_settings: dict,
) -> dict:
    """Create exactly one CSV row for a complete capture series."""
    row = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_name": dataset_name,
        "series_number": series_number,
    }

    # Write every parameter from the experiment section automatically.
    # This means newly added YAML values do not require another code change.
    for parameter_name, value in experiment_config.items():
        if parameter_name == "dataset_name":
            continue
        row[parameter_name] = value

    for feature_name, value in camera_settings.items():
        row[f"camera_{feature_name}"] = value

    return row


# =========================
# Load configuration
# =========================

config = load_yaml(CAMERA_CONFIG_FILE)
led_config = load_yaml(LED_CONFIG_FILE)

camera_settings = config.get("camera")
experiment_config = config.get("experiment")

if not isinstance(camera_settings, dict):
    raise ValueError("Section 'camera' missing in config.yaml")

if not isinstance(experiment_config, dict):
    raise ValueError("Section 'experiment' missing in config.yaml")

required_experiment_parameters = (
    "lighting_height_mm",
    "led_brightness",
    "camera_aperture_f_number",
)

for parameter_name in required_experiment_parameters:
    if parameter_name not in experiment_config:
        raise ValueError(
            f"Parameter 'experiment.{parameter_name}' missing in config.yaml"
        )

dataset_name = experiment_config.get("dataset_name")
lighting_height_mm = float(experiment_config["lighting_height_mm"])
led_brightness = validate_brightness(
    float(experiment_config["led_brightness"])
)
camera_aperture_f_number = float(
    experiment_config["camera_aperture_f_number"]
)

# Store normalized numeric values for the CSV output.
experiment_config["lighting_height_mm"] = lighting_height_mm
experiment_config["led_brightness"] = led_brightness
experiment_config["camera_aperture_f_number"] = camera_aperture_f_number

pi_config = led_config["pi"]
capture_config = led_config["capture"]
lighting_steps = led_config["lighting"]


# =========================
# Raspberry Pi settings
# =========================

pi_host = pi_config["host"]
pi_port = int(pi_config.get("port", 5000))
pi_timeout = float(pi_config.get("timeout_seconds", 5))


# =========================
# Capture settings
# =========================

settling_time = float(
    capture_config.get("settling_time_seconds", 0.2)
)
discard_frames = int(capture_config.get("discard_frames", 1))

output_directory = create_output_directory(
    capture_config.get("output_directory", "images")
)
overview_csv_path = output_directory / OVERVIEW_FILE_NAME

# If no dataset name is configured, use the image folder name.
if dataset_name is None or not str(dataset_name).strip():
    dataset_name = output_directory.name
else:
    dataset_name = str(dataset_name).strip()

series_number = get_next_series_number(output_directory)
number_of_images = len(lighting_steps)

print(f"Dataset: {dataset_name}")
print(f"Capture series: {series_number:02d}")
print(f"Number of lighting positions: {number_of_images}")
print(f"Lighting height: {lighting_height_mm} mm")
print(f"LED brightness: {led_brightness:.2f}")
print(f"Camera aperture: f/{camera_aperture_f_number}")
print(f"Overview CSV: {overview_csv_path}")


# =========================
# Camera and capture sequence
# =========================

camera = neoapi.Cam()
receive_buffer = ""

try:
    camera.Connect()
    print("Camera connected")

    configure_camera(camera, config)

    with socket.create_connection(
        (pi_host, pi_port),
        timeout=pi_timeout,
    ) as connection:
        print(f"Connected to Raspberry Pi: {pi_host}:{pi_port}")

        receive_buffer = send_command(
            connection,
            {"command": "ping"},
            receive_buffer,
        )

        try:
            for image_index, lighting_step in enumerate(
                lighting_steps,
                start=1,
            ):
                lighting_name = lighting_step["name"]
                led_indices = lighting_step["leds"]
                configured_color = lighting_step.get(
                    "color",
                    [255, 255, 255],
                )
                effective_color = scale_color(
                    configured_color,
                    led_brightness,
                )

                print()
                print(
                    f"Capture {image_index}/{number_of_images}: "
                    f"{lighting_name}"
                )

                receive_buffer = send_command(
                    connection,
                    {
                        "command": "set_leds",
                        "leds": led_indices,
                        "color": effective_color,
                    },
                    receive_buffer,
                )

                print(f"Active LEDs: {led_indices}")
                print(
                    f"LED color: {effective_color} "
                    f"(brightness {led_brightness:.2f})"
                )

                time.sleep(settling_time)

                for _ in range(discard_frames):
                    camera.GetImage()

                image = camera.GetImage()

                output_path_without_suffix = (
                    output_directory
                    / f"{series_number:02d}_{lighting_name}"
                )
                output_path = output_path_without_suffix.with_suffix(".bmp")

                if output_path.exists():
                    raise FileExistsError(
                        f"Output file already exists: {output_path}"
                    )

                image.Save(str(output_path_without_suffix))
                print(f"Image saved: {output_path}")

            overview_row = create_overview_row(
                dataset_name=dataset_name,
                series_number=series_number,
                experiment_config=experiment_config,
                camera_settings=camera_settings,
            )
            append_overview_row(
                overview_csv_path,
                overview_row,
            )

            print()
            print("Capture sequence completed")
            print(f"Overview updated: {overview_csv_path}")

        finally:
            try:
                receive_buffer = send_command(
                    connection,
                    {"command": "off"},
                    receive_buffer,
                )
                print("All LEDs switched off")

            except Exception as led_error:
                print(
                    "Warning: LEDs could not be switched off: "
                    f"{led_error}"
                )

finally:
    if camera.IsConnected():
        camera.Disconnect()
        print("Camera disconnected")
