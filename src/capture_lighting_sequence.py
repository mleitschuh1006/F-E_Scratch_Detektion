import json
import socket
import time
from pathlib import Path

import neoapi
import yaml


CAMERA_CONFIG_FILE = Path("config.yaml")
LED_CONFIG_FILE = Path("led_config.yaml")


def load_yaml(file_path: Path) -> dict:
    """Load a YAML configuration file."""
    with file_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid YAML configuration: {file_path}"
        )

    return config


def receive_json_line(
    connection: socket.socket,
    receive_buffer: str,
) -> tuple[dict, str]:
    """Receive exactly one newline-terminated JSON message."""
    while "\n" not in receive_buffer:
        received_data = connection.recv(4096)

        if not received_data:
            raise ConnectionError(
                "Connection to Raspberry Pi was closed"
            )

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
    camera_config = config.get("camera")

    if not isinstance(camera_config, dict):
        raise ValueError(
            "Section 'camera' missing in config.yaml"
        )

    for feature_name, value in camera_config.items():
        feature = getattr(camera.f, feature_name)

        if isinstance(value, str):
            feature.SetString(value)
        else:
            feature.Set(value)

        print(f"{feature_name} = {value}")


def create_output_directory(directory: str) -> Path:
    """Create and return the output directory."""
    output_directory = Path(directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_directory


def get_next_series_number(
    output_directory: Path,
) -> int:
    """
    Determine the next available capture-series number.

    Example:
        01_top.png
        01_down.png
        02_top.png

    The next series number would be 3.
    """
    existing_series_numbers = []

    for file_path in output_directory.glob("*_*"):
        if not file_path.is_file():
            continue

        prefix = file_path.stem.split("_", 1)[0]

        if prefix.isdigit():
            existing_series_numbers.append(int(prefix))

    if not existing_series_numbers:
        return 1

    return max(existing_series_numbers) + 1


# =========================
# Load configuration
# =========================

camera_config = load_yaml(CAMERA_CONFIG_FILE)
led_config = load_yaml(LED_CONFIG_FILE)

pi_config = led_config["pi"]
capture_config = led_config["capture"]
lighting_steps = led_config["lighting"]


# =========================
# Raspberry Pi settings
# =========================

pi_host = pi_config["host"]
pi_port = int(
    pi_config.get("port", 5000)
)
pi_timeout = float(
    pi_config.get("timeout_seconds", 5)
)


# =========================
# Capture settings
# =========================

settling_time = float(
    capture_config.get(
        "settling_time_seconds",
        0.2,
    )
)

discard_frames = int(
    capture_config.get(
        "discard_frames",
        1,
    )
)

output_directory = create_output_directory(
    capture_config.get(
        "output_directory",
        "images",
    )
)

series_number = get_next_series_number(
    output_directory
)

number_of_images = len(lighting_steps)

print(
    f"Capture series: {series_number:02d}"
)
print(
    f"Number of lighting positions: {number_of_images}"
)


# =========================
# Camera and capture sequence
# =========================

camera = neoapi.Cam()
receive_buffer = ""

try:
    camera.Connect()
    print("Camera connected")

    configure_camera(
        camera,
        camera_config,
    )

    with socket.create_connection(
        (pi_host, pi_port),
        timeout=pi_timeout,
    ) as connection:
        print(
            f"Connected to Raspberry Pi: "
            f"{pi_host}:{pi_port}"
        )

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
                color = lighting_step.get(
                    "color",
                    [255, 255, 255],
                )

                print()
                print(
                    f"Capture "
                    f"{image_index}/{number_of_images}: "
                    f"{lighting_name}"
                )

                receive_buffer = send_command(
                    connection,
                    {
                        "command": "set_leds",
                        "leds": led_indices,
                        "color": color,
                    },
                    receive_buffer,
                )

                print(
                    f"Active LEDs: {led_indices}"
                )

                # Wait until lighting and exposure
                # conditions are stable.
                time.sleep(settling_time)

                # Discard old frames that may still
                # be in the camera buffer.
                for _ in range(discard_frames):
                    camera.GetImage()

                image = camera.GetImage()

                output_path = (
                    output_directory
                    / f"{series_number:02d}_{lighting_name}"
                )

                if output_path.with_suffix(".bmp").exists():
                    raise FileExistsError(
                        f"Output file already exists: "
                        f"{output_path.with_suffix('.bmp')}"
                    )

                image.Save(str(output_path))

                print(
                    f"Image saved: "
                    f"{output_path.with_suffix('.bmp')}"
                )

            print()
            print("Capture sequence completed")

        finally:
            # Switch LEDs off even if an error occurs
            # during image acquisition.
            try:
                receive_buffer = send_command(
                    connection,
                    {"command": "off"},
                    receive_buffer,
                )
                print("All LEDs switched off")

            except Exception as led_error:
                print(
                    "Warning: LEDs could not be "
                    f"switched off: {led_error}"
                )

finally:
    if camera.IsConnected():
        camera.Disconnect()
        print("Camera disconnected")