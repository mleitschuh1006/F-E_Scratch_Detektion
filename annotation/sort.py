from pathlib import Path
import csv
import re


# Ordner, in dem deine Gruppenordner liegen:
#
# gruppen/
# ├── gruppe_001/
# ├── gruppe_002/
# └── ...
GROUPS_DIR = Path("pictures/all/_gruppierung/gruppen")

# Die Dateien werden eine Ebene über dem Gruppenordner gespeichert
OUTPUT_DIR = GROUPS_DIR.parent

VALID_EXTENSIONS = {
    ".bmp",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
}


def numeric_sort_key(text: str):
    """
    Sorgt dafür, dass beispielsweise 2 vor 10 sortiert wird.
    """
    match = re.search(r"\d+", text)

    if match:
        return int(match.group())

    return float("inf")


def get_image_series_name(filename: str) -> str:
    """
    Wandelt beispielsweise '108_all.bmp' in '108' um.
    """
    stem = Path(filename).stem

    if stem.lower().endswith("_all"):
        stem = stem[:-4]

    return stem


def main():
    if not GROUPS_DIR.exists():
        raise FileNotFoundError(
            f"Der Gruppenordner wurde nicht gefunden:\n"
            f"{GROUPS_DIR.resolve()}"
        )

    group_folders = sorted(
        [
            folder
            for folder in GROUPS_DIR.iterdir()
            if folder.is_dir()
        ],
        key=lambda folder: numeric_sort_key(folder.name),
    )

    if not group_folders:
        raise RuntimeError(
            f"Im Ordner {GROUPS_DIR.resolve()} wurden "
            "keine Gruppenordner gefunden."
        )

    assignments = []
    text_lines = []

    for group_folder in group_folders:
        image_files = sorted(
            [
                file
                for file in group_folder.iterdir()
                if (
                    file.is_file()
                    and file.suffix.lower() in VALID_EXTENSIONS
                )
            ],
            key=lambda file: numeric_sort_key(file.name),
        )

        image_series = [
            get_image_series_name(file.name)
            for file in image_files
        ]

        # Beispielsweise "gruppe_001" beibehalten
        group_name = group_folder.name

        if image_series:
            text_lines.append(
                f"{group_name}: {', '.join(image_series)}"
            )
        else:
            text_lines.append(
                f"{group_name}: keine Bilder"
            )

        for series_name in image_series:
            assignments.append({
                "bildreihe": series_name,
                "gruppe": group_name,
            })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Übersichtliche Textdatei
    # --------------------------------------------------------

    text_path = OUTPUT_DIR / "gruppen_uebersicht.txt"

    with text_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write("Zuordnung der Bildreihen\n")
        file.write("========================\n\n")

        for line in text_lines:
            file.write(line + "\n")

    # --------------------------------------------------------
    # Neue CSV-Zuordnung
    # --------------------------------------------------------

    csv_path = OUTPUT_DIR / "zuordnung_neu.csv"

    assignments.sort(
        key=lambda row: numeric_sort_key(row["bildreihe"])
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "bildreihe",
                "gruppe",
            ],
            delimiter=";",
        )

        writer.writeheader()
        writer.writerows(assignments)

    print("Zuordnung wurde neu erstellt.")
    print(f"Textdatei: {text_path.resolve()}")
    print(f"CSV-Datei: {csv_path.resolve()}")
    print(f"Gruppen: {len(group_folders)}")
    print(f"Bildreihen: {len(assignments)}")


if __name__ == "__main__":
    main()