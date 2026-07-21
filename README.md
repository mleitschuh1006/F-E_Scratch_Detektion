# F-E_Scratch_Detektion

# Baumer Camera Setup

## Requirements

* Ubuntu 22.04
* Python 3.10
* Conda or Miniconda
* Baumer camera
* `Baumer_Camera_Explorer_3.5.2_lin_x86_64.deb`
* `Baumer_neoAPI_1.6.0_lin_x86_64_python.tar.gz`

## 1. Install Baumer Camera Explorer

```bash
cd ~/Downloads
sudo apt update
sudo apt install ./Baumer_Camera_Explorer_3.5.2_lin_x86_64.deb
```

If dependencies are missing:

```bash
sudo apt --fix-broken install
```

Start Baumer Camera Explorer and verify that the camera is detected and provides a live image.

## 2. Extract neoAPI

```bash
mkdir -p ~/Downloads/baumer_neoapi

tar -xzf ~/Downloads/Baumer_neoAPI_1.6.0_lin_x86_64_python.tar.gz \
    -C ~/Downloads/baumer_neoapi
```

Check the extracted files:

```bash
find ~/Downloads/baumer_neoapi -type f
```

## 3. Create Conda Environment

```bash
conda create -n baumer_cam python=3.10 -y
conda activate baumer_cam
```

Verify the Python version:

```bash
python --version
```

Expected output:

```text
Python 3.10.x
```

## 4. Install Python Packages

```bash
python -m pip install --upgrade pip
pip install numpy opencv-python
```

## 5. Install Baumer neoAPI

Navigate to the extracted neoAPI directory:

```bash
cd ~/Downloads/baumer_neoapi
```

Install the Python wheel:

```bash
pip install ./neoapi-*.whl
```

If the wheel is located in a subdirectory:

```bash
find . -name "*.whl"
pip install ./path/to/neoapi-wheel-file.whl
```

## 6. Verify Installation

```bash
python -c "import neoapi, numpy, cv2; print('Baumer camera environment is ready')"
```

Before running camera scripts, activate the environment:

```bash
conda activate baumer_cam
```
# Notizen 21.07 erste Tests:
- größte Platte: 27,5cm x 16cm
-> Kamerahöhe auf 37,5cm von Holzplatte bis Objektiv Anfang

Beleuchtung Höhe Unterkante Aluprofil über Holzplatte: 6cm
- Datensatz 1