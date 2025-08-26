# Usage Instructions

## Installing the Wheel File Directly

If you prefer to install the `flexynesis-mps` package directly, you can use the wheel file provided in the `dist` directory. Run the following command:

```sh
pip install path/to/your/flexynesis-mps/dist/flexynesis_mps-1.0.0-py3-none-any.whl
```

## Running Flexynesis Natively

1. Clone the repository:
   ```sh
   git clone https://github.com/huseyincavusbi/flexynesis-mps-std.git
   cd flexynesis-mps-std
   ```

2. Create and activate a virtual environment using `uv`:
   ```sh
   uv venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies from the `pyproject.toml` file:
   ```sh
   uv pip install -e .
   ```
   This will install all dependencies specified in `pyproject.toml` and install the package in editable mode.

4. Run Flexynesis:
   ```sh
   python3 -m flexynesis
   ```

5. Try it out:

You can find and try example notebooks in the [mps-notebooks folder](../mps-notebooks/). Click the link to explore and run the notebooks directly.

Or open your Jupyter notebook as usual:

```sh
jupyter notebook
```

## Running the Docker Image

> **Note:** MPS acceleration is only available when running natively on macOS. It is not available inside Docker containers.

To use the `flexynesis-mps` Docker image, follow these steps:

1. Pull the image from Docker Hub:
   ```sh
   docker pull huseyincavus/flexynesis-mps:latest
   ```

2. Run the container (interactive shell):
   ```sh
   docker run --rm -it huseyincavus/flexynesis-mps:latest
   ```

3. Mount a local folder (e.g., for notebooks/data):
   ```sh
   docker run --rm -it -v path/to/your/folder:/app/your-folder huseyincavus/flexynesis-mps:latest
   ```

4. Launch Jupyter Notebook inside the container:
   ```sh
   jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
   ```
   Then open the provided URL in your browser (http://127.0.0.1:8888/...).