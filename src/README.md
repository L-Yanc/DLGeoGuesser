# DLGeoGuesser Source Code

This directory contains the source code for the DLGeoGuesser project.

## Directory Structure

- `dl_geoguesser/`: The main Python package for the project.
    - `__init__.py`: Makes `dl_geoguesser` a Python package.
    - `main.py`: The main entry point for the application.
    - `config.py`: For configuration variables.
    - `chat/`: Contains the chat models.
        - `core_model/`: The fine-tuned chat model.
        - `scratch_model/`: The chat model trained from scratch.
        - `sota_model/`: The state-of-the-art chat model.
    - `vision/`: Contains the vision models.
        - `model_one/`: The first vision model.
        - `model_two/`: The second vision model.
        - `explainability/`: For model explanation techniques like heatmaps.
    - `common/`: For shared code and interfaces.
        - `interfaces.py`: Defines abstract base classes for `VisionModel` and `ChatModel` to ensure interoperability.
        - `utils.py`: For shared utility functions.
