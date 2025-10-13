# Real-Time Face Recognition System

This is a web-based, real-time face recognition application built with Python, Flask, and FaceNet.

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create Conda Environment:**
    This project requires Python 3.10. Create a new Conda environment with this command:
    ```bash
    conda create -n face_env python=3.10 pip
    ```

3.  **Activate the Environment:**
    ```bash
    conda activate face_env
    ```

4.  **Install Dependencies (IMPORTANT: Two-Step Process):**
    First, install the `dlib` dependency from conda-forge to avoid compilation issues:
    ```bash
    conda install -c conda-forge dlib
    ```
    Then, install the rest of the packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Application:**
    ```bash
    python app.py
    ```
    The web server will start at `http://localhost:5000`. The first time it runs, it will create a `face_embeddings.npz` file.

## How to Use

1.  Add known faces by creating a folder for each person inside the `custom_faces` directory and placing their photos inside.
2.  Run the application.
3.  Open your web browser to `http://localhost:5000` to see the live feed.