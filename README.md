# TIL-25 Data Chefs - Main Repository

**Hackathon:** TIL-25 Hackathon
**Team:** Data Chefs
**Author:** lolkabash

## 📖 Overview

Welcome to the main repository for the "Data Chefs" team participating in the TIL-25 Hackathon! This repository serves as a central hub or provides overarching project components related to our submissions for various challenges.

*(You can elaborate here if this repository serves a specific purpose like a unified deployment, shared utilities, or overall project documentation not specific to one challenge.)*

## 💻 Technologies Used

*   **Python:** Core programming language for scripts and utilities.
*   **Roff:** Potentially used for generating documentation or man pages.
*   **Dockerfile:** Suggests containerization for deployment or a consistent development environment for shared tools.
*   **(Mention any other key libraries, frameworks, or tools used across the project if this repo contains shared code.)**

## 🏆 Hackathon Challenges & Solutions

Our team, Data Chefs, participated in several challenges during the TIL-25 Hackathon. Below is a summary of our efforts and links to the respective repositories:

1.  **OCR Challenge**
    *   **Description:** Training an OCR model.
    *   **Repository:** [lolkabash/til-25-data-chefs-OCR](https://github.com/lolkabash/til-25-data-chefs-OCR)
    *   **Key Technologies:** Python, Jupyter Notebooks, Shell.
    *   *(Briefly summarize the solution or key achievement for OCR)*

2.  **Reinforcement Learning (RL) Challenge**
    *   **Description:** Training an RL model.
    *   **Repository:** [lolkabash/til-25-data-chefs-RL](https://github.com/lolkabash/til-25-data-chefs-RL)
    *   **Key Technologies:** Jupyter Notebooks (Python).
    *   *(Briefly summarize the solution or key achievement for RL)*

3.  **Surprise Challenge**
    *   **Description:** Solving the surprise challenge.
    *   **Repository:** [lolkabash/til-25-data-chefs-surprise](https://github.com/lolkabash/til-25-data-chefs-surprise)
    *   **Key Technologies:** Python, Shell, Docker.
    *   *(Briefly summarize the solution or key achievement for the Surprise Challenge)*

4.  **Computer Vision (CV) Challenge**
    *   **Description:** Training a CV model.
    *   **Repository:** [lolkabash/til-25-data-chefs-CV](https://github.com/lolkabash/til-25-data-chefs-CV)
    *   **Key Technologies:** Jupyter Notebooks, Python, PowerShell, Shell.
    *   *(Briefly summarize the solution or key achievement for CV)*

## ⚙️ Purpose of This Repository (til-25-data-chefs)

*(This section needs to be filled based on the actual content and purpose of THIS specific repository. Here are some possibilities:)*

*   **Option A: Overall Documentation & Coordination**
    *   This repository might contain overall project planning documents, final presentation materials, or links to team resources.
    *   The Roff files could be for generating comprehensive documentation about the team's approach or individual projects.

*   **Option B: Shared Utilities / Deployment**
    *   If there are common Python scripts, utilities, or a Dockerized environment used across multiple challenges, they might reside here.
    *   The Dockerfile could be for a shared development environment or a final deployment of a combined solution/demo.

*   **Option C: Meta-Repository / Submodule Management (less likely given separate repos)**
    *   If this repo used Git submodules to manage the individual challenge repos (though you have them separate).

**(Please clarify the primary role of this 'til-25-data-chefs' repository so this section can be more accurate.)**

## 🚀 Setup and Usage (for this repository)

*(This section will depend heavily on the purpose defined above.)*

### Prerequisites
*   Python (version, e.g., 3.8+)
*   Docker (if applicable)
*   Git
*   (Any tools needed to process Roff files, e.g., `groff`)

### Installation & Setup
1.  Clone the repository:
    ```bash
    git clone https://github.com/lolkabash/til-25-data-chefs.git
    cd til-25-data-chefs
    ```
2.  **(Python dependencies - if applicable)**
    ```bash
    pip install -r requirements.txt
    ```
3.  **(Docker Setup - if applicable)**
    Build the Docker image:
    ```bash
    docker build -t data-chefs-main .
    ```
4.  **(Roff documentation - if applicable)**
    *(Instructions on how to generate documentation from Roff files, e.g.)*
    ```bash
    # man -l docs/some_documentation.roff
    # or groff -ms -Tps docs/some_documentation.roff > output.ps
    ```

### Running / Using This Repository
*(Provide instructions based on what this repository offers, e.g., how to run shared scripts, launch the Docker container, or access documentation.)*

## 📁 File Structure (Example - adapt as needed)

```
til-25-data-chefs/
├── docs/                       # Documentation files (possibly Roff files)
├── src/                        # Shared Python source code/utilities (if any)
├── scripts/                    # General utility scripts
├── Dockerfile                  # Docker configuration (if any)
├── requirements.txt            # Python dependencies (if any)
├── .gitignore
└── README.md
```
*(Adjust the file structure to match your actual repository layout.)*

---

# DSTA BrainHack TIL-AI 2025

![Banner for TIL-AI](https://static.wixstatic.com/media/b03c31_bdb8962d37364d7c8cc3e6ae234bb172~mv2.png/v1/crop/x_0,y_1,w_3392,h_1453/fill/w_3310,h_1418,al_c,q_95,usm_0.66_1.00_0.01,enc_avif,quality_auto/Brainhack%20KV_v12_FOR_WEB.png)

**Contents**
- [DSTA BrainHack TIL-AI 2025](#dsta-brainhack-til-ai-2025)
  - [Get started](#get-started)
  - [Understanding this repo](#understanding-this-repo)
  - [Build, test, and submit](#build-test-and-submit)
  - [Links](#links)

## Get started

Here's a quick overview of the initial setup instructions. You can find a more detailed tutorial, including advanced usage for power users, in the [Wiki](https://github.com/til-ai/til-25/wiki).

Use this repository as a template to create your own, and clone it into your Vertex AI workbench. You'll want to keep your repository private, so you'll need to [create a GitHub Personal Access Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).

You'll also need to initialize the Git submodules:

```bash
git submodule update --init
```

This repository requires Python 3.12 or newer to work. Use `conda` to create a virtual environment with a Python version of at least 3.12.

```bash
conda create --name til python=3.12
conda activate til
```

Finally, install the development dependencies into your newly created virtual environment.

```bash
pip install -r requirements-dev.txt
```

## Understanding this repo

There's a subdirectory for each challenge: [`asr/`](/asr), [`cv/`](/cv), [`ocr/`](/ocr/), and [`rl/`](/rl). Each contains:

* A `src/` directory, where your code lives.
  * `*_manager.py`, which manages your model. This is where your inference and computation takes place.
  * `*_server.py`, which runs a local web server that talks to the rest of the competition infrastructure.
* `Dockerfile`, which is used to build your Docker image for each model.
* `requirements.txt`, which lists the dependencies you need to have bundled into your Docker image.
* `README.md`, which contains specifications for the format of each challenge.

You'll also find a final subdirectory, [`test/`](/test). This contains tools to test and score your model locally.

There are also two Git submodules, `til-25-finals` and `til-25-environment`. `finals` contains code that will be pulled into your repo for Semifinals and Finals. `environment` contains the `til_environment` package, which will help you train and test your RL model, and is installed by pip during setup. Don't delete or modify the contents of `til-25-finals/`, `til-25-environment/`, or `.gitmodules`.

## Build, test, and submit

Submitting your model for evaluation is simple: just build your Docker image, test it, and submit. You can find a more detailed tutorial, including advanced usage for power users, in the [Wiki](https://github.com/til-ai/til-25/wiki).

You'll first want to `cd` into the directory you want to build. Then, build the image using Docker, following the naming scheme below. You should then run and test your model before using `til submit` to submit your image for evaluation.

```bash

# cd into the directory. For example, `cd ./asr/`
cd CHALLENGE

# Build your image. Remember the . at the end.
docker build -t TEAM_ID-CHALLENGE:TAG .

# Optionally, you can run your model locally to test it.
docker run -p PORT --gpus all -d IMAGE_NAME:TAG
python test/test_CHALLENGE.py

# Push it for submission
til submit TEAM_ID-CHALLENGE:TAG
```

## Links

* The repo [Wiki](https://github.com/til-ai/til-25/wiki) contains tutorials, specifications, resources, and more.
* Your [Vertex AI Workbench](https://console.cloud.google.com/vertex-ai/workbench/instances?project=til-ai-2025) on Google Cloud Platform is where you'll do most of your development.
* The [Guardian's Handbook](https://tribegroup.notion.site/BrainHack-2025-TIL-AI-Guardian-s-Handbook-1885263ef45a80fdb547d0f22741a5ba) houses the Leaderboard and info about the competition.
* [TIL-AI Curriculum](https://drive.google.com/drive/folders/18zP4pHt5E6YqA3usey16ETEzKNeAn5X9) on Google Drive contains educational materials specially crafted for TIL-AI.
* The [#hackoverflow](https://discord.com/channels/1344138493357719573/1344204681110487068) channel on the TIL-AI Discord server is a forum just for Guardians like you.

---

Code in this repo is licensed under the MIT License.
