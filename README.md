# Tracking Repository

This repository implements the multi-object tracking component for the AI for FineTech Competition. It focuses on processing surveillance video sequences, detecting and tracking pedestrians, and generating outputs that will later be merged with the face re-identification results in the Integration Repository.

## Overview

- **Objective:** Track pedestrians in video frames and generate output files in the required format.
- **Evaluation Metric:** HOTA (Higher Order Tracking Accuracy)
- **Output:** A CSV file (e.g., `submissions/tracking_results{last_number}.csv`) containing tracking results formatted for integration.

## Directory Structure

```
tracking_repo/
├── data/               # datasets, Scripts and notebooks for data preprocessing and video handling
├── models/             # Model weights and scripts for training.
├── notebooks/          # Jupyter notebooks for experimentation and analysis
├── outputs/            # Folder to store output files (e.g. tracking_results{last_number}.csv)
├── utils/              # Utility functions (e.g., data loaders, HOTA metric evaluation)
├── README.md
└── requirements.txt
```

## Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/tracking_repo.git
   cd tracking_repo
   ```

2. **Create a Virtual Environment and Install Dependencies:**

   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Data Preparation:**

   - Place your video dataset or update the data paths as needed.
   - Run any required preprocessing scripts located in the `data/` folder.

## Usage

1. **Training and Evaluation:**

   - Train your tracking model using the scripts in the `models/` directory.
   - Evaluate the model's performance using the provided evaluation scripts that compute the HOTA metric.

   Example command:

   ```bash
   python models/train_tracking.py --config config.yaml
   ```

2. **Generating Output:**

   - Once the model is trained and evaluated, run the output generation script to produce the tracking results.
   - Save the output CSV file in the `outputs/` folder following the naming pattern: `tracking_results{last_number}.csv`.

   Example command:

   ```bash
   python models/generate_tracking_output.py --output outputs/tracking_results1.csv
   ```

   **Output Format:**

   - **Columns:** `ID`, `Frame`, `Objects`, `Objective`
   - **Objects Column:** A list of dictionaries, each containing keys such as `tracked_id`, `x`, `y`, `w`, `h`, and `confidence`.

## Integration with the Integration Repository

The final submission for the competition will be assembled in the Integration Repository. Your output file (`outputs/tracking_results{last_number}.csv`) should conform to the following:

- **Format:** Must match the sample provided in the Integration Repository’s README.
- **Purpose:** It will be merged with the output from the face re-identification project to create the final submission file.

Refer to the [Integration Repository README](https://github.com/InceptionISA/Multi-Object-Tracking) for more details on how the outputs are combined.

## Experiment Tracking and Logging

- **Logging:** Document each experiment’s hyperparameters, evaluation scores, and any notes in a dedicated log file or through commit messages.
- **Versioning:** Use Git branches and commit messages to track progress and ensure reproducibility.

## Contributing

- Create your branch and follow the repository guidelines and branch naming conventions.
- Submit pull requests for new features or bug fixes, and ensure thorough testing before merging.
  `
