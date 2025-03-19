# Human Sentiment Analysis and Trends Across Social Media

This project applies **Logistic Regression**, **BERT**, and **K-Means clustering** to analyze sentiment trends on social media. Using **transformers, PyTorch, pandas, numpy, scikit-learn, seaborn,** and more, we aim to extract insights into public opinion and social dynamics.

---

## Prerequisites

Before running the project, ensure you have the following:

### Files
Make sure the following files are in the same directory:
- `sentimentdataset.csv`
- `project.py` *(Preprocessing, sentiment analysis, model training)*
- `clustering.py` *(Topic clustering & visualization)*
- `dashboard.py` *(Model performance comparison)*
- `bert_sentiment_model` *(Our Pretrained BERT model - delete if you want to retrain the BERT model)*

### Python Packages
Install the required packages in your Conda environment using the commands below:
```bash
pip install pandas
pip install numpy
pip install nltk
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install transformers
pip install deep-translator
pip install --upgrade xformers
pip install wordcloud
```

### Installing PyTorch
Since PyTorch requires different installations based on your hardware, visit [PyTorch's official website](https://pytorch.org/) and select the appropriate installation command.

#### Example:
For Nvidia GPUs with 12GB VRAM or more with CUDA 12.6:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

For CPU-only installation:
```bash
pip install torch torchvision torchaudio
```

### System Requirements
1. Ensure you have the right PyTorch installed (GPU or CPU version)
2. Make sure you have Python 3.11 or later
3. If you have installed the GPU CUDA version, an **NVIDIA GPU** supporting **CUDA v12.6** is required.  
   - Download and install CUDA v12.6 from [NVIDIA's official website](https://developer.nvidia.com/cuda-downloads).

---

## Instructions to Run
You may safely ignore any red warnings displayed in the terminal while running the scripts.

### Step 1: Data Cleaning & Model Training
1. Run `project.py`:
   - This script cleans the dataset and trains both **Logistic Regression** and **BERT** on the sentiment data.
   - It may take several minutes to hours depending on your hardware.
   - If you want to **train a new BERT model**, delete the `bert_sentiment_model` directory before running the script.

### Step 2: Clustering Analysis
2. Run `clustering.py`:
    - Performs **K-Means clustering** on the dataset.
    - Generates **PCA visualization** and **word clouds** for topic analysis.

### Step 3: Model Performance Evaluation
3. Run `dashboard.py`:
    - Displays **accuracy metrics** of **Logistic Regression vs. BERT**.
    - Provides a **visual comparison of model performances**.

---

## Notes
- **Hardware Acceleration:** If using **CUDA for GPU acceleration**, ensure CUDA drivers are properly installed.
- **Performance Warning:** Training **BERT** without GPU acceleration **will be very slow**.