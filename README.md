# 📰 Fake News Detection Using BERT

An NLP-based Machine Learning project that classifies news headlines as **Fake News or Real News** using a fine-tuned BERT (Bidirectional Encoder Representations from Transformers) model.

## 🚀 Project Overview

Fake news spreads misleading or fabricated information through digital platforms. This project aims to identify potentially fake news headlines using Natural Language Processing (NLP) and a pretrained BERT model.

The model analyzes news headlines and predicts whether the given headline belongs to:

- **Fake (0):** Potentially fabricated or misleading news.
- **Real (1):** Real news.

The trained model also provides prediction probabilities for both classes.

## 🎯 Objectives

- Understand and preprocess news datasets.
- Combine fake and real news datasets.
- Apply NLP preprocessing techniques.
- Fine-tune a pretrained BERT model.
- Evaluate classification performance using multiple metrics.
- Predict whether a given news headline is fake or real.

## 📂 Dataset

The project uses two CSV datasets:

| Dataset | Records | Label |
|---|---:|---:|
| Fake.csv | 23,481 | 0 (Fake) |
| True.csv | 21,417 | 1 (Real) |
| **Combined Dataset** | **44,898** | Binary |

### Dataset Features

- `title` – News headline
- `text` – Full news article
- `subject` – News category
- `date` – Publication date
- `label` – Target variable

The two datasets are combined, shuffled, and saved as `Combined.csv`.

**Important:** The model in this implementation uses the `title` column for training and prediction, rather than the full article text.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- TensorFlow
- Hugging Face Transformers
- BERT (bert-base-uncased)
- NLTK
- Scikit-learn
- Matplotlib
- Google Colab

## ⚙️ Project Workflow

1. **Data Loading**
   - Load Fake.csv and True.csv.
   - Examine dataset structure and records.

2. **Data Integration**
   - Assign binary labels to both datasets.
   - Combine the datasets.
   - Shuffle the records.
   - Generate Combined.csv.

3. **Text Preprocessing**
   - Remove special characters and numbers.
   - Convert text to lowercase.
   - Apply Porter Stemming using NLTK.

4. **Dataset Splitting**

   The dataset is divided into three subsets:

   | Dataset | Percentage |
   |---|---:|
   | Training | 64% |
   | Validation | 16% |
   | Testing | 20% |

5. **BERT Tokenization**
   - Load the pretrained `bert-base-uncased` tokenizer.
   - Tokenize preprocessed news headlines.
   - Apply padding and truncation.
   - Set maximum sequence length to 42.
   - Generate input IDs, token type IDs, and attention masks.

6. **Model Training**
   - Load pretrained BERT for sequence classification.
   - Configure two output classes.
   - Train using TensorFlow.

7. **Model Evaluation**
   - Evaluate test accuracy and loss.
   - Generate classification report.
   - Plot confusion matrix.
   - Calculate ROC-AUC score.

8. **Prediction**
   - Accept a news headline.
   - Preprocess and tokenize the input.
   - Predict Fake or Real.
   - Display class probabilities.

## 🧠 Model Architecture

**Pretrained Model:** BERT Base Uncased

**Task:** Binary Sequence Classification

| Hyperparameter | Value |
|---|---|
| Model | bert-base-uncased |
| Number of classes | 2 |
| Optimizer | Adam |
| Learning rate | 2e-5 |
| Loss function | Sparse Categorical Crossentropy |
| Batch size | 64 |
| Epochs | 3 |
| Maximum sequence length | 42 |
| Framework | TensorFlow |

BERT uses bidirectional contextual representations to understand the relationships between words in a headline.

## 📊 Model Performance

The model was evaluated on 8,979 test samples.

### Evaluation Metrics

| Metric | Score |
|---|---:|
| Test Accuracy | **97.45%** |
| ROC-AUC | **0.9969** |
| Fake Precision | 0.97 |
| Fake Recall | 0.98 |
| Fake F1-Score | 0.98 |
| Real Precision | 0.97 |
| Real Recall | 0.97 |
| Real F1-Score | 0.97 |

### Confusion Matrix

| Actual / Predicted | Fake | Real |
|---|---:|---:|
| Fake | 4,548 | 111 |
| Real | 118 | 4,202 |

### Training Performance

| Epoch | Training Accuracy | Validation Accuracy |
|---|---:|---:|
| 1 | 93.73% | 95.96% |
| 2 | 97.34% | 97.02% |
| 3 | 98.49% | 97.22% |

The model achieved 97.45% test accuracy after three training epochs.

## 📁 Project Structure

```text
Fake-News-Detection-BERT/
│
├── Fake_News_BERT.ipynb
│
├── Fake.csv
├── True.csv
│
├── Combined.csv
│
├── bertv3_model/
│   ├── config.json
│   ├── model weights
│   └── other model files
│
└── README.md
```

*Note: `Combined.csv` and `bertv3_model/` are generated during execution. The original notebook saves the model under `/content/bertv3_model` in Google Colab.*

## 💻 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Fake-News-Detection-BERT.git

cd Fake-News-Detection-BERT
```

### 2. Install Dependencies

```bash
pip install pandas numpy nltk scikit-learn matplotlib
pip install tensorflow==2.19.1
pip install transformers==4.41.2
```

### 3. Download the Datasets

Place the following files in your working directory:

```text
Fake.csv
True.csv
```

### 4. Run the Notebook

Open `Fake_News_BERT.ipynb` using Jupyter Notebook or Google Colab.

Execute the notebook cells sequentially to:

- Load and preprocess datasets.
- Train the BERT model.
- Evaluate model performance.
- Generate predictions.

**Note:** The notebook was executed in Google Colab with GPU support. Training time and hardware requirements may vary.

## 🔍 Sample Prediction

**Input Headline:**

```text
Sheriff David Clarke Becomes An Internet Joke
For Threatening To Poke People In The Eye
```

**Model Output:**

```text
Prediction: FAKE NEWS

Probability of being fake: 99.95%
Probability of being real: 0.05%
```

## 📈 Future Improvements

- Extend classification from headlines to full news articles.
- Experiment with different BERT variants such as DistilBERT and RoBERTa.
- Build a web application using FastAPI or Flask.
- Create an interactive interface for real-time predictions.
- Deploy the trained model using Docker and cloud services.
- Evaluate model generalization on newer and independent news datasets.

## ⚠️ Limitations

- The model is trained on a specific historical dataset.
- Predictions are based on headlines, not complete articles.
- High test accuracy does not guarantee reliable performance on current or unseen news sources.
- Model confidence scores should not be interpreted as proof that a news story is true or false.

This project is an experimental text classification system, not a substitute for independent fact-checking.

---

⭐ If you find this project useful, consider giving the repository a star!

