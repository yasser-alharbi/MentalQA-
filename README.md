# MentalQA

MentalQA is an intelligent mental health assistant system powered by LLMs (ALLaM-7B) and AraBERT-based classification models. This repository contains the source code for the question/answer classifiers, and the main Generation Pipeline with a Gradio web interface.

> **Note on Data:** All the datasets used for training and evaluation can be found in the [MentalQA GitHub Repository](https://github.com/hasanhuz/MentalQA).

## 🗂️ Project Structure

```text
MentalQA/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── src/
│   ├── app/
│   │   └── app.py            # Main application pipeline and Gradio UI (ALLaM-7B)
│   └── classifiers/
│       ├── classify_c1_mentalqa.py # Classifier 1: Question categorization
│       └── classify_c2_mentalqa.py # Classifier 2: Answer categorization
```

## 🚀 Phases of Execution

### Phase 1: Environment & Setup
1. Clone this repository or download the source code.
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have the dataset files downloaded from the [original repository](https://github.com/hasanhuz/MentalQA) if you intend to retrain the classifiers.

### Phase 2: Running Classifiers
- Navigate to the `src/classifiers/` folder.
- Execute the scripts to train the AraBERT models or run inference:
  ```bash
  python classify_c1_mentalqa.py
  python classify_c2_mentalqa.py
  ```
- *Note: Ensure your local datasets are present in the directories the scripts are referencing.*

### Phase 3: Launching the MentalQA Web UI
The generation pipeline integrates the `C1` classifier and the `ALLaM-7B` model to answer user questions effectively.
- Run the main application:
  ```bash
  cd src/app
  python app.py
  ```
- This will launch a localized Gradio web application for the MentalQA assistant.

## 🛠️ Built With
- **[Hugging Face Transformers](https://huggingface.co/)** - For ALLaM-7B and AraBERT models.
- **[Gradio](https://gradio.app/)** - For the User Interface.
- **[PyTorch](https://pytorch.org/)** - Deep Learning framework.
