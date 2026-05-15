# MentalQA

MentalQA is an intelligent mental health assistant system powered by LLMs (ALLaM-7B) and AraBERT-based classification models. This repository contains the source code for the question/answer classifiers, and the main Generation Pipeline with a Gradio web interface.

> **Note on Data:** All the datasets used for training and evaluation can be found in the [MentalQA GitHub Repository](https://github.com/hasanhuz/MentalQA).

## ✨ System Example

| Type | Content |
| :--- | :--- |
| ❓ **Question** | هل يعتبر الخوف من عدم الإنجاب مستقبلاً حالة عادية خاصةً لما أكون متعلقة بأطفال كثيراً وأنا على وجه جواز أنا خايفة جداً |
| 👨‍⚕️ **True (Doctor) Answer** | سد تى نعم ان هناك العديد من القلق النفسى المرتبط بحالات العقم و حب الانجاب لدرج ان لم يتحقق هدا تكون له مضار سلوكيه و نفسيه ودعينى اشير الى مثال اخر نتعامل معه نحن اخصائ النسائيه و العقم و حالات الاحباط التى تتم ايضا بمرحله النفاس و كيفيه رفض الام لطفلها و هناك العديد و عليه انصح و قبل الزواج ان نتعا مل معهز دابراهيم هنداوى مستشار النسائيه والعقم والغدد وطفال الانابيب-مدينه الحسين الطبيه-الاردن XXXXXX@yahoo.com |
| 🤖 **MentalQA Answer** | أهلاً بك، يبدو أنك تمرين بمشاعر قوية تجاه مسألة الإنجاب. يمكن أن يكون الخوف وعدم اليقين حول المستقبل أمراً طبيعياً، خاصة عندما تكونين مرتبطة بفكرة الأمومة. يُنصح بالتحدث مع شريكك بصدق حول مخاوفكما، فهذا يعزز التفاهم والدعم المتبادل. بالإضافة إلى التحدث مع مختصة نفسية موثوقة لتشخيص حالتك وتقديم الدعم المناسب. تذكري دائماً أن البحث عن المساعدة المهنية ليس علامة ضعف بل دليل على الوعي والرغبة في تحسين الحالة النفسية والعاطفية. |

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
