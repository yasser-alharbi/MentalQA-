# -*- coding: utf-8 -*-
"""MentalQA Using ALLaM-7B
"""
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    pipeline
)
import torch
from tqdm import tqdm
import gradio as gr

# ── Identity ─────────────────────────────────────────
SYSTEM_MSG = (
    "أنت مساعد ذكي للصحة النفسيةاسمه MentalQA\n"
    "لا تذكر اسمك أو منصة عملك إلا إذا سُئلت صراحةً عن هويتك.\n"
    "بالإضافة إلى ذلك:\n"
    "عندما يحييك أحد بتحية عربية:\n"
    "     - السلام عليكم => وعليكم السلام\n"
    "     - صباح الخير   => صباح النور\n"
    "     - مساء الخير   => مساء النور\n\n"
)

# The model prompt
def build_prompt_arabic(question, final_qt_list):
    qt_str = ", ".join(final_qt_list)

    prompt = (
        "أجب باللغة العربية استنادًا إلى القواعد التالية:\n"
        "1) هذه ليست استشارة طبية بديلة؛ قدّم إرشادات عامة وتمهيدية.\n"
        "2) لا تستخدم أسماء شخصية أو تدّعي ملكية.\n"
        "3) إذا كان السؤال خارج الصحة النفسية، قل: 'عذراً، ولكن هذا السؤال خارج نطاق قدرتي.'\n"
        "4) استرشد بقيم final_QT (A تشخيص، B علاج، C تشريح، D وبائيات، E نمط حياة، F خيارات مقدم الخدمة، G أخرى).\n"
        "5) إذا كانت حالة المريض حرجة، أبدِ تعاطفك أولاً ثم وجّه النصيحة.\n"
        "6) إذا احتاج المريض لتوجيه مباشر، قل: 'قد يفيد التواصل مع مختص نفسي أو مستشار موثوق.'\n\n"
        "مثال توضيحي للإجابة المفصّلة مع خطوات التفكير:\n"
        "سؤال: أشعر بإرهاقٍ مستمر ولا أستطيع التركيز، ماذا أفعل؟\n"
        "التفكير خطوة بخطوة:\n"
        "1) تحديد ما إذا كان الإرهاق جسدياً أم نفسياً.\n"
        "2) فحص نمط النوم والعادات اليومية.\n"
        "3) التفكير في عوامل الضغط والرعاية الذاتية.\n"
        "4) وضع خطة من نصائح تدريجية سهلة التطبيق.\n"
        "الإجابة النهائية:\n"
        "قد يرتبط الإرهاق بعدم انتظام النوم أو بضغوطٍ نفسية متراكمة. "
        "من المهم أولاً مراجعة نمط حياتك: اضبط مواعيد نوم ثابتة، وابتعد عن المنبّهات قبل النوم بساعتين. "
        "مارس المشي الخفيف أو تمارين الاسترخاء يوميّاً لتخفيف التوتر. "
        "إذا استمر الإرهاق أكثر من أسبوعين رغم هذه التغييرات، فكر في زيارة طبيب لفحص فيتامين د ووظائف الغدة الدرقية. "
        "دوّن مشاعرك في مفكرة يومية لتفريغ القلق وتشخيص الأسباب بدقة.\n"
        "—\n\n"
        f"final_QT: {qt_str}\n\n"
        "سؤال المستخدم:\n"
        f"{question}\n\n"
        "اكتب فقرة واحدة مفصّلة لا تقل عن ثلاث جمل مترابطة، بعد أن تفكّر خطوة بخطوة، \n"
        "الإجابة النهائية:\n"
    )
    return prompt

model_id = "ALLaM-AI/ALLaM-7B-Instruct-preview"
tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    low_cpu_mem_usage=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

classifier_model_path = "../classifiers/my_c1_model_97_f1_test_valid"
classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_model_path)
classifier_model = AutoModelForSequenceClassification.from_pretrained(classifier_model_path)
classifier_device = "cuda" if torch.cuda.is_available() else "cpu"
classifier_model.to(classifier_device)

clf_pipeline = pipeline(
    task="text-classification",
    model=classifier_model,
    tokenizer=classifier_tokenizer,
    device=0 if classifier_device == "cuda" else -1
)

label_mapping = {
    "LABEL_0": "A", "LABEL_1": "B", "LABEL_2": "C", 
    "LABEL_3": "D", "LABEL_4": "E", "LABEL_5": "F", "LABEL_6": "G"
}

def classify_question(question_text, threshold=0.5):
    preds = clf_pipeline(question_text)
    best_pred = max(preds, key=lambda x: x["score"])
    if best_pred["score"] < threshold:
        return None
    else:
        mapped_label = label_mapping.get(best_pred["label"], best_pred["label"])
        return mapped_label

def generate_answer_chat(prompt_text, model=model, tokenizer=tok, max_new_tokens=128):
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": prompt_text},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    
    prompt_len = input_ids.shape[1]
    gen_ids = model.generate(
        input_ids=input_ids,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.15,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )[0]
    
    answer_ids = gen_ids[prompt_len:]
    return tokenizer.decode(answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()

def get_mentalqa_answer(question_text, threshold=0.5):
    pred_label = classify_question(question_text, threshold=threshold)
    prompt_text = build_prompt_arabic(question=question_text, final_qt_list=[pred_label])
    answer = generate_answer_chat(prompt_text)
    return answer

# ---------- Gradio UI ----------
custom_css = """
#container {max-width: 640px; margin: 1.5rem auto;}
#question_box label, #answer_box label,
#question_box textarea, #answer_box textarea {
  direction: rtl; text-align: right;
}
"""

with gr.Blocks(css=custom_css, theme="soft") as demo:
    gr.Markdown(
        "<h2 style='text-align:center;'>🧠 MentalQA – مساعد الصحة النفسية</h2>"
        "<p style='text-align:center;'>اكتب سؤالك النفسي باللغة العربية وسيجيبك النموذج.</p>"
    )

    with gr.Group(elem_id="container"):
        question = gr.Textbox(lines=3, placeholder="اكتب سؤالك هنا...", label="سؤال:", elem_id="question_box")
        answer = gr.Textbox(lines=5, label="الإجابة:", elem_id="answer_box")
        submit_btn = gr.Button("إرسال")

        def on_submit(q):
            return get_mentalqa_answer(q, threshold=0.5)

        submit_btn.click(on_submit, inputs=question, outputs=answer)
        question.submit(on_submit, inputs=question, outputs=answer)

if __name__ == "__main__":
    demo.launch(share=True)
