import json
import os
import random
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments
from transformers import TrainerCallback
from datasets import load_dataset
from deep_translator import GoogleTranslator
from langdetect import detect

use_fine_tuned = True  # Set to True if you want to use the fine-tuned model

if use_fine_tuned and os.path.exists("./fine_tuned_model"):
    model = T5ForConditionalGeneration.from_pretrained("./fine_tuned_model")
    tokenizer = T5Tokenizer.from_pretrained("./fine_tuned_model")
else:
    model = T5ForConditionalGeneration.from_pretrained("valhalla/t5-base-qg-hl")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

model.config.pad_token_id = tokenizer.pad_token_id


class PrintLearningRateCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "learning_rate" in logs:
            print(f"🔁 Learning rate: {logs['learning_rate']:.8f}")



def freeze_encoder_layers(model, num_layers_to_freeze=6):
    for i, layer in enumerate(model.encoder.block):
        if i < num_layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

freeze_encoder_layers(model, num_layers_to_freeze=6)

def read_text_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().strip()

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def translate_to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        print(f"Translation to English failed: {e}")
        return text

def translate_to_arabic(text):
    try:
        return GoogleTranslator(source='en', target='ar').translate(text)
    except Exception as e:
        print(f"Translation to Arabic failed: {e}")
        return text


def generate_questions(context, lang="en"):
    if lang == "ar":
        prefix = "أنشئ أسئلة: "
    else:
        prefix = "generate questions: "

    input_text = prefix + context
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        input_ids,
        max_length=128,
        num_beams=5,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.8,
        num_return_sequences=5
    )
    decoded_outputs = [
        tokenizer.decode(out, skip_special_tokens=True).strip()
        for out in outputs
        if tokenizer.decode(out, skip_special_tokens=True).strip()
    ]
    return [q for q in decoded_outputs if len(q.split()) > 4]

def generate_fill_in_the_blank_question(context):
    input_text = f"Generate a fill-in-the-blank question: {context}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        input_ids,
        max_length=128,
        num_beams=4,
        do_sample=True,
        top_p=0.92,
        top_k=50,
        temperature=0.9,
        num_return_sequences=1
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def generate_true_false_question(context):
    input_text = f"Generate a true/false question: {context}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        input_ids,
        max_length=128,
        num_beams=4,
        do_sample=True,
        top_p=0.92,
        top_k=50,
        temperature=0.9,
        num_return_sequences=1
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def classify_question(question):
    q_lower = question.lower()
    if "؟" in question or "?" in question:
        if "صواب" in question or "خطأ" in question or "true" in q_lower or "false" in q_lower:
            return "True/False"
        elif "___" in question or "....." in question:
            return "Fill in the Blank"
        elif any(opt in q_lower for opt in ["اختر", "من بين", "أي من التالي"]):
            return "Multiple Choice"
        else:
            return "Essay"
    return "Essay"

def generate_answer(context, question):
    input_text = f"question: {question} context: {context}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        input_ids,
        max_length=64,
        num_beams=4,
        early_stopping=True,
        temperature=0.8,
        top_p=0.95
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    if not answer or len(answer.split()) < 2:
        return "إجابة مفتوحة من الطالب."
    return answer

def generate_distractors_with_model(context, question, correct_answer, lang="en", num_distractors=3):
    prompt = (
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Correct Answer: {correct_answer}\n"
        f"Generate {num_distractors} plausible but incorrect answers."
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        input_ids,
        max_length=128,
        num_return_sequences=num_distractors,
        num_beams=5,
        early_stopping=True
    )
    distractors = [tokenizer.decode(out, skip_special_tokens=True).strip() for out in outputs]
    distractors = [d for d in distractors if d.strip() and d.lower() != correct_answer.lower()]
    if lang == "ar":
        distractors = [translate_to_arabic(d) for d in distractors]
    return distractors

def generate_mcq_options(original_context, question_text, lang="ar"):
    correct = generate_answer(original_context, question_text)
    distractors = generate_distractors_with_model(original_context, question_text, correct, lang=lang)
    options = [correct] + distractors
    options = list(dict.fromkeys(options))  # Remove duplicates
    random.shuffle(options)
    correct_index = options.index(correct)
    return {
        "options": options,
        "correctAnswer": correct_index
    }

def save_questions(data, filename="questions.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def prepare_data_for_training(input_file, output_file):
    text = read_text_from_file(input_file)
    lang = detect_language(text)
    original_context = text

    questions = generate_questions(original_context)

    data = []
    for q in questions:
        question_type = classify_question(q)
        question_entry = {
            "question": q,
            "type": question_type
        }

        if question_type == "Multiple Choice":
            question_entry["answer"] = generate_mcq_options(original_context, q, lang=lang)
        elif question_type == "True/False":
            tf_answer = generate_answer(original_context, q)
            tf_answer = "True" if "نعم" in tf_answer or "صحيح" in tf_answer else "False"
            question_entry["answer"] = tf_answer
        elif question_type == "Fill in the Blank":
            question_entry["answer"] = generate_fill_in_the_blank_question(original_context)
        else:
            answer = generate_answer(original_context, q)
            question_entry["answer"] = answer

        if question_entry["question"]:  # استبعاد الأسئلة الفارغة
            data.append(question_entry)

    save_questions(data, output_file)

def convert_qa_json_to_jsonl(json_file="train.json", input_text_file="train_input.txt", output_file="train_data.jsonl"):
    if not os.path.exists(json_file) or not os.path.exists(input_text_file):
        print(f"❌ File {json_file} or {input_text_file} not found.")
        return None

    with open(json_file, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    with open(input_text_file, "r", encoding="utf-8") as f:
        context = f.read().strip()
    # context_en = translate_to_english(context)

    data = []
    for qa in qa_data:
        q = qa["question"]
        a = qa["answer"] if isinstance(qa["answer"], str) else qa["answer"].get("text", "")

        if a.lower() not in context.lower():
            highlighted = context + f" <hl> {a} <hl>"
        else:
            highlighted = context.replace(a, f"<hl> {a} <hl>", 1)

        data.append({
            "input": f"generate question: {highlighted}",
            "target": q
        })

    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Converted to JSONL: {output_file} ({len(data)} items)")
    return output_file

def fine_tune_model(jsonl_file):
    dataset = load_dataset("json", data_files={"train": jsonl_file}, split="train")

    def preprocess(example):
        inputs = tokenizer(example["input"], max_length=512, truncation=True, padding="max_length")
        targets = tokenizer(example["target"], max_length=64, truncation=True, padding="max_length")
        inputs["labels"] = targets["input_ids"]
        return inputs

    dataset = dataset.map(preprocess)

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        learning_rate=3e-5,
        logging_dir="./logs",
        save_steps=500,
        logging_steps=100,
        save_total_limit=2,
        dataloader_pin_memory=False
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[PrintLearningRateCallback()]
    )

    trainer.train()

def save_trained_model():
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")

def test_model():
    try:
        with open("input.txt", "r", encoding="utf-8") as f:
            test_context = f.read().strip()
    except FileNotFoundError:
        print("⚠️ ملف input.txt غير موجود.")
        return

    if not test_context:
        print("⚠️ ملف input.txt فارغ.")
        return

    context_lang = detect(test_context)

    print("\n🧪 اختبار النموذج بعد التدريب:\n")
    questions = generate_questions(test_context, lang=context_lang)
    output = []

    count = 0
    for q in questions:
        if detect(q) != context_lang:
            continue  # تجاهل الأسئلة التي ليست بنفس لغة النص

        q_type = classify_question(q, lang=context_lang)

        item = {"question": q, "type": q_type}

        if q_type == "Multiple Choice":
            mcq = generate_mcq_options(test_context, q, lang=context_lang)
            item["answer"] = {
                "options": mcq["options"],
                "correctAnswer": mcq["correctAnswer"]
            }

        elif q_type == "Essay":
            item["answer"] = "Open-ended response from the student about the topic."

        else:
            ans = generate_answer(test_context, q, lang=context_lang)
            item["answer"] = ans

        output.append(item)
        count += 1
        if count >= 5:
            break

    with open("questions_output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    print("✅ تم حفظ 5 أسئلة باللغة المطابقة في questions_output.json")

if __name__ == "__main__":
    context_folder = "training_context"
    question_folder = "training_question"
    all_jsonl_data = []
    test_model()

    for i in range(1, 21):  
        context_path = os.path.join(context_folder, f"{i}.txt")
        question_path = os.path.join(question_folder, f"{i}.json")

        if not os.path.exists(context_path) or not os.path.exists(question_path):
            print(f"❌ الملف {i} مفقود: {context_path} أو {question_path}")
            continue

        with open(context_path, "r", encoding="utf-8") as f:
            context = f.read().strip()

        with open(question_path, "r", encoding="utf-8") as f:
            qa_data = json.load(f)

        for qa in qa_data:
            q = qa["question"]
            if isinstance(qa["answer"], str):
                a = qa["answer"]
            else:
                a = qa["answer"].get("text", "")

            if a.lower() not in context.lower():
                highlighted = context + f" <hl> {a} <hl>"
            else:
                highlighted = context.replace(a, f"<hl> {a} <hl>", 1)

            all_jsonl_data.append({
                "input": f"generate question: {highlighted}",
                "target": q
            })

    jsonl_file = "train_data_all.jsonl"
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for item in all_jsonl_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ تم تحويل جميع الملفات إلى {jsonl_file} بإجمالي {len(all_jsonl_data)} سؤالاً.")


    if all_jsonl_data:
        fine_tune_model(jsonl_file)
        save_trained_model()
        print("\n✅ تم تدريب النموذج وحفظه بنجاح.")
        test_model()  # ✅ اختبار النموذج المدرب
    else:
        print("\n⚠️ لم يتم العثور على بيانات كافية للتدريب.")

