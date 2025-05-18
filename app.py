from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# === 1. Load model & tokenizer yang ringan ===
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load model untuk classification 2 kelas
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2)

# === 2. Load dataset kecil (gunakan SST2 subset) ===
dataset = load_dataset("glue", "sst2")
# cuma 200 data untuk training biar ringan
train_dataset = dataset["train"].select(range(200))
eval_dataset = dataset["validation"].select(
    range(50))  # 50 data untuk evaluasi


def preprocess(examples):
    return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=64)


train_dataset = train_dataset.map(preprocess, batched=True)
train_dataset = train_dataset.rename_column("label", "labels")
train_dataset.set_format(type='torch', columns=[
                         'input_ids', 'attention_mask', 'labels'])

eval_dataset = eval_dataset.map(preprocess, batched=True)
eval_dataset = eval_dataset.rename_column("label", "labels")
eval_dataset.set_format(type='torch', columns=[
                        'input_ids', 'attention_mask', 'labels'])

# === 3. LoRA Fine-tuning ===

# Untuk DistilBERT, kita definisikan target modules secara manual
# Karena "distilbert" mungkin tidak ada di mapping default
target_modules = ["q_lin", "k_lin", "v_lin", "out_lin", "ffn.lin1", "ffn.lin2"]
print(f"Target modules for LoRA: {target_modules}")

lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=target_modules,  # Target modules untuk DistilBERT
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)

model_lora = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="./results_lora",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    logging_steps=10,
    save_total_limit=1,
    save_strategy="epoch",
    fp16=False,  # non fp16 untuk safety di GPU low-end
    load_best_model_at_end=True
)

# Fungsi untuk menghitung metrik evaluasi


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='binary')
    return {"accuracy": acc, "f1": f1}


trainer = Trainer(
    model=model_lora,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

print("Starting LoRA fine-tuning...")
trainer.train()

# Evaluasi model LoRA
print("\n=== Evaluasi Model LoRA ===")
eval_results = trainer.evaluate()
print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"F1 Score: {eval_results['eval_f1']:.4f}")

# === 4. Pendekatan alternatif: Multiple LoRA Adapters ===
print("\n=== Implementing Multiple LoRA Adapters as alternative to AdapterFusion ===")

# Kita akan mengimpementasikan beberapa adapters LoRA untuk mensimulasikan fungsionalitas adapter fusion
# 1. Buat beberapa adapter LoRA dengan konfigurasi berbeda (simulasi domain-specific adapters)
domains = ["domain_news", "domain_social"]
domain_models = {}
domain_results = {}

# Buat dan latih adapter untuk setiap domain
for domain in domains:
    print(f"\nTraining LoRA adapter for domain: {domain}")
    # Buat konfigurasi LoRA dengan parameter berbeda untuk tiap domain
    domain_config = LoraConfig(
        r=4 if domain == "domain_news" else 8,  # Gunakan r berbeda per domain
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.1 if domain == "domain_news" else 0.2,  # Dropout berbeda
        bias="none",
        task_type="SEQ_CLS"
    )

    # Buat model baru untuk setiap domain
    domain_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2)
    domain_model_lora = get_peft_model(domain_model, domain_config)

    # Train model untuk domain ini
    domain_trainer = Trainer(
        model=domain_model_lora,
        args=TrainingArguments(
            output_dir=f"./results_{domain}",
            evaluation_strategy="epoch",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=1,
            logging_steps=10,
            save_total_limit=1,
            save_strategy="epoch",
            fp16=False,
            load_best_model_at_end=True
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    domain_trainer.train()

    # Evaluasi model domain
    print(f"\n=== Evaluasi Model Domain {domain} ===")
    domain_eval_results = domain_trainer.evaluate()
    print(f"Accuracy: {domain_eval_results['eval_accuracy']:.4f}")
    print(f"F1 Score: {domain_eval_results['eval_f1']:.4f}")

    # Simpan model ke dictionary
    domain_models[domain] = domain_model_lora
    domain_results[domain] = domain_eval_results

    # Simpan adapter ke disk
    domain_model_lora.save_pretrained(f"./adapter_{domain}")
    print(f"Saved adapter for {domain}")

# === 5. Simulasi "Adapter Ensemble" sebagai alternatif AdapterFusion ===
print("\n=== Implementing LoRA Adapter Ensemble as alternative to AdapterFusion ===\n")

# Buat model baru untuk simulasi "ensemble"
ensemble_model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2)

# Simulasi adapter ensemble dengan membuat LoRA baru yang mengkombinasikan insights dari adapters sebelumnya
ensemble_config = LoraConfig(
    r=6,  # Nilai rata-rata dari adapters sebelumnya
    lora_alpha=16,
    target_modules=target_modules,
    lora_dropout=0.15,  # Nilai rata-rata dari adapters sebelumnya
    bias="none",
    task_type="SEQ_CLS"
)

ensemble_model_lora = get_peft_model(ensemble_model, ensemble_config)

# Train dengan lebih sedikit epoch (kita bisa bayangkan model ini belajar dari insights adapters sebelumnya)
ensemble_trainer = Trainer(
    model=ensemble_model_lora,
    args=TrainingArguments(
        output_dir="./results_ensemble",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        logging_steps=10,
        save_total_limit=1,
        save_strategy="epoch",
        fp16=False,
        load_best_model_at_end=True
    ),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

print("Training adapter ensemble...")
ensemble_trainer.train()

# Evaluasi model ensemble
print("\n=== Evaluasi Model Ensemble ===")
ensemble_eval_results = ensemble_trainer.evaluate()
print(f"Accuracy: {ensemble_eval_results['eval_accuracy']:.4f}")
print(f"F1 Score: {ensemble_eval_results['eval_f1']:.4f}")

# === 6. Bandingkan hasil evaluasi semua model ===
print("\n=== Perbandingan Hasil Evaluasi ===")
print(
    f"Model LoRA: Accuracy={eval_results['eval_accuracy']:.4f}, F1={eval_results['eval_f1']:.4f}")
for domain in domains:
    print(
        f"Model Domain {domain}: Accuracy={domain_results[domain]['eval_accuracy']:.4f}, F1={domain_results[domain]['eval_f1']:.4f}")
print(
    f"Model Ensemble: Accuracy={ensemble_eval_results['eval_accuracy']:.4f}, F1={ensemble_eval_results['eval_f1']:.4f}")

# === 7. Melakukan inferensi dengan model yang sudah di-fine-tune ===
print("\nPerforming inference with fine-tuned models:")

# Contoh kalimat untuk inferensi
test_sentences = [
    "this movie was fantastic and very enjoyable",
    "the plot was terrible and the acting was worse"
]

# 1. Inferensi dengan model LoRA pertama
print("\n--- Primary LoRA Model Inference ---")
for sentence in test_sentences:
    inputs = tokenizer(sentence, return_tensors="pt",
                       truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model_lora(**inputs)
    prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label = torch.argmax(prediction, dim=1).item()
    confidence = prediction[0][label].item()
    sentiment = "Positive" if label == 1 else "Negative"
    print(f"Sentence: '{sentence}'")
    print(f"Sentiment: {sentiment} (confidence: {confidence:.4f})")

# 2. Inferensi dengan model Domain-Specific LoRA
for domain in domains:
    print(f"\n--- Domain '{domain}' LoRA Model Inference ---")
    domain_model = domain_models[domain]
    for sentence in test_sentences:
        inputs = tokenizer(sentence, return_tensors="pt",
                           truncation=True, padding=True, max_length=64)
        with torch.no_grad():
            outputs = domain_model(**inputs)
        prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
        label = torch.argmax(prediction, dim=1).item()
        confidence = prediction[0][label].item()
        sentiment = "Positive" if label == 1 else "Negative"
        print(f"Sentence: '{sentence}'")
        print(f"Sentiment: {sentiment} (confidence: {confidence:.4f})")

# 3. Inferensi dengan model Ensemble LoRA (pengganti Adapter Fusion)
print("\n--- Ensemble LoRA Model Inference ---")
for sentence in test_sentences:
    inputs = tokenizer(sentence, return_tensors="pt",
                       truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = ensemble_model_lora(**inputs)
    prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label = torch.argmax(prediction, dim=1).item()
    confidence = prediction[0][label].item()
    sentiment = "Positive" if label == 1 else "Negative"
    print(f"Sentence: '{sentence}'")
    print(f"Sentiment: {sentiment} (confidence: {confidence:.4f})")
