"""
Modulo per il fine-tuning di LLMs su dataset CoSQL per la traduzione
di linguaggio naturale a SQL in contesti conversazionali multi-turn.
"""

import os
import json
import re
import gc
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
import torch
import random


def load_database_schema(database_id, database_dir):
    """
    Carica e processa lo schema di un database dal file SQL corrispondente.
    
    Args:
        database_id (str): Identificativo del database
        database_dir (str): Directory contenente i database
        
    Returns:
        str: Schema formattato con statements CREATE TABLE o messaggio di errore
    """
    schema_file = os.path.join(database_dir, database_id, "schema.sql")
    
    if not os.path.exists(schema_file):
        return f"Database: {database_id}, NO SCHEMA AVAILABLE"
    
    try:
        with open(schema_file, 'r', encoding='utf-8', errors='ignore') as f:
            sql_content = f.read()
        
        # Estrai tutti i CREATE TABLE statements
        create_tables = []
        table_pattern = r'(CREATE\s+TABLE\s+[^;]+;)'
        matches = re.findall(table_pattern, sql_content, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            # Normalizza la formattazione rimuovendo spazi extra
            clean_table = re.sub(r'\s+', ' ', match.strip())
            create_tables.append(clean_table)
        
        final_schema = "\n\n".join(create_tables) if create_tables else f"Database: {database_id}, NO TABLES FOUND"
        return final_schema
        
    except Exception as e:
        print(f"Errore durante la lettura dello schema {database_id}: {e}")
        return f"Database: {database_id}, ERROR READING SCHEMA"


def process_dialog(dialog, max_turns=5):
    """
    Processa un dialogo multi-turn convertendolo in examples di traduzione.
    
    Args:
        dialog (dict): Dialogo contenente database_id e interaction
        max_turns (int): Numero massimo di turn da processare
        
    Returns:
        list: Lista di examples di traduzione formattati per il training
    """
    database_id = dialog.get('database_id', '')
    interactions = dialog.get('interaction', [])
    
    if not database_id or not interactions:
        return []
    
    schema = load_database_schema(database_id, database_dir)
    system_header = f"""### SYSTEM INSTRUCTIONS ###
        You are a helpful SQL assistant that generates precise SQL queries based on database schemas and natural language questions. Generate only the SQL query without explanations.

        ### SCHEMA ###
        {schema}

        ### DIALOGUE ###
    """
    
    examples = []
    dialogue_lines = []
    
    for i, turn in enumerate(interactions[:max_turns]):
        utterance = turn.get('utterance', '').strip()
        query = turn.get('query', '').strip()
        
        if not utterance or not query:
            continue
        
        if i == 0:
            prompt = f"{system_header}\nQ: {utterance}"
        else:
            prompt = f"{system_header}\n" + "\n".join(dialogue_lines) + f"\nQ: {utterance}"
        
        examples.append({
            "prompt": prompt,
            "response": query
        })
        
        dialogue_lines.append(f"Q: {utterance}")
        dialogue_lines.append(f"A: {query}")
    
    return examples


def process_training_data(train_data, limit):
    """
    Processa l'intero dataset di training convertendo i dialoghi in examples.
    
    Args:
        train_data (list): Lista dei dialoghi di training
        limit (int): Limite al numero di dialoghi da processare (None per tutti)
        
    Returns:
        list: Lista di examples di training pronti per la tokenizzazione
    """
    print("Processing training data...")
    train_examples = []
    if limit is None:
        for dialog in train_data:
            train_examples.extend(process_dialog(dialog))
    else:
        for dialog in train_data[:limit]:
            train_examples.extend(process_dialog(dialog))
    
    random.shuffle(train_examples)
    return train_examples


def process_dev_data(dev_data, limit):
    """
    Processa il dataset di validazione convertendo i dialoghi in examples.
    
    Args:
        dev_data (list): Lista dei dialoghi di validazione
        limit (int): Limite al numero di dialoghi da processare (None per tutti)
        
    Returns:
        list: Lista di examples di validazione pronti per la tokenizzazione
    """
    print("Processing validation data...")
    dev_examples = []
    if limit is None:
        for dialog in dev_data:
            dev_examples.extend(process_dialog(dialog))
    else:
        for dialog in dev_data[:limit]:
            dev_examples.extend(process_dialog(dialog))
    return dev_examples


def load_model(model_id):
    """
    Carica il modello base e il tokenizer dal repository Hugging Face.
    
    Args:
        model_id (str): Identificativo del modello su Hugging Face
        
    Returns:
        tuple: (modello, tokenizer) caricati e configurati
    """
    print(f"Caricamento modello: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True
    ).cpu()
        
    print(f"Modello caricato. Dimensione vocabolario: {tokenizer.vocab_size}")
    return model, tokenizer


def configure_lora(model, params=None):
    """
    Configura e applica LoRA (Low-Rank Adaptation) al modello.
    
    Args:
        model: Modello base da adattare
        params (dict): Parametri di configurazione LoRA
        
    Returns:
        Modello con LoRA applicato o None se parametri mancanti
    """
    print("Configurazione LoRA in corso...")
    
    if params is None:
        print("Errore: Nessun parametro fornito per la configurazione LoRA")
        return None
    
    lora_config = LoraConfig(**params)
    model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA applicato. Parametri addestrabili: {trainable_params:,}")
    
    return model


def tokenize_function(examples):
    """
    Tokenizza gli examples di training applicando il masking sui prompt.
    
    Args:
        examples (dict): Batch di examples contenenti 'prompt' e 'response'
        
    Returns:
        dict: Examples tokenizzati con input_ids, attention_mask e labels
    """
    input_texts = []
    target_texts = []
    
    for prompt, response in zip(examples['prompt'], examples['response']):
        full_text = f"{prompt}\nSQL: {response}"
        prompt_text = f"{prompt}\nSQL:"
        
        input_texts.append(full_text)
        target_texts.append(prompt_text)
    
    # Tokenizzazione del testo completo
    model_inputs = tokenizer(
        input_texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None
    )
    
    # Tokenizzazione dei soli prompt per identificare dove inizia la risposta
    prompt_inputs = tokenizer(
        target_texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None
    )
    
    # Creazione delle labels con masking sui prompt
    labels = []
    for i, (input_ids, prompt_ids) in enumerate(zip(model_inputs['input_ids'], prompt_inputs['input_ids'])):
        label = input_ids.copy()
        
        # Masking dei token del prompt (impostati a -100 per essere ignorati dalla loss)
        prompt_length = len(prompt_ids)
        for j in range(min(prompt_length, len(label))):
            label[j] = -100
            
        labels.append(label)
    
    return {
        'input_ids': model_inputs['input_ids'],
        'attention_mask': model_inputs['attention_mask'],
        'labels': labels
    }


def create_dataset(examples, batch_size):
    """
    Crea un dataset tokenizzato da una lista di examples.
    
    Args:
        examples (list): Lista di examples contenenti prompt e response
        batch_size (int): Dimensione del batch per la tokenizzazione
        
    Returns:
        Dataset: Dataset tokenizzato pronto per il training
    """
    return Dataset.from_list(examples).map(
        tokenize_function,
        batched=True,
        batch_size=batch_size,
        remove_columns=['prompt', 'response']
    )


def setup_training(training_params, model, tokenizer, train_dataset, eval_dataset):
    """
    Configura il trainer per l'addestramento del modello.
    
    Args:
        training_params (dict): Parametri di training
        model: Modello da addestrare
        tokenizer: Tokenizer del modello
        train_dataset: Dataset di training
        eval_dataset: Dataset di validazione
        
    Returns:
        Trainer: Trainer configurato e pronto per l'addestramento
    """
    print("Configurazione training in corso...")
    
    training_args = TrainingArguments(**training_params)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator
    )
    
    return trainer


def save_model(model, tokenizer, output_dir):
    """
    Salva il modello addestrato e il tokenizer nella directory specificata.
    
    Args:
        model: Modello da salvare
        tokenizer: Tokenizer da salvare
        output_dir (str): Directory di destinazione
    """
    print(f"Salvataggio modello in: {output_dir}")
    try:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("Modello salvato con successo")
    except Exception as e:
        print(f"Errore durante il salvataggio: {e}")


def load_finetuned_model(model_dir, base_model_id):
    """
    Carica un modello fine-tuned con adapter LoRA.
    
    Args:
        model_dir (str): Directory contenente il modello salvato
        base_model_id (str): ID del modello base utilizzato
        
    Returns:
        tuple: (modello, tokenizer) caricati o (None, None) in caso di errore
    """
    print("Caricamento modello fine-tuned...")
    
    # Liberazione memoria
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    try:
        # Caricamento tokenizer
        print("Caricamento tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Caricamento modello base
        print("Caricamento modello base...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float32,
            device_map=None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).cpu()
        
        # Applicazione adapter LoRA
        print("Applicazione adapter LoRA...")
        model = PeftModel.from_pretrained(
            base_model,
            model_dir,
            torch_dtype=torch.float32,
        ).cpu()
        
        print(f"Modello LoRA caricato da: {model_dir}")
        print("Dispositivo: CPU")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Errore durante il caricamento del modello: {e}")
        return None, None


def generate_sql(prompt):
    """
    Genera una query SQL dal prompt fornito utilizzando il modello caricato.
    
    Args:
        prompt (str): Prompt contenente schema e domanda
        
    Returns:
        str: Query SQL generata dal modello
    """
    text = f"{prompt}\nSQL:"
    
    # Tokenizzazione
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=max_length,
        padding=True
    )
    
    # Spostamento su device del modello
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generazione
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )
    
    # Decodifica della parte generata
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # Pulizia output (estrazione fino al primo punto e virgola)
    if ';' in generated_text:
        generated_text = generated_text.split(';')[0] + ';'
    
    return generated_text


# Configurazione del modello corrente
nome = "model-000"
model_id = "deepseek-ai/deepseek-coder-1.3b-instruct"

output_dir = f'./{nome}'
final_dir = f'./{nome}_final'

# Configurazione LoRA
params = {
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "lora_dropout": 0.1,
    "bias": 'none',
    "task_type": TaskType.CAUSAL_LM
}

# Parametri di training
training_params = {
    'output_dir': output_dir,
    'per_device_train_batch_size': 1,
    'gradient_accumulation_steps': 8,
    'num_train_epochs': 2,
    'learning_rate': 2e-5,
    'warmup_steps': 200,
    'logging_steps': 50,
    'save_steps': 250,
    'fp16': False,
    'dataloader_num_workers': 0,
    'report_to': ['tensorboard'],
    'save_total_limit': 2,
    'eval_strategy': 'steps',
    'eval_steps': 100,
    'per_device_eval_batch_size': 1
}


batch_size = 32
max_length = 650
limit_train = None 
limit_dev = None


if __name__ == "__main__":
    # Caricamento dataset
    print("Caricamento dataset CoSQL...")
    dataset_dir = 'dataset/cosql_dataset/cosql_dataset'
    train_file = os.path.join(dataset_dir, 'sql_state_tracking', 'cosql_train.json')
    dev_file = os.path.join(dataset_dir, 'sql_state_tracking', 'cosql_dev.json')
    database_dir = os.path.join(dataset_dir, 'database')

    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    with open(dev_file, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)

    print(f"Dataset caricato - Training: {len(train_data)} dialoghi, Validation: {len(dev_data)} dialoghi")

    # Processing dei dataset
    train_examples = process_training_data(train_data, limit=limit_train)
    dev_examples = process_dev_data(dev_data, limit=limit_dev)

    print(f"Examples processati - Training: {len(train_examples)}, Validation: {len(dev_examples)}")
    
    if len(train_examples) == 0:
        print("Errore: Nessun example di training valido trovato")
        exit(1)

    # Caricamento e configurazione modello
    model, tokenizer = load_model(model_id)
    model = configure_lora(model, params)

    # Creazione dataset tokenizzati
    train_dataset = create_dataset(train_examples, batch_size)
    eval_dataset = create_dataset(dev_examples, batch_size)

    print(f"Dataset tokenizzati - Training: {len(train_dataset)}, Validation: {len(eval_dataset)}")

    # Esecuzione training
    try:    
        trainer = setup_training(training_params, model, tokenizer, train_dataset, eval_dataset)
        trainer.train()
        print("Training completato con successo")
    except Exception as e:
        print(f"Errore durante il training: {e}")

    # Salvataggio modello finale
    save_model(model, tokenizer, final_dir)
    print("Fine-tuning completato")
