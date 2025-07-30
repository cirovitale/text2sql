"""
Modulo di inferenza per la generazione di query SQL da linguaggio naturale
in contesti conversazionali multi-turn utilizzando modelli linguistici fine-tuned.
"""

import os
import json
import re
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel


class SQLStoppingCriteria(StoppingCriteria):
    """
    Modulo di stopping criteria per la generazione di query SQL.
    Si ferma quando rileva la fine di una query SQL valida.
    """
    
    def __init__(self, tokenizer):
        """
        Inizializza i criteri di arresto per SQL.
        
        Args:
            tokenizer: Tokenizer del modello per la decodifica
        """
        self.tokenizer = tokenizer
        self.semicolon_token_ids = self._get_token_ids([';'])
        self.newline_after_sql_patterns = ['\n', '\nQ:', '\nSQL:', '\n\n']
        
    def _get_token_ids(self, texts):
        """
        Ottiene gli ID dei token per una lista di testi.
        
        Args:
            texts (list): Lista di stringhe da convertire in token ID
            
        Returns:
            set: Set di token ID corrispondenti
        """
        token_ids = set()
        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            for token_id in tokens:
                token_ids.add(token_id)
        return token_ids
    
    def __call__(self, input_ids, scores, **kwargs):
        """
        Verifica se il modello deve interrompere la generazione.
        
        Args:
            input_ids: Sequenza di token generati
            scores: Scores dei token
            **kwargs: Parametri aggiuntivi
            
        Returns:
            bool: True se deve fermarsi, False altrimenti
        """
        # Verifica lunghezza minima
        if len(input_ids[0]) < 5:
            return False
            
        # Analizza gli ultimi 10 token generati
        last_tokens = input_ids[0][-10:]
        last_text = self.tokenizer.decode(last_tokens, skip_special_tokens=True)
        
        # Controllo per punto e virgola seguito da pattern di fine
        if ';' in last_text:
            semicolon_index = last_text.rfind(';')
            after_semicolon = last_text[semicolon_index + 1:].strip()
            
            # Ferma se dopo il punto e virgola c'è solo whitespace o pattern di nuova query
            if (not after_semicolon or 
                after_semicolon.startswith(('Q:', 'SQL:', '\n', 'A:')) or
                len(after_semicolon) > 20):  # Se c'è troppo testo dopo ;
                return True
        
        # Controllo per pattern di nuova conversazione
        conversation_patterns = ['Q:', '\nQ:', 'SQL:', '\nSQL:', 'A:', '\nA:']
        for pattern in conversation_patterns:
            if pattern in last_text and ';' in last_text[:last_text.find(pattern)]:
                return True
                
        return False


class SQLInferenceModel:
    """
    Modulo di inferenza per la generazione di query SQL da conversazioni multi-turn.
    Supporta sia modelli base che modelli fine-tuned con adapter LoRA.
    """
    
    def __init__(self, checkpoint_path=None, base_model_id="deepseek-ai/deepseek-coder-1.3b-instruct"):
        """
        Inizializza il modello di inferenza SQL.
        
        Args:
            checkpoint_path (str, optional): Path al checkpoint del modello fine-tuned
            base_model_id (str): ID del modello base da utilizzare
        """
        self.checkpoint_path = checkpoint_path
        self.base_model_id = base_model_id
        self.model = None
        self.tokenizer = None
        self.max_length = 650  # Stesso valore usato nel training model-007
        
        print("Inizializzazione sistema di inferenza SQL...")
        self._load_model()
    
    def _load_model(self):
        """
        Carica il modello e il tokenizer.
        Se specificato un checkpoint_path, carica il modello fine-tuned con LoRA,
        altrimenti carica solo il modello base.
        """
        # Liberazione memoria
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        try:
            if self.checkpoint_path is None:
                self._load_base_model()
            else:
                self._load_finetuned_model()
                
        except Exception as e:
            print(f"Errore durante il caricamento del modello: {e}")
            raise
    
    def _load_base_model(self):
        """Carica solo il modello base senza fine-tuning."""
        print("Caricamento modello base...")
        
        # Caricamento tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Caricamento modello
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.float32,
            device_map=None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).cpu()
        
        print(f"Modello base caricato: {self.base_model_id}")
    
    def _load_finetuned_model(self):
        """Carica il modello fine-tuned con adapter LoRA."""
        print("Caricamento modello fine-tuned...")
        
        # Caricamento tokenizer dal checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Caricamento modello base
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.float32,
            device_map=None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).cpu()
        
        # Applicazione adapter LoRA
        self.model = PeftModel.from_pretrained(
            base_model,
            self.checkpoint_path,
            torch_dtype=torch.float32,
        ).cpu()
        
        print(f"Modello fine-tuned caricato da: {self.checkpoint_path}")
    
    def _format_dialogue_prompt(self, schema, dialogue_history, current_query):
        """
        Formatta il prompt per la generazione includendo schema, cronologia e query corrente.
        
        Args:
            schema (str): Schema del database
            dialogue_history (list): Cronologia del dialogo precedente
            current_query (str): Query corrente da processare
            
        Returns:
            str: Prompt formattato per il modello
        """
        system_header = f"""### SYSTEM INSTRUCTIONS ###
            You are a helpful SQL assistant. Generate ONLY a single SQL query for the current question. Stop immediately after the semicolon (;).

            ### SCHEMA ###
            {schema}

            ### DIALOGUE ###
        """
        
        dialogue_lines = []
        
        # Limita la cronologia per evitare prompt troppo lunghi
        recent_history = dialogue_history[-3:] if len(dialogue_history) > 3 else dialogue_history
        
        # Costruzione della cronologia del dialogo
        for turn in recent_history:
            query = turn.get('query', '').strip()
            response = turn.get('response', '').strip()
            
            if query and response:
                dialogue_lines.append(f"Q: {query}")
                dialogue_lines.append(f"A: {response}")
        
        # Costruzione prompt finale
        if dialogue_lines:
            prompt = f"{system_header}\n" + "\n".join(dialogue_lines) + f"\nQ: {current_query}\nA:"
        else:
            prompt = f"{system_header}\nQ: {current_query}\nA:"
        
        return prompt
    
    def generate_sql(self, schema, dialogue_history, current_query, verbose=False):
        """
        Genera una query SQL basata su schema, cronologia del dialogo e query corrente.
        
        Args:
            schema (str): Schema del database
            dialogue_history (list): Cronologia delle query precedenti
            current_query (str): Query corrente in linguaggio naturale
            verbose (bool): Se True, stampa informazioni di debug
            
        Returns:
            str: Query SQL generata
            
        Raises:
            ValueError: Se il modello non è caricato correttamente
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Modello non caricato correttamente")
        
        # Formattazione del prompt
        prompt = self._format_dialogue_prompt(schema, dialogue_history, current_query)
        
        if verbose:
            print(f"Prompt generato (lunghezza cronologia: {len(dialogue_history)}):")
            print("-" * 50)
            print(prompt)
            print("-" * 50)
        
        # Tokenizzazione
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
            padding=True
        )
        
        # Spostamento su device del modello
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Configurazione criteri di arresto
        sql_stopping_criteria = SQLStoppingCriteria(self.tokenizer)
        stopping_criteria = StoppingCriteriaList([sql_stopping_criteria])
        
        # Generazione con parametri ottimizzati
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                max_new_tokens=150,
                do_sample=True,
                temperature=0.3,
                top_p=0.8,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
            )
        
        # Decodifica della parte generata
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if verbose:
            print(f"Testo grezzo generato: '{generated_text}'")

        # Pulizia e validazione dell'output
        cleaned_sql = self._clean_generated_sql(generated_text)
        
        return cleaned_sql

    def _clean_generated_sql(self, text):
        """
        Pulisce l'output generato per estrarre una query SQL valida.
        
        Args:
            text (str): Testo generato dal modello
            
        Returns:
            str: Query SQL pulita e validata
        """
        if not text:
            return "SELECT 1;"
        
        # Rimozione di pattern di nuova conversazione
        conversation_patterns = ['\nQ:', '\nSQL:', '\nA:', '\n\n']
        for pattern in conversation_patterns:
            if pattern in text:
                text = text[:text.find(pattern)]
        
        # Ricerca della prima query SQL valida
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Rimozione di prefissi comuni
            line = re.sub(r'^(SQL:\s*|A:\s*)', '', line, flags=re.IGNORECASE)
            line = line.strip()
            
            # Verifica se inizia con keyword SQL valida
            if line.upper().startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER')):
                # Estrazione fino al primo punto e virgola
                if ';' in line:
                    line = line.split(';')[0] + ';'
                elif not line.endswith(';'):
                    line += ';'
                
                return line
        
        # Fallback: pulizia generale del testo
        clean_text = text.strip()
        clean_text = re.sub(r'^(SQL:\s*|A:\s*)', '', clean_text, flags=re.IGNORECASE)
        
        # Taglio al primo punto e virgola
        if ';' in clean_text:
            clean_text = clean_text.split(';')[0] + ';'
        else:
            # Taglio al primo newline
            if '\n' in clean_text:
                clean_text = clean_text.split('\n')[0]
            if clean_text and not clean_text.endswith(';'):
                clean_text += ';'
        
        return clean_text if clean_text else "SELECT 1;"
    
    def process_dialogue(self, schema, dialogue_array, verbose=False):
        """
        Processa un'intera conversazione multi-turn generando SQL per ogni query.
        
        Args:
            schema (str): Schema del database
            dialogue_array (list): Lista di dict contenenti le query del dialogo
            verbose (bool): Se True, stampa informazioni dettagliate per ogni turn
            
        Returns:
            list: Lista dei risultati per ogni turn con query e SQL generato
        """
        results = []
        dialogue_history = []
        
        for i, turn in enumerate(dialogue_array):
            current_query = turn.get('query', '').strip()
            if not current_query:
                continue
            
            if verbose:
                print(f"Turn {i+1}: {current_query}")
            
            # Generazione SQL per la query corrente
            generated_sql = self.generate_sql(schema, dialogue_history, current_query, verbose)
            
            # Salvataggio del risultato
            result = {
                'turn': i + 1,
                'query': current_query,
                'generated_sql': generated_sql
            }
            results.append(result)
            
            # Aggiornamento cronologia per i turn successivi
            dialogue_history.append({
                'query': current_query,
                'response': generated_sql
            })
        
        return results


def main():
    """
    Caso di studio in ambito medico.
    """
    
    # Inizializzazione del modello (utilizzare None per modello base)
    inference_model = SQLInferenceModel(
        checkpoint_path="./model-007/checkpoint-1000", 
        base_model_id="deepseek-ai/deepseek-coder-1.3b-instruct"
    )
    
    # Schema di esempio per sistema di prenotazioni mediche
    schema = """CREATE TABLE User (
        id INTEGER PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        surname VARCHAR(100) NOT NULL,
        email VARCHAR(150) UNIQUE NOT NULL,
        phone VARCHAR(20)
    );
    
    CREATE TABLE Category (
        id INTEGER PRIMARY KEY,
        name VARCHAR(100) NOT NULL CHECK (
            name IN ('Cardiology', 'Orthopedics', 'Dermatology', 'Neurology', 'General Medicine')
        )
    );
    
    CREATE TABLE Specialist (
        id INTEGER PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        surname VARCHAR(100) NOT NULL,
        category_id INTEGER NOT NULL,
        email VARCHAR(150) UNIQUE NOT NULL,
        FOREIGN KEY (category_id) REFERENCES Category(id)
    );
    
    CREATE TABLE Appointment (
        id INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL,
        specialist_id INTEGER NOT NULL,
        datetime DATETIME NOT NULL,
        status VARCHAR(20) NOT NULL CHECK (status IN ('active', 'cancelled', 'completed')),
        FOREIGN KEY (user_id) REFERENCES User(id),
        FOREIGN KEY (specialist_id) REFERENCES Specialist(id)
    );"""
    
    # Esempio di conversazione multi-turn
    dialogue_array = [
        {"query": "Hello, my name is Ciro and my surname is Vitale. I would like to know what my next appointment is."},
        {"query": "What are the appointments I have deleted?"},
        {"query": "What is my registered email?"},
        {"query": "List all available specialists in dermatology."},
        {"query": "Can I view all my appointments with Dr. Rossi?"},
        {"query": "What is the total number of appointments by category?"}
    ]
    
    print("Avvio elaborazione conversazione multi-turn...")
    print(f"Schema database: {len(schema.split('CREATE TABLE'))-1} tabelle")
    print(f"Conversazione: {len(dialogue_array)} turn")
    
    # Processing conversazione
    results = inference_model.process_dialogue(schema, dialogue_array, verbose=False)
    
    # Visualizzazione dei risultati
    print("\n" + "="*60)
    print("RISULTATI CONVERSAZIONE")
    print("="*60)
    
    for result in results:
        print(f"Turn {result['turn']}: {result['query']}")
        print(f"SQL: {result['generated_sql']}")
        print("-"*40)


if __name__ == "__main__":
    main() 