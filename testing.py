"""
Modulo per la valutazione (testing) di modelli Text-to-SQL su dataset CoSQL 
utilizzando metriche Exact Set Match per conversazioni multi-turn.
"""

import os
import json
import re
from collections import defaultdict
from typing import List, Dict, Set
import sqlparse
from sqlparse import sql, tokens

from inference import SQLInferenceModel


def load_database_schema(database_id: str, database_dir: str) -> str:
    """
    Carica e processa lo schema di un database dal file SQL corrispondente.
    
    Args:
        database_id: Identificativo del database
        database_dir: Directory contenente i database
        
    Returns:
        Schema formattato con statements CREATE TABLE o None se non disponibile
    """
    schema_file = os.path.join(database_dir, database_id, "schema.sql")
    
    if not os.path.exists(schema_file):
        print(f"Database: {database_id}, NO SCHEMA AVAILABLE")
        return None
    
    try:
        with open(schema_file, 'r', encoding='utf-8', errors='ignore') as f:
            sql_content = f.read()
        
        create_tables = []
        table_pattern = r'(CREATE\s+TABLE\s+[^;]+;)'
        matches = re.findall(table_pattern, sql_content, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            clean_table = re.sub(r'\s+', ' ', match.strip())
            create_tables.append(clean_table)
        
        final_schema = "\n\n".join(create_tables) if create_tables else None
        return final_schema
        
    except Exception as e:
        print(f"Errore durante la lettura dello schema {database_id}: {e}")
        return None


class ExactSetMatcher:
    """
    Implementazione della metrica Exact Set Match per Text-to-SQL.
    
    Verifica se il set di componenti della query SQL predetta coincide
    esattamente con quello della query di riferimento.
    """
    
    def __init__(self):
        """Inizializza il matcher per confronti exact set match."""
        pass
    
    def normalize_sql(self, sql_query: str) -> str:
        """
        Normalizza una query SQL per il confronto strutturale.
        
        Args:
            sql_query: Query SQL da normalizzare
            
        Returns:
            Query SQL normalizzata con formattazione standard
        """
        try:
            # Parsing con sqlparse
            parsed = sqlparse.parse(sql_query.strip())[0]
            
            # Formattazione standard
            formatted = sqlparse.format(
                str(parsed),
                keyword_case='upper',
                identifier_case='lower',
                strip_comments=True,
                reindent=False
            )
            
            # Normalizzazione spazi e rimozione punto e virgola finale
            normalized = ' '.join(formatted.split())
            if normalized.endswith(';'):
                normalized = normalized[:-1]
                
            return normalized.strip()
            
        except Exception as e:
            # Fallback: normalizzazione semplice
            sql_clean = ' '.join(sql_query.strip().replace(';', '').split())
            return sql_clean.upper()
    
    def extract_sql_components(self, sql_query: str) -> Set[str]:
        """
        Estrae il set di componenti SQL per l'exact set matching.
        
        Ogni componente rappresenta una clausola o elemento strutturale
        della query (keywords, nomi, operatori, funzioni, etc.).
        
        Args:
            sql_query: Query SQL da analizzare
            
        Returns:
            Set di componenti estratti dalla query
        """
        try:
            parsed = sqlparse.parse(sql_query)[0]
            components = set()
            
            # Estrazione componenti principali tramite parsing
            for token in parsed.flatten():
                if token.ttype is tokens.Keyword:
                    keyword = token.value.upper().strip()
                    if keyword in ['SELECT', 'FROM', 'WHERE', 'GROUP', 'BY', 'HAVING', 
                                  'ORDER', 'LIMIT', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 
                                  'ON', 'AND', 'OR', 'DISTINCT']:
                        components.add(f"KEYWORD:{keyword}")
                        
                elif token.ttype is tokens.Name:
                    # Nomi di tabelle, colonne, alias
                    name = token.value.lower().strip()
                    if name and len(name) > 1:
                        components.add(f"NAME:{name}")
                        
                elif token.ttype is tokens.Name.Builtin:
                    # Funzioni di aggregazione
                    func = token.value.upper().strip()
                    components.add(f"FUNCTION:{func}")
                    
                elif token.ttype is tokens.Operator:
                    # Operatori di confronto
                    op = token.value.strip()
                    if op in ['=', '>', '<', '>=', '<=', '!=', '<>', 'LIKE', 'IN']:
                        components.add(f"OPERATOR:{op}")
                        
                elif token.ttype is tokens.Number.Integer:
                    # Valori numerici
                    components.add(f"NUMBER:{token.value}")
                    
                elif token.ttype is tokens.String.Single:
                    # Stringhe letterali
                    components.add(f"STRING:{token.value}")
            
            return components
            
        except Exception as e:
            # Fallback: estrazione con regex
            return self._extract_simple_components(sql_query)
    
    def _extract_simple_components(self, sql_query: str) -> Set[str]:
        """
        Estrazione di componenti tramite regex come fallback.
        
        Args:
            sql_query: Query SQL da analizzare
            
        Returns:
            Set di componenti estratti tramite pattern regex
        """
        components = set()
        sql_upper = sql_query.upper()
        
        # Keywords principali
        keywords = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY', 
                   'JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'DISTINCT']
        for keyword in keywords:
            if keyword in sql_upper:
                components.add(f"KEYWORD:{keyword}")
        
        # Funzioni di aggregazione
        aggregates = re.findall(r'\b(COUNT|SUM|AVG|MIN|MAX)\s*\(', sql_query, re.IGNORECASE)
        for agg in aggregates:
            components.add(f"FUNCTION:{agg.upper()}")
        
        # Identificatori (tabelle, colonne)
        names = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', sql_query)
        
        for name in names:
            name_clean = name.lower()
            if name_clean not in ['select', 'from', 'where', 'group', 'by', 'having', 
                                 'order', 'join', 'inner', 'left', 'right', 'on', 
                                 'and', 'or', 'distinct', 'count', 'sum', 'avg', 'min', 'max']:
                components.add(f"NAME:{name_clean}")
        
        return components
    
    def exact_set_match(self, predicted_sql: str, gold_sql: str) -> bool:
        """
        Verifica se due query SQL hanno un exact set match.
        
        Implementa la metrica Question Match: una predizione è corretta
        se il set di componenti coincide esattamente con la query di riferimento.
        
        Args:
            predicted_sql: Query SQL predetta dal modello
            gold_sql: Query SQL di riferimento (ground truth)
            
        Returns:
            True se le query hanno exact set match, False altrimenti
        """
        try:
            # Primo tentativo: confronto diretto dopo normalizzazione
            pred_normalized = self.normalize_sql(predicted_sql)
            gold_normalized = self.normalize_sql(gold_sql)
            
            if pred_normalized == gold_normalized:
                return True
            
            # Secondo tentativo: confronto basato su componenti
            pred_components = self.extract_sql_components(predicted_sql)
            gold_components = self.extract_sql_components(gold_sql)
            
            # Exact set match: i set devono essere identici
            return pred_components == gold_components
            
        except Exception as e:
            print(f"Errore durante il confronto SQL: {e}")
            # Fallback: confronto stringa normalizzata
            pred_clean = ' '.join(predicted_sql.strip().lower().split())
            gold_clean = ' '.join(gold_sql.strip().lower().split())
            return pred_clean == gold_clean


class CoSQLEvaluator:
    """
    Valutatore per dataset CoSQL con metriche Exact Set Match.
    
    Implementa le metriche Question Match e Interaction Match per la
    valutazione di modelli Text-to-SQL su conversazioni multi-turn.
    """
    
    def __init__(self, 
                 checkpoint_path: str = None,
                 base_model_id: str = "deepseek-ai/deepseek-coder-1.3b-instruct",
                 dataset_dir: str = 'dataset/cosql_dataset/cosql_dataset'):
        """
        Inizializza il valutatore CoSQL.
        
        Args:
            checkpoint_path: Path al checkpoint del modello fine-tuned (None per modello base)
            base_model_id: ID del modello base da utilizzare
            dataset_dir: Directory contenente il dataset CoSQL
        """
        self.dataset_dir = dataset_dir
        self.database_dir = os.path.join(dataset_dir, 'database')
        self.dev_file = os.path.join(dataset_dir, 'sql_state_tracking', 'cosql_dev.json')
        
        # Inizializzazione modello di inferenza
        self.inference_model = SQLInferenceModel(checkpoint_path, base_model_id)
        self.matcher = ExactSetMatcher()
        
        # Statistiche di valutazione
        self.stats = {
            'total_questions': 0,
            'correct_questions': 0,
            'total_interactions': 0,
            'correct_interactions': 0
        }
    
    def load_test_data(self, limit: int = None) -> List[Dict]:
        """
        Carica i dati di test dal file CoSQL dev.
        
        Args:
            limit: Numero massimo di dialoghi da caricare (None per tutti)
            
        Returns:
            Lista dei dialoghi di test caricati
        """
        print("Caricamento dataset di test...")
        
        with open(self.dev_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        if limit:
            test_data = test_data[:limit]
        
        print(f"Caricati {len(test_data)} dialoghi di test")
        return test_data
    
    def evaluate_interaction(self, dialog: Dict) -> Dict:
        """
        Valuta una singola interazione multi-turn.
        
        Calcola metriche Question Match e Interaction Match per il dialogo.
        
        Args:
            dialog: Dizionario contenente database_id e lista di interaction
            
        Returns:
            Dizionario con risultati della valutazione per l'interazione
        """
        database_id = dialog.get('database_id', '')
        interactions = dialog.get('interaction', [])
        
        if not database_id or not interactions:
            return {'valid': False}
        
        # Caricamento schema database
        schema = load_database_schema(database_id, self.database_dir)
        
        if schema is None:
            return {'valid': False}
        
        # Processing dialogo
        dialogue_history = []
        turn_results = []
        interaction_correct = True
        
        for i, turn in enumerate(interactions):
            utterance = turn.get('utterance', '').strip()
            gold_query = turn.get('query', '').strip()
            
            if not utterance or not gold_query:
                continue
            
            # Generazione predizione SQL
            try:
                predicted_sql = self.inference_model.generate_sql(schema, dialogue_history, utterance)
            except Exception as e:
                print(f"Errore generazione SQL per turn {i+1}: {e}")
                predicted_sql = "SELECT 1;"
            
            # Calcolo Question Match
            is_question_correct = self.matcher.exact_set_match(predicted_sql, gold_query)
            
            # Aggiornamento statistiche
            self.stats['total_questions'] += 1
            if is_question_correct:
                self.stats['correct_questions'] += 1
            else:
                interaction_correct = False  # Se una domanda è sbagliata, tutta l'interazione è sbagliata
            
            # Salvataggio risultato del turn
            turn_result = {
                'turn': i + 1,
                'utterance': utterance,
                'gold_sql': gold_query,
                'predicted_sql': predicted_sql,
                'question_match': is_question_correct
            }
            turn_results.append(turn_result)
            
            # Aggiornamento cronologia per turn successivi
            dialogue_history.append({
                'query': utterance,
                'response': predicted_sql
            })
        
        # Calcolo Interaction Match
        self.stats['total_interactions'] += 1
        if interaction_correct and len(turn_results) > 0:
            self.stats['correct_interactions'] += 1
        
        return {
            'valid': True,
            'database_id': database_id,
            'interaction_match': interaction_correct,
            'turn_results': turn_results
        }
    
    def run_evaluation(self, limit: int = None, verbose: bool = True) -> Dict:
        """
        Esegue la valutazione completa con metriche Exact Set Match.
        
        Args:
            limit: Numero massimo di dialoghi da valutare (None per tutti)
            verbose: Se True, stampa informazioni dettagliate durante la valutazione
            
        Returns:
            Dizionario con tutti i risultati della valutazione
        """
        print("Avvio valutazione CoSQL - Exact Set Match")
        print("=" * 60)
        
        # Caricamento dati di test
        test_data = self.load_test_data(limit)
        
        # Valutazione di ogni interazione
        all_results = []
        
        for idx, dialog in enumerate(test_data):
            if verbose and (idx + 1) % 10 == 0:
                print(f"Processati {idx + 1}/{len(test_data)} dialoghi...")
            
            result = self.evaluate_interaction(dialog)
            if result['valid']:
                all_results.append(result)
                
                # Visualizzazione dettagli
                if verbose:
                    print(f"\nDialogo {idx + 1} ({result['database_id']}):")
                    for turn in result['turn_results']:
                        status = "CORRECT" if turn['question_match'] else "INCORRECT"
                        print(f"  Turn {turn['turn']} [{status}]: {turn['utterance'][:50]}...")
                        if not turn['question_match']:
                            print(f"     Gold: {turn['gold_sql']}")
                            print(f"     Pred: {turn['predicted_sql']}")
        
        # Calcolo metriche finali
        question_match_score = (self.stats['correct_questions'] / self.stats['total_questions'] 
                               if self.stats['total_questions'] > 0 else 0.0)
        
        interaction_match_score = (self.stats['correct_interactions'] / self.stats['total_interactions'] 
                                  if self.stats['total_interactions'] > 0 else 0.0)
        
        results = {
            'question_match': f"{question_match_score:.4f}",
            'interaction_match': f"{interaction_match_score:.4f}",
            'total_questions': self.stats['total_questions'],
            'correct_questions': self.stats['correct_questions'],
            'total_interactions': self.stats['total_interactions'],
            'correct_interactions': self.stats['correct_interactions'],
            'detailed_results': all_results
        }
        
        return results
    
    def print_results(self, results: Dict):
        """
        Stampa i risultati della valutazione in formato leggibile.
        
        Args:
            results: Dizionario contenente tutti i risultati della valutazione
        """
        print("\n" + "=" * 60)
        print("RISULTATI VALUTAZIONE CoSQL - EXACT SET MATCH")
        print("=" * 60)
        
        print(f"Question Match: {results['question_match']:.4f} ({results['correct_questions']}/{results['total_questions']})")
        print(f"Interaction Match: {results['interaction_match']:.4f} ({results['correct_interactions']}/{results['total_interactions']})")
        
        print(f"\nStatistiche dettagliate:")
        print(f"   Domande totali valutate: {results['total_questions']}")
        print(f"   Domande con exact match: {results['correct_questions']}")
        print(f"   Interazioni totali valutate: {results['total_interactions']}")
        print(f"   Interazioni con exact match completo: {results['correct_interactions']}")
        
        # Analisi performance per database
        db_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        for interaction in results['detailed_results']:
            db_id = interaction['database_id']
            db_stats[db_id]['total'] += 1
            if interaction['interaction_match']:
                db_stats[db_id]['correct'] += 1
        
        # print(f"\nPerformance per database (top 10):")
        # sorted_dbs = sorted(db_stats.items(), key=lambda x: x[1]['total'], reverse=True)[:10]
        # for db_id, stats in sorted_dbs:
        #     accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        #     print(f"   {db_id}: {accuracy:.3f} ({stats['correct']}/{stats['total']})")


def main():    
    # Configurazione parametri
    CHECKPOINT_PATH = "./model-007/checkpoint-1000"
    BASE_MODEL_ID = "deepseek-ai/deepseek-coder-1.3b-instruct"
    DATASET_DIR = 'dataset/cosql_dataset/cosql_dataset'
    
    # Limite per test rapido (None per valutare tutto il dev set)
    TEST_LIMIT = 50
    
    try:
        # Inizializzazione evaluator
        evaluator = CoSQLEvaluator(
            checkpoint_path=CHECKPOINT_PATH,
            base_model_id=BASE_MODEL_ID,
            dataset_dir=DATASET_DIR
        )
        
        results = evaluator.run_evaluation(limit=TEST_LIMIT, verbose=True)
        
        evaluator.print_results(results)
        
        results_file = "results_000.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nRisultati salvati in: {results_file}")
        
    except Exception as e:
        print(f"Errore durante la valutazione: {e}")
        raise


if __name__ == "__main__":
    main()