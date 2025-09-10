
import json
import numpy as np
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DomainWeights:
    """Domain-specific term weights for fraud guidance evaluation"""
    critical_contacts: Dict[str, float]
    fraud_types: Dict[str, float] 
    regulatory_terms: Dict[str, float]
    empathy_markers: Dict[str, float]
    action_words: Dict[str, float]

class DomainWeightedEvaluator:
    def __init__(self):
        # Load models
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Load spaCy for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
            
        # Initialize domain weights
        self.domain_weights = self._create_domain_weights()
        
        # Critical entity patterns for exact matching
        self.critical_entities = self._define_critical_entities()
        
    def _create_domain_weights(self) -> DomainWeights:
        """Define domain-specific term weights based on fraud guidance importance"""
        return DomainWeights(
            critical_contacts={
                '0300 123 2040': 5.0,
                'action fraud': 4.0,
                '999': 4.0,
                '101': 3.0,
                '0800 111 6768': 3.5,  # FCA helpline
                '116 123': 3.0,        # Samaritans
            },
            fraud_types={
                'app fraud': 3.0,
                'authorised push payment': 3.0,
                'vishing': 2.5,
                'voice phishing': 2.5,
                'romance scam': 2.5,
                'romance fraud': 2.5,
                'investment fraud': 2.5,
                'purchase fraud': 2.0,
                'identity theft': 2.0,
            },
            regulatory_terms={
                'psr': 3.0,
                'fca': 2.5,
                'ncsc': 2.0,
                'app code': 2.5,
                'payment services regulations': 2.5,
                'financial conduct authority': 2.5,
            },
            empathy_markers={
                'not your fault': 2.5,
                'understandable': 2.0,
                'don\'t blame yourself': 2.5,
                'not to blame': 2.5,
                'these criminals': 2.0,
                'you\'re not alone': 2.0,
            },
            action_words={
                'immediately': 2.0,
                'report': 2.0,
                'contact': 2.0,
                'never': 2.0,
                'stop': 2.0,
                'hang up': 2.5,
            }
        )
    
    def _define_critical_entities(self) -> Dict[str, Set[str]]:
        """Define critical entities that must be exact matches"""
        return {
            'phone_numbers': {
                '0300 123 2040', '999', '101', '0800 111 6768', '116 123'
            },
            'organizations': {
                'action fraud', 'fca', 'ncsc', 'psr', 'samaritans', 'victim support'
            },
            'procedures': {
                'app code', 'authorised push payment', 'payment services regulations'
            }
        }
    
    def extract_entities(self, text: str) -> Dict[str, Set[str]]:
        """Extract critical entities from text using NER and regex"""
        entities = {
            'phone_numbers': set(),
            'organizations': set(), 
            'procedures': set()
        }
        
        text_lower = text.lower()
        
        # Extract phone numbers with regex
        phone_patterns = [
            r'0300\s*123\s*2040',
            r'\b999\b',
            r'\b101\b', 
            r'0800\s*111\s*6768',
            r'\b116\s*123\b'
        ]
        
        for pattern in phone_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                # Normalize spacing
                normalized = re.sub(r'\s+', ' ', match.strip())
                entities['phone_numbers'].add(normalized)
        
        # Extract organizations and procedures
        for category, terms in self.critical_entities.items():
            if category != 'phone_numbers':
                for term in terms:
                    if term in text_lower:
                        entities[category].add(term)
        
        return entities
    
    def calculate_entity_accuracy(self, response_entities: Dict[str, Set[str]], 
                                reference_entities: Dict[str, Set[str]]) -> Dict[str, float]:
        """Calculate entity-level accuracy scores"""
        scores = {}
        
        for category in ['phone_numbers', 'organizations', 'procedures']:
            response_set = response_entities.get(category, set())
            reference_set = reference_entities.get(category, set())
            
            if not reference_set:
                scores[category] = 1.0  # No reference entities to match
                continue
                
            if not response_set:
                scores[category] = 0.0  # Missing all reference entities
                continue
                
            # Calculate intersection over union
            intersection = response_set & reference_set
            union = response_set | reference_set
            
            scores[category] = len(intersection) / len(reference_set) if reference_set else 1.0
        
        return scores
    
    def apply_domain_weights(self, text: str) -> np.ndarray:
        """Apply domain-specific weights to text embedding"""
        # Get base embedding
        base_embedding = self.embedding_model.encode([text])[0]
        
        # Calculate weight multiplier based on domain terms
        text_lower = text.lower()
        weight_multiplier = 1.0
        
        # Check for domain terms and apply weights
        all_weights = {}
        for weight_dict in [
            self.domain_weights.critical_contacts,
            self.domain_weights.fraud_types,
            self.domain_weights.regulatory_terms,
            self.domain_weights.empathy_markers,
            self.domain_weights.action_words
        ]:
            all_weights.update(weight_dict)
        
        # Calculate weighted importance
        importance_score = 0
        for term, weight in all_weights.items():
            if term in text_lower:
                importance_score += weight
        
        # Apply logarithmic scaling to prevent extreme weights
        weight_multiplier = 1 + np.log(1 + importance_score * 0.1)
        
        # Scale embedding by weight multiplier
        weighted_embedding = base_embedding * weight_multiplier
        
        return weighted_embedding
    
    def evaluate_response_pair(self, question: str, baseline_response: str, 
                              finetuned_response: str, reference_response: str) -> Dict:
        """Evaluate BM vs FM response against reference using combined approach"""
        
        # 1. Domain-weighted cosine similarity
        ref_embedding = self.apply_domain_weights(reference_response)
        bm_embedding = self.apply_domain_weights(baseline_response)  
        fm_embedding = self.apply_domain_weights(finetuned_response)
        
        bm_similarity = cosine_similarity([bm_embedding], [ref_embedding])[0][0]
        fm_similarity = cosine_similarity([fm_embedding], [ref_embedding])[0][0]
        
        # 2. Entity accuracy evaluation
        ref_entities = self.extract_entities(reference_response)
        bm_entities = self.extract_entities(baseline_response)
        fm_entities = self.extract_entities(finetuned_response)
        
        bm_entity_scores = self.calculate_entity_accuracy(bm_entities, ref_entities)
        fm_entity_scores = self.calculate_entity_accuracy(fm_entities, ref_entities)
        
        # 3. Combined scoring
        bm_entity_avg = np.mean(list(bm_entity_scores.values()))
        fm_entity_avg = np.mean(list(fm_entity_scores.values()))
        
        # Weighted combination: 60% semantic similarity, 40% entity accuracy
        bm_composite = (0.6 * bm_similarity) + (0.4 * bm_entity_avg)
        fm_composite = (0.6 * fm_similarity) + (0.4 * fm_entity_avg)
        
        return {
            'baseline_semantic': float(bm_similarity),
            'finetuned_semantic': float(fm_similarity),
            'baseline_entity': float(bm_entity_avg),
            'finetuned_entity': float(fm_entity_avg),
            'baseline_composite': float(bm_composite),
            'finetuned_composite': float(fm_composite),
            'improvement': float(fm_composite - bm_composite),
            'semantic_improvement': float(fm_similarity - bm_similarity),
            'entity_improvement': float(fm_entity_avg - bm_entity_avg),
            'baseline_entities': {k: list(v) for k, v in bm_entities.items()},
            'finetuned_entities': {k: list(v) for k, v in fm_entities.items()},
            'reference_entities': {k: list(v) for k, v in ref_entities.items()}
        }
    
    def load_ground_truth(self) -> List[Dict]:
        """Load 1000 Q&A pairs as ground truth reference"""
        try:
            with open('model_training/1000_master_fraud_qa_dataset.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error("Ground truth dataset not found")
            return []
    
    def find_best_reference_match(self, question: str, ground_truth: List[Dict]) -> str:
        """Find best matching reference response from ground truth dataset"""
        question_embedding = self.embedding_model.encode([question])[0]
        
        best_match = ""
        best_score = 0
        
        for item in ground_truth:
            instruction_embedding = self.embedding_model.encode([item['instruction']])[0]
            similarity = cosine_similarity([question_embedding], [instruction_embedding])[0][0]
            
            if similarity > best_score:
                best_score = similarity
                best_match = item['output']
        
        return best_match
    
    def run_evaluation(self) -> Dict:
        """Run complete technical evaluation"""
        logger.info("Loading datasets...")
        
        # Load evaluation responses  
        with open('Models_Responses.json', 'r') as f:
            evaluation_data = json.load(f)
        
        # Load ground truth
        ground_truth = self.load_ground_truth()
        
        results = []
        
        logger.info(f"Evaluating {len(evaluation_data)} response pairs...")
        
        for item in evaluation_data:
            question = item['question']
            baseline_response = item['BM']
            finetuned_response = item['FM']
            
            # Find best reference match from ground truth
            reference_response = self.find_best_reference_match(question, ground_truth)
            
            if not reference_response:
                logger.warning(f"No reference found for question: {question[:50]}...")
                continue
            
            # Evaluate response pair
            eval_result = self.evaluate_response_pair(
                question, baseline_response, finetuned_response, reference_response
            )
            
            # Add metadata
            eval_result.update({
                'question_id': item['question_no'],
                'category': item['category'],
                'question': question,
                'baseline_response': baseline_response,
                'finetuned_response': finetuned_response,
                'reference_response': reference_response
            })
            
            results.append(eval_result)
            
            logger.info(f"Evaluated {item['question_no']}: FM improvement = {eval_result['improvement']:.3f}")
        
        return self._calculate_summary_statistics(results)
    
    def _calculate_summary_statistics(self, results: List[Dict]) -> Dict:
        """Calculate comprehensive summary statistics"""
        if not results:
            return {}
        
        # Extract scores
        bm_semantic = [r['baseline_semantic'] for r in results]
        fm_semantic = [r['finetuned_semantic'] for r in results]
        bm_entity = [r['baseline_entity'] for r in results]
        fm_entity = [r['finetuned_entity'] for r in results]
        bm_composite = [r['baseline_composite'] for r in results]
        fm_composite = [r['finetuned_composite'] for r in results]
        improvements = [r['improvement'] for r in results]
        
        # Statistical analysis
        from scipy import stats
        
        # Paired t-test for statistical significance
        t_stat, p_value = stats.ttest_rel(fm_composite, bm_composite)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((np.std(bm_composite)**2) + (np.std(fm_composite)**2)) / 2)
        cohens_d = (np.mean(fm_composite) - np.mean(bm_composite)) / pooled_std
        
        summary = {
            'detailed_results': results,
            'summary_statistics': {
                'total_evaluations': len(results),
                'baseline_semantic_mean': float(np.mean(bm_semantic)),
                'finetuned_semantic_mean': float(np.mean(fm_semantic)),
                'baseline_entity_mean': float(np.mean(bm_entity)),
                'finetuned_entity_mean': float(np.mean(fm_entity)),
                'baseline_composite_mean': float(np.mean(bm_composite)),
                'finetuned_composite_mean': float(np.mean(fm_composite)),
                'mean_improvement': float(np.mean(improvements)),
                'improvement_std': float(np.std(improvements)),
                'improvement_consistency': float(np.sum(np.array(improvements) > 0) / len(improvements)),
                'statistical_significance': {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': bool(p_value < 0.05),
                    'cohens_d': float(cohens_d),
                    'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
                }
            }
        }
        
        return summary

def main():
    """Main evaluation execution"""
    evaluator = DomainWeightedEvaluator()
    results = evaluator.run_evaluation()
    
    if not results:
        logger.error("Evaluation failed - no results generated")
        return
    
    # Save results
    output_dir = Path("technical_evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "domain_weighted_evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    stats = results['summary_statistics']
    print("\n" + "="*60)
    print("DOMAIN-WEIGHTED TECHNICAL EVALUATION RESULTS")
    print("="*60)
    print(f"Total Evaluations: {stats['total_evaluations']}")
    print(f"Baseline Composite Score: {stats['baseline_composite_mean']:.3f}")
    print(f"Fine-tuned Composite Score: {stats['finetuned_composite_mean']:.3f}")
    print(f"Mean Improvement: {stats['mean_improvement']:.3f} ± {stats['improvement_std']:.3f}")
    print(f"Improvement Consistency: {stats['improvement_consistency']*100:.1f}%")
    print(f"\nStatistical Significance:")
    print(f"  p-value: {stats['statistical_significance']['p_value']:.4f}")
    print(f"  Effect Size: {stats['statistical_significance']['effect_size']} (d = {stats['statistical_significance']['cohens_d']:.3f})")
    print(f"  Significant: {stats['statistical_significance']['significant']}")
    
    logger.info(f"Results saved to {output_dir}/")

if __name__ == "__main__":
    main()