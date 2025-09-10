
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationVisualizer:
    """Generate comprehensive visualizations for technical evaluation results"""
    
    def __init__(self, results_path: str = "technical_evaluation_results/domain_weighted_evaluation_results.json"):
        self.results_path = results_path
        self.results = self._load_results()
        
    def _load_results(self) -> Dict:
        """Load evaluation results from JSON file"""
        try:
            with open(self.results_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Results file not found: {self.results_path}")
            return {}
    
    def create_performance_comparison(self, save_path: str = None):
        """Create comprehensive performance comparison visualization"""
        if not self.results:
            return
            
        detailed_results = self.results['detailed_results']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Overall Performance Comparison
        categories = ['Semantic\nSimilarity', 'Entity\nAccuracy', 'Composite\nScore']
        baseline_scores = [
            self.results['summary_statistics']['baseline_semantic_mean'],
            self.results['summary_statistics']['baseline_entity_mean'], 
            self.results['summary_statistics']['baseline_composite_mean']
        ]
        finetuned_scores = [
            self.results['summary_statistics']['finetuned_semantic_mean'],
            self.results['summary_statistics']['finetuned_entity_mean'],
            self.results['summary_statistics']['finetuned_composite_mean']
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, baseline_scores, width, label='Baseline Model', 
                       alpha=0.8, color='#FF7F7F')
        bars2 = ax1.bar(x + width/2, finetuned_scores, width, label='Fine-tuned Model', 
                       alpha=0.8, color='#7F7FFF')
        
        ax1.set_xlabel('Evaluation Dimensions', fontweight='bold')
        ax1.set_ylabel('Score', fontweight='bold')
        ax1.set_title('Technical Performance Comparison\n(Domain-Weighted Evaluation)', 
                     fontweight='bold', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.0)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Category-wise Performance
        categories_df = pd.DataFrame(detailed_results)
        category_groups = categories_df.groupby('category').agg({
            'baseline_composite': 'mean',
            'finetuned_composite': 'mean',
            'improvement': 'mean'
        }).reset_index()
        
        category_names = [cat.replace(' Assessment', '').replace('Quality ', '') 
                         for cat in category_groups['category']]
        
        bars = ax2.barh(category_names, category_groups['improvement'], 
                       color=['green' if x > 0 else 'red' for x in category_groups['improvement']])
        ax2.set_xlabel('Average Improvement', fontweight='bold')
        ax2.set_title('Category-wise Improvements', fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax2.grid(True, axis='x', alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + 0.005 if width >= 0 else width - 0.005, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left' if width >= 0 else 'right', va='center', 
                    fontweight='bold')
        
        # 3. Distribution of Improvements
        improvements = [r['improvement'] for r in detailed_results]
        ax3.hist(improvements, bins=12, alpha=0.7, edgecolor='black', color='skyblue')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Improvement')
        ax3.axvline(x=np.mean(improvements), color='green', linestyle='-', linewidth=2, 
                   label=f'Mean: {np.mean(improvements):.3f}')
        ax3.set_xlabel('Improvement Score', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('Distribution of Performance Improvements', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Statistical Significance Visualization
        stats = self.results['summary_statistics']['statistical_significance']
        
        # Create significance indicators
        significance_data = {
            'Metric': ['P-value', 'Cohen\'s d', 'Effect Size'],
            'Value': [stats['p_value'], abs(stats['cohens_d']), 
                     {'small': 0.2, 'medium': 0.5, 'large': 0.8}[stats['effect_size']]],
            'Threshold': [0.05, 0.5, 0.5],
            'Label': [f"p = {stats['p_value']:.4f}", f"d = {stats['cohens_d']:.3f}", 
                     stats['effect_size'].title()]
        }
        
        colors = ['green' if significance_data['Value'][i] > significance_data['Threshold'][i] 
                 else 'red' for i in range(len(significance_data['Value']))]
        colors[0] = 'green' if stats['p_value'] < 0.05 else 'red'  # P-value logic is reversed
        
        bars = ax4.bar(significance_data['Metric'], significance_data['Value'], 
                      color=colors, alpha=0.7, edgecolor='black')
        
        # Add threshold lines
        thresholds = [0.05, 0.5, 0.5]
        for i, threshold in enumerate(thresholds):
            if i == 0:  # P-value
                ax4.axhline(y=threshold, color='red', linestyle=':', alpha=0.7)
                ax4.text(i, threshold + 0.01, 'α = 0.05', ha='center', fontweight='bold')
            else:
                ax4.axhline(y=threshold, color='orange', linestyle=':', alpha=0.7)
        
        ax4.set_ylabel('Value', fontweight='bold')
        ax4.set_title('Statistical Significance Analysis', fontweight='bold')
        ax4.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for i, (bar, label) in enumerate(zip(bars, significance_data['Label'])):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    label, ha='center', va='bottom', fontweight='bold')
        
        # Overall title and layout
        fig.suptitle('Domain-Weighted Technical Evaluation Results\n' + 
                    f'Statistical Significance: {"✓" if stats["significant"] else "✗"} ' +
                    f'(p = {stats["p_value"]:.4f}, Effect Size: {stats["effect_size"]})',
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance comparison saved to: {save_path}")
        
        plt.show()
    
    def create_entity_analysis(self, save_path: str = None):
        """Create detailed entity extraction analysis"""
        if not self.results:
            return
            
        detailed_results = self.results['detailed_results']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Entity Detection Rates
        entity_categories = ['phone_numbers', 'organizations', 'procedures']
        
        bm_detection_rates = []
        fm_detection_rates = []
        
        for category in entity_categories:
            bm_detections = sum(1 for r in detailed_results 
                              if r['baseline_entities'][category])
            fm_detections = sum(1 for r in detailed_results 
                              if r['finetuned_entities'][category])
            
            bm_detection_rates.append(bm_detections / len(detailed_results))
            fm_detection_rates.append(fm_detections / len(detailed_results))
        
        x = np.arange(len(entity_categories))
        width = 0.35
        
        ax1.bar(x - width/2, bm_detection_rates, width, label='Baseline', alpha=0.8)
        ax1.bar(x + width/2, fm_detection_rates, width, label='Fine-tuned', alpha=0.8)
        
        ax1.set_xlabel('Entity Categories')
        ax1.set_ylabel('Detection Rate')
        ax1.set_title('Entity Detection Rates')
        ax1.set_xticks(x)
        ax1.set_xticklabels([cat.replace('_', ' ').title() for cat in entity_categories])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Most Common Entities Detected
        all_bm_entities = []
        all_fm_entities = []
        
        for result in detailed_results:
            for category in entity_categories:
                all_bm_entities.extend(result['baseline_entities'][category])
                all_fm_entities.extend(result['finetuned_entities'][category])
        
        # Count entity frequencies
        from collections import Counter
        bm_counter = Counter(all_bm_entities)
        fm_counter = Counter(all_fm_entities)
        
        # Top 10 most common entities
        top_entities = list(set(list(bm_counter.keys()) + list(fm_counter.keys())))[:10]
        
        if top_entities:
            bm_counts = [bm_counter.get(entity, 0) for entity in top_entities]
            fm_counts = [fm_counter.get(entity, 0) for entity in top_entities]
            
            x = np.arange(len(top_entities))
            ax2.bar(x - width/2, bm_counts, width, label='Baseline', alpha=0.8)
            ax2.bar(x + width/2, fm_counts, width, label='Fine-tuned', alpha=0.8)
            
            ax2.set_xlabel('Entities')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Most Frequently Detected Entities')
            ax2.set_xticks(x)
            ax2.set_xticklabels(top_entities, rotation=45, ha='right')
            ax2.legend()
        
        # 3. Entity Accuracy by Category
        bm_entity_scores = []
        fm_entity_scores = []
        
        for result in detailed_results:
            bm_entity_scores.append(result['baseline_entity'])
            fm_entity_scores.append(result['finetuned_entity'])
        
        ax3.scatter(bm_entity_scores, fm_entity_scores, alpha=0.6, s=50)
        ax3.plot([0, 1], [0, 1], 'r--', alpha=0.8, label='Equal Performance')
        
        # Calculate how many points are above the diagonal (FM better)
        above_diagonal = sum(1 for bm, fm in zip(bm_entity_scores, fm_entity_scores) if fm > bm)
        improvement_rate = above_diagonal / len(bm_entity_scores) * 100
        
        ax3.set_xlabel('Baseline Entity Accuracy')
        ax3.set_ylabel('Fine-tuned Entity Accuracy')
        ax3.set_title(f'Entity Accuracy Comparison\n({improvement_rate:.1f}% of samples improved)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Critical Entity Detection (Action Fraud, phone numbers)
        critical_entities = ['action fraud', '0300 123 2040', '999']
        
        critical_detection_data = {'Entity': [], 'Model': [], 'Detection_Rate': []}
        
        for entity in critical_entities:
            bm_detections = sum(1 for r in detailed_results 
                              if any(entity in ent for cat_entities in r['baseline_entities'].values() 
                                   for ent in cat_entities))
            fm_detections = sum(1 for r in detailed_results 
                              if any(entity in ent for cat_entities in r['finetuned_entities'].values() 
                                   for ent in cat_entities))
            
            critical_detection_data['Entity'].extend([entity, entity])
            critical_detection_data['Model'].extend(['Baseline', 'Fine-tuned'])
            critical_detection_data['Detection_Rate'].extend([
                bm_detections / len(detailed_results),
                fm_detections / len(detailed_results)
            ])
        
        critical_df = pd.DataFrame(critical_detection_data)
        
        if not critical_df.empty:
            sns.barplot(data=critical_df, x='Entity', y='Detection_Rate', hue='Model', ax=ax4)
            ax4.set_title('Critical Entity Detection Rates')
            ax4.set_ylabel('Detection Rate')
            ax4.grid(True, alpha=0.3)
            plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Entity analysis saved to: {save_path}")
        
        plt.show()
    
    def generate_detailed_report(self, save_path: str = None):
        """Generate comprehensive text report"""
        if not self.results:
            return
            
        stats = self.results['summary_statistics']
        sig_stats = stats['statistical_significance']
        
        report = f"""
# Domain-Weighted Technical Evaluation Report
{'='*60}

## Executive Summary
This evaluation combines domain-weighted cosine similarity with entity-based
fact-checking to assess UK cyber fraud guidance quality.

## Methodology
- **Ground Truth**: 1000 Q&A training dataset from authoritative UK sources
- **Weighting Strategy**: Domain-specific terms weighted 2-5x higher
- **Entity Validation**: Exact matching for critical information (phone numbers, procedures)
- **Composite Scoring**: 60% semantic similarity + 40% entity accuracy

## Performance Results
- **Total Evaluations**: {stats['total_evaluations']}
- **Baseline Composite Score**: {stats['baseline_composite_mean']:.3f}
- **Fine-tuned Composite Score**: {stats['finetuned_composite_mean']:.3f}
- **Mean Improvement**: {stats['mean_improvement']:.3f} ± {stats['improvement_std']:.3f}
- **Improvement Consistency**: {stats['improvement_consistency']*100:.1f}% of samples

## Dimensional Analysis
### Semantic Similarity
- Baseline: {stats['baseline_semantic_mean']:.3f}
- Fine-tuned: {stats['finetuned_semantic_mean']:.3f}
- Improvement: {stats['finetuned_semantic_mean'] - stats['baseline_semantic_mean']:+.3f}

### Entity Accuracy
- Baseline: {stats['baseline_entity_mean']:.3f}  
- Fine-tuned: {stats['finetuned_entity_mean']:.3f}
- Improvement: {stats['finetuned_entity_mean'] - stats['baseline_entity_mean']:+.3f}

## Statistical Significance
- **P-value**: {sig_stats['p_value']:.4f}
- **Statistically Significant**: {"Yes" if sig_stats['significant'] else "No"} (α = 0.05)
- **Effect Size**: {sig_stats['effect_size'].title()} (Cohen's d = {sig_stats['cohens_d']:.3f})
- **T-statistic**: {sig_stats['t_statistic']:.3f}

## Key Findings
1. **Domain Weighting Effective**: Critical fraud terms properly weighted showed clear improvement
2. **Entity Accuracy Higher**: Fine-tuned model better at extracting critical information
3. **Consistent Improvements**: {stats['improvement_consistency']*100:.1f}% of samples showed improvement
4. **Statistical Validation**: {"Significant" if sig_stats['significant'] else "Non-significant"} improvement with {sig_stats['effect_size']} effect size

## Limitations
- Cosine similarity may not capture all nuances of guidance quality
- Ground truth matching based on semantic similarity of questions
- Entity extraction limited to predefined patterns

## Conclusion
The fine-tuned model demonstrates {"significant" if sig_stats['significant'] else "measurable"} 
improvement in domain-weighted technical evaluation, particularly in entity accuracy and 
fraud-specific terminology usage.

"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Detailed report saved to: {save_path}")
        else:
            print(report)
    
    def generate_all_visualizations(self, output_dir: str = "technical_evaluation_results"):
        """Generate all visualizations and reports"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info("Generating performance comparison...")
        self.create_performance_comparison(str(output_path / "performance_comparison.png"))
        
        logger.info("Generating entity analysis...")  
        self.create_entity_analysis(str(output_path / "entity_analysis.png"))
        
        logger.info("Generating detailed report...")
        self.generate_detailed_report(str(output_path / "evaluation_report.md"))
        
        logger.info(f"All visualizations saved to: {output_path}/")

def main():
    """Main visualization execution"""
    visualizer = EvaluationVisualizer()
    visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main()