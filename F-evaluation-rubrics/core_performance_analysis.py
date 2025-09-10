
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

class CorePerformanceAnalyzer:
    """
    Core performance analysis for manual grading evaluation data.
    Focuses on essential metrics and clear visualizations.
    """
    
    def __init__(self):
        self.dimensions = [
            'UK_Contact_Accuracy',
            'Conversational_Quality_Empathy', 
            'Practical_Utility_Quality',
            'Professional_Boundary_Adherence'
        ]
        
        self.dimension_labels = [
            'UK Contact\nAccuracy',
            'Conversational\nQuality/Empathy',
            'Practical Utility\nQuality', 
            'Professional\nBoundary Adherence'
        ]
        
        # Raw data extracted from manual grading PDF
        self.grading_data = self._load_grading_data()
        self.results_df = None
        self._setup_styling()
    
    def _setup_styling(self):
        """Setup matplotlib styling for clean, academic visualizations"""
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False,
        })
        
        # Color palette for baseline vs fine-tuned
        self.colors = {
            'baseline': '#FF6B6B',      # Red-ish
            'finetuned': '#4ECDC4',     # Teal
            'improvement': '#45B7D1',    # Blue
            'positive': '#96CEB4',       # Green
            'negative': '#FFEAA7'        # Yellow
        }
    
    def _load_grading_data(self):
        """
        Load the manual grading data extracted from PDF.
        Returns structured data for all 50 questions across 4 dimensions.
        """
        # Complete manual grading data extracted from PDF
        grading_data = {
            # Q1-Q10
            'Q1': {'UK_Contact_Accuracy': [5, 5], 'Conversational_Quality_Empathy': [2, 2], 'Practical_Utility_Quality': [3, 4], 'Professional_Boundary_Adherence': [4, 4]},
            'Q2': {'UK_Contact_Accuracy': [4, 3], 'Conversational_Quality_Empathy': [3, 5], 'Practical_Utility_Quality': [3, 4], 'Professional_Boundary_Adherence': [5, 5]},
            'Q3': {'UK_Contact_Accuracy': [4, 5], 'Conversational_Quality_Empathy': [4, 4], 'Practical_Utility_Quality': [3, 5], 'Professional_Boundary_Adherence': [4, 5]},
            'Q4': {'UK_Contact_Accuracy': [4, 5], 'Conversational_Quality_Empathy': [3, 3], 'Practical_Utility_Quality': [3, 5], 'Professional_Boundary_Adherence': [4, 5]},
            'Q5': {'UK_Contact_Accuracy': [4, 4], 'Conversational_Quality_Empathy': [3, 5], 'Practical_Utility_Quality': [3, 4], 'Professional_Boundary_Adherence': [3, 5]},
            'Q6': {'UK_Contact_Accuracy': [5, 5], 'Conversational_Quality_Empathy': [4, 4], 'Practical_Utility_Quality': [3, 4], 'Professional_Boundary_Adherence': [4, 5]},
            'Q7': {'UK_Contact_Accuracy': [4, 5], 'Conversational_Quality_Empathy': [4, 3], 'Practical_Utility_Quality': [3, 5], 'Professional_Boundary_Adherence': [4, 5]},
            'Q8': {'UK_Contact_Accuracy': [3, 5], 'Conversational_Quality_Empathy': [4, 5], 'Practical_Utility_Quality': [3, 5], 'Professional_Boundary_Adherence': [5, 5]},
            'Q9': {'UK_Contact_Accuracy': [5, 5], 'Conversational_Quality_Empathy': [3, 4], 'Practical_Utility_Quality': [3, 5], 'Professional_Boundary_Adherence': [3, 5]},
            'Q10': {'UK_Contact_Accuracy': [3, 5], 'Conversational_Quality_Empathy': [4, 4], 'Practical_Utility_Quality': [3, 5], 'Professional_Boundary_Adherence': [3, 5]},
            
            # Q11-Q20
            'Q11': {'UK_Contact_Accuracy': [3, 5], 'Conversational_Quality_Empathy': [4, 5], 'Practical_Utility_Quality': [3, 5], 'Professional_Boundary_Adherence': [4, 5]},
            'Q12': {'UK_Contact_Accuracy': [4, 5], 'Conversational_Quality_Empathy': [4, 5], 'Practical_Utility_Quality': [3, 5], 'Professional_Boundary_Adherence': [4, 5]},
            'Q13': {'UK_Contact_Accuracy': [5, 5], 'Conversational_Quality_Empathy': [4, 5], 'Practical_Utility_Quality': [3, 5], 'Professional_Boundary_Adherence': [5, 5]},
            'Q14': {'UK_Contact_Accuracy': [4, 5], 'Conversational_Quality_Empathy': [4, 4], 'Practical_Utility_Quality': [4, 5], 'Professional_Boundary_Adherence': [5, 5]},
            'Q15': {'UK_Contact_Accuracy': [4, 5], 'Conversational_Quality_Empathy': [4, 5], 'Practical_Utility_Quality': [4, 3], 'Professional_Boundary_Adherence': [4, 4]},
            'Q16': {'UK_Contact_Accuracy': [4, 5], 'Conversational_Quality_Empathy': [5, 5], 'Practical_Utility_Quality': [4, 5], 'Professional_Boundary_Adherence': [4, 4]},
            'Q17': {'UK_Contact_Accuracy': [4, 5], 'Conversational_Quality_Empathy': [3, 5], 'Practical_Utility_Quality': [4, 5], 'Professional_Boundary_Adherence': [4, 5]},
            'Q18': {'UK_Contact_Accuracy': [4, 5], 'Conversational_Quality_Empathy': [4, 5], 'Practical_Utility_Quality': [3, 5], 'Professional_Boundary_Adherence': [3, 5]},
            'Q19': {'UK_Contact_Accuracy': [2, 5], 'Conversational_Quality_Empathy': [3, 5], 'Practical_Utility_Quality': [3, 5], 'Professional_Boundary_Adherence': [3, 5]}, 
            'Q20': {'UK_Contact_Accuracy': [3, 5], 'Conversational_Quality_Empathy': [3, 5], 'Practical_Utility_Quality': [3, 5], 'Professional_Boundary_Adherence': [3, 5]},
            
            # Q21-Q30
            'Q21': {'UK_Contact_Accuracy': [5, 5], 'Conversational_Quality_Empathy': [5, 5], 'Practical_Utility_Quality': [3, 5], 'Professional_Boundary_Adherence': [5, 5]},
            'Q22': {'UK_Contact_Accuracy': [3, 4], 'Conversational_Quality_Empathy': [4, 4], 'Practical_Utility_Quality': [2, 5], 'Professional_Boundary_Adherence': [4, 4]},
            'Q23': {'UK_Contact_Accuracy': [5, 5], 'Conversational_Quality_Empathy': [4, 5], 'Practical_Utility_Quality': [4, 5], 'Professional_Boundary_Adherence': [5, 5]},
            'Q24': {'UK_Contact_Accuracy': [4, 4], 'Conversational_Quality_Empathy': [3, 3], 'Practical_Utility_Quality': [3, 4], 'Professional_Boundary_Adherence': [4, 4]},
            'Q25': {'UK_Contact_Accuracy': [3, 5], 'Conversational_Quality_Empathy': [2, 5], 'Practical_Utility_Quality': [2, 5], 'Professional_Boundary_Adherence': [4, 5]},
            'Q26': {'UK_Contact_Accuracy': [2, 5], 'Conversational_Quality_Empathy': [3, 5], 'Practical_Utility_Quality': [3, 5], 'Professional_Boundary_Adherence': [4, 5]},
            'Q27': {'UK_Contact_Accuracy': [5, 5], 'Conversational_Quality_Empathy': [2, 4], 'Practical_Utility_Quality': [3, 5], 'Professional_Boundary_Adherence': [3, 5]},
            'Q28': {'UK_Contact_Accuracy': [5, 4], 'Conversational_Quality_Empathy': [2, 2], 'Practical_Utility_Quality': [4, 4], 'Professional_Boundary_Adherence': [4, 4]},
            'Q29': {'UK_Contact_Accuracy': [4, 5], 'Conversational_Quality_Empathy': [3, 5], 'Practical_Utility_Quality': [3, 5], 'Professional_Boundary_Adherence': [3, 5]},
            'Q30': {'UK_Contact_Accuracy': [4, 5], 'Conversational_Quality_Empathy': [4, 5], 'Practical_Utility_Quality': [3, 5], 'Professional_Boundary_Adherence': [4, 5]},
            
            # Q31-Q40
            'Q31': {'UK_Contact_Accuracy': [4, 5], 'Conversational_Quality_Empathy': [4, 4], 'Practical_Utility_Quality': [3, 5], 'Professional_Boundary_Adherence': [4, 5]},
            'Q32': {'UK_Contact_Accuracy': [3, 5], 'Conversational_Quality_Empathy': [3, 5], 'Practical_Utility_Quality': [3, 5], 'Professional_Boundary_Adherence': [4, 5]},
            'Q33': {'UK_Contact_Accuracy': [3, 5], 'Conversational_Quality_Empathy': [4, 4], 'Practical_Utility_Quality': [3, 5], 'Professional_Boundary_Adherence': [4, 5]},
            'Q34': {'UK_Contact_Accuracy': [2, 5], 'Conversational_Quality_Empathy': [3, 5], 'Practical_Utility_Quality': [3, 5], 'Professional_Boundary_Adherence': [2, 5]},
            'Q35': {'UK_Contact_Accuracy': [2, 5], 'Conversational_Quality_Empathy': [3, 4], 'Practical_Utility_Quality': [3, 4], 'Professional_Boundary_Adherence': [3, 5]},
            'Q36': {'UK_Contact_Accuracy': [3, 5], 'Conversational_Quality_Empathy': [3, 3], 'Practical_Utility_Quality': [4, 5], 'Professional_Boundary_Adherence': [5, 5]},
            'Q37': {'UK_Contact_Accuracy': [4, 5], 'Conversational_Quality_Empathy': [4, 4], 'Practical_Utility_Quality': [4, 5], 'Professional_Boundary_Adherence': [4, 5]},
            'Q38': {'UK_Contact_Accuracy': [3, 5], 'Conversational_Quality_Empathy': [4, 4], 'Practical_Utility_Quality': [3, 4], 'Professional_Boundary_Adherence': [4, 4]},
            'Q39': {'UK_Contact_Accuracy': [4, 4], 'Conversational_Quality_Empathy': [3, 3], 'Practical_Utility_Quality': [3, 4], 'Professional_Boundary_Adherence': [4, 4]},
            'Q40': {'UK_Contact_Accuracy': [4, 4], 'Conversational_Quality_Empathy': [3, 3], 'Practical_Utility_Quality': [3, 2], 'Professional_Boundary_Adherence': [3, 3]},
            
            # Q41-Q50
            'Q41': {'UK_Contact_Accuracy': [5, 5], 'Conversational_Quality_Empathy': [4, 4], 'Practical_Utility_Quality': [4, 5], 'Professional_Boundary_Adherence': [5, 5]},
            'Q42': {'UK_Contact_Accuracy': [5, 5], 'Conversational_Quality_Empathy': [4, 4], 'Practical_Utility_Quality': [4, 5], 'Professional_Boundary_Adherence': [4, 4]},
            'Q43': {'UK_Contact_Accuracy': [5, 5], 'Conversational_Quality_Empathy': [3, 4], 'Practical_Utility_Quality': [4, 5], 'Professional_Boundary_Adherence': [5, 5]},
            'Q44': {'UK_Contact_Accuracy': [3, 5], 'Conversational_Quality_Empathy': [4, 4], 'Practical_Utility_Quality': [3, 4], 'Professional_Boundary_Adherence': [4, 4]},
            'Q45': {'UK_Contact_Accuracy': [3, 4], 'Conversational_Quality_Empathy': [3, 4], 'Practical_Utility_Quality': [4, 5], 'Professional_Boundary_Adherence': [4, 5]},
            'Q46': {'UK_Contact_Accuracy': [4, 5], 'Conversational_Quality_Empathy': [4, 3], 'Practical_Utility_Quality': [3, 5], 'Professional_Boundary_Adherence': [3, 5]},
            'Q47': {'UK_Contact_Accuracy': [5, 5], 'Conversational_Quality_Empathy': [3, 4], 'Practical_Utility_Quality': [3, 5], 'Professional_Boundary_Adherence': [4, 5]},
            'Q48': {'UK_Contact_Accuracy': [5, 5], 'Conversational_Quality_Empathy': [3, 3], 'Practical_Utility_Quality': [3, 5], 'Professional_Boundary_Adherence': [3, 5]},
            'Q49': {'UK_Contact_Accuracy': [5, 5], 'Conversational_Quality_Empathy': [3, 3], 'Practical_Utility_Quality': [5, 5], 'Professional_Boundary_Adherence': [4, 4]},
            'Q50': {'UK_Contact_Accuracy': [3, 3], 'Conversational_Quality_Empathy': [3, 4], 'Practical_Utility_Quality': [3, 4], 'Professional_Boundary_Adherence': [3, 3]}
        }
        
        return grading_data
    
    def process_data(self):
        """Process raw grading data into structured DataFrame"""
        processed_data = []
        
        for question_id, scores in self.grading_data.items():
            question_data = {'Question_ID': question_id}
            
            for dim in self.dimensions:
                baseline_score = scores[dim][0]
                finetuned_score = scores[dim][1]
                
                # Handle missing data (marked as 0 in Q19)
                if baseline_score == 0 and finetuned_score == 0:
                    baseline_score = np.nan
                    finetuned_score = np.nan
                
                question_data[f'{dim}_Baseline'] = baseline_score
                question_data[f'{dim}_Finetuned'] = finetuned_score
                question_data[f'{dim}_Improvement'] = finetuned_score - baseline_score if not np.isnan(baseline_score) else np.nan
                question_data[f'{dim}_Improvement_Pct'] = ((finetuned_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else np.nan
        
            processed_data.append(question_data)
        
        self.results_df = pd.DataFrame(processed_data)
        return self.results_df
    
    def calculate_core_metrics(self):
        """Calculate essential performance metrics"""
        if self.results_df is None:
            self.process_data()
        
        metrics = {}
        
        for i, dim in enumerate(self.dimensions):
            baseline_col = f'{dim}_Baseline'
            finetuned_col = f'{dim}_Finetuned'
            improvement_col = f'{dim}_Improvement'
            
            # Core metrics
            baseline_mean = self.results_df[baseline_col].mean()
            finetuned_mean = self.results_df[finetuned_col].mean()
            improvement_mean = self.results_df[improvement_col].mean()
            
            # Improvement analysis
            improvements = self.results_df[improvement_col].dropna()
            positive_improvements = (improvements > 0).sum()
            total_valid = len(improvements)
            improvement_consistency = (positive_improvements / total_valid * 100) if total_valid > 0 else 0
            
            metrics[dim] = {
                'dimension_label': self.dimension_labels[i],
                'baseline_mean': baseline_mean,
                'baseline_std': self.results_df[baseline_col].std(),
                'finetuned_mean': finetuned_mean,
                'finetuned_std': self.results_df[finetuned_col].std(),
                'mean_improvement': improvement_mean,
                'improvement_std': self.results_df[improvement_col].std(),
                'improvement_consistency_pct': improvement_consistency,
                'questions_improved': positive_improvements,
                'questions_declined': (improvements < 0).sum(),
                'questions_unchanged': (improvements == 0).sum(),
                'total_questions': total_valid
            }
        
        return metrics
    
    def generate_core_visualizations(self):
        """Generate essential visualizations for core performance analysis"""
        if self.results_df is None:
            self.process_data()
        
        metrics = self.calculate_core_metrics()
        output_dir = Path("visualization_results")
        output_dir.mkdir(exist_ok=True)
        
        # 1. Overall Performance Comparison
        self._plot_performance_overview(metrics, output_dir)
        
        # 2. Improvement Analysis
        self._plot_improvement_analysis(metrics, output_dir)
        
        # 3. Dimension Breakdown
        self._plot_dimension_breakdown(output_dir)
        
        print(f"Core visualizations saved to: {output_dir}/")
    
    def _plot_performance_overview(self, metrics, output_dir):
        """Create comprehensive performance overview visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Manual Grading Analysis - Core Performance Overview', fontsize=16, fontweight='bold')
        
        dimensions = list(metrics.keys())
        dim_labels = [metrics[dim]['dimension_label'] for dim in dimensions]
        
        # Plot 1: Mean Score Comparison
        baseline_means = [metrics[dim]['baseline_mean'] for dim in dimensions]
        finetuned_means = [metrics[dim]['finetuned_mean'] for dim in dimensions]
        
        x = np.arange(len(dimensions))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, baseline_means, width, label='Baseline', 
                       color=self.colors['baseline'], alpha=0.8, edgecolor='black', linewidth=0.8)
        bars2 = ax1.bar(x + width/2, finetuned_means, width, label='Fine-tuned', 
                       color=self.colors['finetuned'], alpha=0.8, edgecolor='black', linewidth=0.8)
        
        ax1.set_xlabel('Evaluation Dimensions')
        ax1.set_ylabel('Average Score (1-5)')
        ax1.set_title('Mean Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(dim_labels, rotation=45, ha='right')
        ax1.legend()
        ax1.set_ylim(0, 5.5)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Improvement Consistency
        improvement_consistency = [metrics[dim]['improvement_consistency_pct'] for dim in dimensions]
        
        bars = ax2.bar(range(len(dimensions)), improvement_consistency, 
                      color=self.colors['improvement'], alpha=0.8, edgecolor='black', linewidth=0.8)
        ax2.set_xlabel('Evaluation Dimensions')
        ax2.set_ylabel('Questions Improved (%)')
        ax2.set_title('Improvement Consistency Rate')
        ax2.set_xticks(range(len(dimensions)))
        ax2.set_xticklabels(dim_labels, rotation=45, ha='right')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        
        # Plot 3: Mean Improvement Values
        mean_improvements = [metrics[dim]['mean_improvement'] for dim in dimensions]
        improvement_stds = [metrics[dim]['improvement_std'] for dim in dimensions]
        
        bars = ax3.bar(range(len(dimensions)), mean_improvements, 
                      yerr=improvement_stds, capsize=5,
                      color=self.colors['improvement'], alpha=0.8, edgecolor='black', linewidth=0.8)
        ax3.set_xlabel('Evaluation Dimensions')
        ax3.set_ylabel('Mean Improvement Score')
        ax3.set_title('Average Improvement per Dimension')
        ax3.set_xticks(range(len(dimensions)))
        ax3.set_xticklabels(dim_labels, rotation=45, ha='right')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Color bars based on positive/negative improvement
        for i, bar in enumerate(bars):
            if mean_improvements[i] > 0:
                bar.set_color(self.colors['positive'])
            else:
                bar.set_color(self.colors['negative'])
        
        # Plot 4: Sample Size Information
        sample_sizes = [metrics[dim]['total_questions'] for dim in dimensions]
        questions_improved = [metrics[dim]['questions_improved'] for dim in dimensions]
        questions_declined = [metrics[dim]['questions_declined'] for dim in dimensions]
        questions_unchanged = [metrics[dim]['questions_unchanged'] for dim in dimensions]
        
        ax4.bar(range(len(dimensions)), questions_improved, 
               label='Improved', color=self.colors['positive'], alpha=0.8)
        ax4.bar(range(len(dimensions)), questions_unchanged, 
               bottom=questions_improved, label='Unchanged', color='gray', alpha=0.6)
        ax4.bar(range(len(dimensions)), questions_declined, 
               bottom=np.array(questions_improved) + np.array(questions_unchanged),
               label='Declined', color=self.colors['negative'], alpha=0.8)
        
        ax4.set_xlabel('Evaluation Dimensions')
        ax4.set_ylabel('Number of Questions')
        ax4.set_title('Question-Level Improvement Distribution')
        ax4.set_xticks(range(len(dimensions)))
        ax4.set_xticklabels(dim_labels, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'core_performance_overview.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'core_performance_overview.pdf', bbox_inches='tight')
        plt.close()
    
    def _plot_improvement_analysis(self, metrics, output_dir):
        """Create detailed improvement analysis visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Improvement Analysis - Manual Grading Results', fontsize=16, fontweight='bold')
        
        dimensions = list(metrics.keys())
        dim_labels = [metrics[dim]['dimension_label'] for dim in dimensions]
        
        # Plot 1: Improvement Distribution by Dimension
        improvement_data = []
        dimension_labels = []
        
        for dim in dimensions:
            improvement_col = f'{dim}_Improvement'
            improvements = self.results_df[improvement_col].dropna()
            improvement_data.extend(improvements.tolist())
            dimension_labels.extend([metrics[dim]['dimension_label']] * len(improvements))
        
        # Create DataFrame for seaborn
        plot_df = pd.DataFrame({
            'Improvement': improvement_data,
            'Dimension': dimension_labels
        })
        
        sns.boxplot(data=plot_df, x='Dimension', y='Improvement', ax=ax1)
        ax1.set_title('Improvement Score Distribution by Dimension')
        ax1.set_xlabel('Evaluation Dimensions')
        ax1.set_ylabel('Improvement Score')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No Improvement')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.legend()
        
        # Plot 2: Overall Improvement Summary
        total_questions = sum(metrics[dim]['total_questions'] for dim in dimensions)
        total_improved = sum(metrics[dim]['questions_improved'] for dim in dimensions)
        total_declined = sum(metrics[dim]['questions_declined'] for dim in dimensions)
        total_unchanged = sum(metrics[dim]['questions_unchanged'] for dim in dimensions)
        
        categories = ['Improved', 'Unchanged', 'Declined']
        values = [total_improved, total_unchanged, total_declined]
        colors = [self.colors['positive'], 'gray', self.colors['negative']]
        
        wedges, texts, autotexts = ax2.pie(values, labels=categories, colors=colors, 
                                          autopct='%1.1f%%', startangle=90)
        ax2.set_title(f'Overall Question Improvement Distribution\n(Total: {total_questions} questions)')
        
        # Add count labels
        for i, (wedge, autotext) in enumerate(zip(wedges, autotexts)):
            autotext.set_text(f'{values[i]}\n({values[i]/total_questions*100:.1f}%)')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'improvement_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'improvement_analysis.pdf', bbox_inches='tight')
        plt.close()
    
    def _plot_dimension_breakdown(self, output_dir):
        """Create detailed dimension-by-dimension breakdown"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        fig.suptitle('Dimension-by-Dimension Performance Breakdown', fontsize=16, fontweight='bold')
        
        for i, dim in enumerate(self.dimensions):
            baseline_col = f'{dim}_Baseline'
            finetuned_col = f'{dim}_Finetuned'
            
            baseline_scores = self.results_df[baseline_col].dropna()
            finetuned_scores = self.results_df[finetuned_col].dropna()
            
            # Create histogram comparison
            ax = axes[i]
            
            bins = np.arange(0.5, 6.5, 1)
            ax.hist(baseline_scores, bins=bins, alpha=0.7, label='Baseline', 
                   color=self.colors['baseline'], edgecolor='black', linewidth=0.5)
            ax.hist(finetuned_scores, bins=bins, alpha=0.7, label='Fine-tuned', 
                   color=self.colors['finetuned'], edgecolor='black', linewidth=0.5)
            
            ax.set_title(f'{self.dimension_labels[i]}')
            ax.set_xlabel('Score (1-5)')
            ax.set_ylabel('Number of Questions')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_xticks([1, 2, 3, 4, 5])
            
            # Add mean lines
            ax.axvline(baseline_scores.mean(), color=self.colors['baseline'], 
                      linestyle='--', linewidth=2, alpha=0.8, 
                      label=f'Baseline Mean: {baseline_scores.mean():.1f}')
            ax.axvline(finetuned_scores.mean(), color=self.colors['finetuned'], 
                      linestyle='--', linewidth=2, alpha=0.8,
                      label=f'Fine-tuned Mean: {finetuned_scores.mean():.1f}')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'dimension_breakdown.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'dimension_breakdown.pdf', bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self):
        """Generate comprehensive text summary report"""
        if self.results_df is None:
            self.process_data()
        
        metrics = self.calculate_core_metrics()
        
        report = []
        report.append("="*80)
        report.append("MANUAL GRADING ANALYSIS - CORE PERFORMANCE REPORT")
        report.append("="*80)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Questions Analyzed: {len(self.results_df)}")
        report.append("")
        
        report.append("DIMENSION-WISE PERFORMANCE SUMMARY:")
        report.append("-" * 50)
        
        for dim, data in metrics.items():
            report.append(f"\n{data['dimension_label'].replace(chr(10), ' ')}")
            report.append(f"  Baseline Mean: {data['baseline_mean']:.2f} ± {data['baseline_std']:.2f}")
            report.append(f"  Fine-tuned Mean: {data['finetuned_mean']:.2f} ± {data['finetuned_std']:.2f}")
            report.append(f"  Mean Improvement: {data['mean_improvement']:.2f}")
            report.append(f"  Improvement Consistency: {data['improvement_consistency_pct']:.1f}%")
            report.append(f"  Questions: {data['questions_improved']} improved, {data['questions_declined']} declined, {data['questions_unchanged']} unchanged")
        
        # Overall summary
        total_questions = sum(data['total_questions'] for data in metrics.values())
        total_improved = sum(data['questions_improved'] for data in metrics.values())
        overall_improvement_rate = (total_improved / total_questions * 100) if total_questions > 0 else 0
        
        report.append(f"\nOVERALL SUMMARY:")
        report.append("-" * 20)
        report.append(f"Total Evaluations: {total_questions}")
        report.append(f"Total Improvements: {total_improved} ({overall_improvement_rate:.1f}%)")
        report.append(f"Average Baseline Score: {np.mean([data['baseline_mean'] for data in metrics.values()]):.2f}")
        report.append(f"Average Fine-tuned Score: {np.mean([data['finetuned_mean'] for data in metrics.values()]):.2f}")
        report.append(f"Overall Improvement: {np.mean([data['mean_improvement'] for data in metrics.values()]):.2f}")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)
    
    def run_complete_analysis(self):
        """Run complete core performance analysis"""
        print("Starting Manual Grading Core Performance Analysis...")
        print("="*60)
        
        # Process data
        self.process_data()
        print(f"✓ Processed data for {len(self.results_df)} questions")
        
        # Generate visualizations
        self.generate_core_visualizations()
        print("✓ Generated core visualizations")
        
        # Generate summary report
        summary = self.generate_summary_report()
        
        # Save summary to file
        output_dir = Path("visualization_results")
        with open(output_dir / "performance_summary.txt", 'w') as f:
            f.write(summary)
        
        # Print summary
        print("\n" + summary)
        
        return summary

def main():
    """Main execution function"""
    analyzer = CorePerformanceAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()