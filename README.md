# Dissertation Appendix: UK Cyber Fraud Assistant

## Overview

This appendix contains all the technical artefacts, training materials, and evaluation data referenced in the dissertation "Developing AI-based cyber fraud guidance systems specifically adapted for the UK context through parameter-efficient fine-tuning."

## Appendix Structure

### Appendix A: Data Collection and Processing Protocols
- `scraper.py` - Main web scraping implementation
- `site_scrapers.py` - Site-specific scraping configurations  
- Contains complete technical procedures for web scraping and content processing

### Appendix B: Prompt Engineering Templates  
- `additional_qa_pairs.json` -  gemini-generated Q&A pairs showing prompt engineering results
- `additional_qa_pairs_part2.json` - Extended prompt engineering examples
- `additional_qa_pairs_part3.json` - Final prompt engineering iterations
- ` gemini_generated_complete.json` - Complete set of AI-generated training pairs
- `actionfraud_qa_example.json` - Example of document-to-dialogue conversion
- Full document-to-dialogue conversion prompts and instructions

### Appendix C: Training Dataset Schema
- `111_master_fraud_qa_dataset.json` - Initial dataset (111 pairs)
- `278_master_fraud_qa_dataset.json` - Expanded dataset (278 pairs) 
- `1000_master_fraud_qa_dataset.json` - Final training dataset (1000 pairs)
- Complete data structure and format specifications

### Appendix D: Sample Training Data
- `sample_training_data.json` - Representative question-answer pairs (first 50 entries)
- Demonstrates data quality and format consistency

### Appendix E: Model Configuration Parameters
- `111_Unsloth_Fine_Tuning.ipynb` - Initial training configuration
- `278_Unsloth_Fine_Tuning.ipynb` - Intermediate training setup
- `278_Unsloth_Fine_Tuning_v2.ipynb` - Optimized training parameters
- `278_Unsloth_Fine_Tuning_v2.1.ipynb` - Final parameter tuning
- `1000_Unsloth_Fine_Tuning_with_perplexity.ipynb` - Large dataset training with metrics
- Complete LoRA and training hyperparameter specifications

### Appendix F: Training Performance Logs
- `generate_figure14_training_curves.py` - Training curve visualization
- `generate_perplexity_plot.py` - Perplexity analysis and plotting
- Detailed loss curves, perplexity metrics, and convergence analysis

### Appendix G: Evaluation Framework Rubrics
- `core_performance_analysis.py` - Manual evaluation analysis framework
- Complete assessment criteria for all four evaluation dimensions

### Appendix H: Technical Evaluation Implementation
- `complete_evaluation_data.py` - Complete evaluation data handling
- `standard_cosine_evaluation.py` - Cosine similarity implementation
- `verify_implementation.py` - Implementation verification scripts
- `regenerate_visualizations.py` - Evaluation visualization generation
- `TECHNICAL_EVALUATION_DOCUMENTATION.md` - Technical setup documentation
- Cosine similarity setup, embedding model configuration, and calculation methods

### Appendix I: Statistical Analysis Methods
- `domain_weighted_evaluation.py` - Statistical analysis implementation
- `evaluation_visualizer.py` - Statistical visualization scripts
- Complete formulae, confidence interval calculations, and significance testing procedures


This appendix provides complete transparency and reproducibility for all research findings presented in the dissertation.