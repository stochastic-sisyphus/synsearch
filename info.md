tree -I "__pycache__"
.
├── README.md
├── __init__.py
├── config
│   ├── config.yaml
│   └── visualization_config.yaml
├── data
│   ├── input
│   ├── output
│   └── processed
├── logs
│   └── pipeline.log
├── outputs
│   ├── checkpoints
│   └── figures
├── plan.md
├── requirements.txt
├── run.py
├── scripts
│   └── validate_dependencies.py
├── setup.py
├── src
│   ├── __init__.py
│   ├── cli.py
│   ├── cluster_manager.py
│   ├── clustering
│   │   ├── __init__.py
│   │   ├── attention_clustering.py
│   │   ├── cluster_explainer.py
│   │   ├── cluster_manager.py
│   │   ├── clustering_utils.py
│   │   ├── dynamic_cluster_manager.py
│   │   ├── dynamic_clusterer.py
│   │   ├── graph_clusterer.py
│   │   ├── hybrid_cluster_manager.py
│   │   └── streaming_manager.py
│   ├── dashboard
│   │   └── app.py
│   ├── data_exploration.py
│   ├── data_loader
│   │   └── flexible_loader.py
│   ├── data_loader.py
│   ├── data_preparation.py
│   ├── data_validator.py
│   ├── embedding_generator.py
│   ├── evaluation
│   │   ├── __init__.py
│   │   ├── cluster_evaluator.py
│   │   ├── eval_pipeline.py
│   │   ├── metrics.py
│   │   └── pipeline_evaluator.py
│   ├── main.py
│   ├── main_with_training.py
│   ├── personalization
│   │   └── user_config.py
│   ├── preprocessing
│   │   └── domain_agnostic_preprocessor.py
│   ├── preprocessor.py
│   ├── summarization
│   │   ├── __init__.py
│   │   ├── adaptive_style.py
│   │   ├── adaptive_summarizer.py
│   │   ├── enhanced_summarizer.py
│   │   ├── hybrid_summarizer.py
│   │   ├── model_trainer.py
│   │   └── summarizer.py
│   ├── synsearch.code-workspace
│   ├── utils
│   │   ├── __init__.py
│   │   ├── checkpoint_manager.py
│   │   ├── cluster_selector.py
│   │   ├── logging_config.py
│   │   ├── logging_utils.py
│   │   ├── metrics_calculator.py
│   │   ├── metrics_utils.py
│   │   ├── model_utils.py
│   │   └── style_selector.py
│   └── visualization
│       ├── __init__.py
│       ├── cluster_visualizer.py
│       └── embedding_visualizer.py
└── tests
    ├── __init__.py
    ├── test_data_loader.py
    ├── test_data_pipeline.py
    ├── test_data_validator.py
    ├── test_embedding_generator.py
    ├── test_embedding_visualizer.py
    ├── test_enhanced_pipeline.py
    ├── test_evaluation_metrics.py
    ├── test_integration.py
    └── test_summarizer.py

21 directories, 70 files