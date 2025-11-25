#!/usr/bin/env python3
"""
CleaveRNA - RNA cleavage site prediction tool

This package provides tools for predicting RNA cleavage sites using machine learning
and structural analysis. It includes feature extraction, model training, and prediction
capabilities.

Modules:
    CleaveRNA: Main prediction and training module
    Feature: Feature extraction and processing module

Command-line tools:
    cleaverna: Main CleaveRNA analysis tool
    cleaverna-feature: Feature extraction tool
"""

__version__ = "1.0.0"
__author__ = "reytakop"
__email__ = "your.email@example.com"  # Replace with your actual email
__description__ = "RNA cleavage site prediction tool using machine learning and structural analysis"

# Import main classes and functions for easy access
try:
    # Import main functions that are needed for the entry points
    from .CleaveRNA import main as cleaverna_main
    from .Feature import main as feature_main
    
    # Other imports for API usage (optional)
    try:
        from .CleaveRNA import (
            ProgressTracker,
            predict_execution_time,
            format_time,
            create_cfg_file,
            train_and_save_svm,
            perform_cross_validation,
            predict_with_confidence,
            read_fasta_sequence,
            run_rnafold_and_get_structure,
            dotbracket_to_pairs,
        )
        
        from .Feature import (
            convert_t_to_u,
            run_rnaplfold,
            parse_rnaplfold_output,
            find_CS,
            prepare_sequences,
            write_queries_to_fasta,
            process_specific_query,
            construct_intarna_command,
            process_intarna_queries,
            merge_numerical_columns,
            post_process_features,
            merge_all_generated_files,
        )
    except ImportError:
        # If individual imports fail, that's okay for basic functionality
        pass
        
except ImportError as e:
    # Handle import errors gracefully for development
    print(f"Warning: Main modules could not be imported: {e}")
    cleaverna_main = None
    feature_main = None

# Package metadata
__all__ = [
    '__version__',
    '__author__',
    '__email__',
    '__description__',
    'cleaverna_main',
    'feature_main',
    'ProgressTracker',
    'predict_execution_time',
    'format_time',
    'create_cfg_file',
    'train_and_save_svm',
    'perform_cross_validation',
    'predict_with_confidence',
    'read_fasta_sequence',
    'run_rnafold_and_get_structure',
    'dotbracket_to_pairs',
    'convert_t_to_u',
    'run_rnaplfold',
    'parse_rnaplfold_output',
    'find_CS',
    'prepare_sequences',
    'write_queries_to_fasta',
    'process_specific_query',
    'construct_intarna_command',
    'process_intarna_queries',
    'merge_numerical_columns',
    'post_process_features',
    'merge_all_generated_files',
]

def get_version():
    """Return the version string."""
    return __version__

def get_info():
    """Return package information."""
    return {
        'name': 'CleaveRNA',
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'description': __description__,
    }

# Banner for command-line tools
BANNER = f"""
╔══════════════════════════════════════════════════════════════╗
║                        CleaveRNA v{__version__}                        ║
║            RNA Cleavage Site Prediction Tool                ║
║                                                              ║
║    🧬 Machine Learning + Structural Analysis 🧬             ║
╚══════════════════════════════════════════════════════════════╝
"""