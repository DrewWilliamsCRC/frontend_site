#!/usr/bin/env python3
"""
Download essential NLTK data packages
This script downloads common NLTK data packages needed for NLP tasks.
"""

import nltk # type: ignore
import os
import sys

print("Downloading essential NLTK data packages...")

# Create directory for NLTK data if it doesn't exist
nltk_data_dir = os.path.expanduser("~/nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)

# Download essential NLTK packages
try:
    # Sentiment analysis
    nltk.download('vader_lexicon')
    
    # Tokenization
    nltk.download('punkt')
    
    # POS tagging
    nltk.download('averaged_perceptron_tagger')
    
    # Stopwords
    nltk.download('stopwords')
    
    # WordNet for lemmatization
    nltk.download('wordnet')
    
    print(f"All NLTK packages downloaded successfully to {nltk_data_dir}")
except Exception as e:
    print(f"Error downloading NLTK data: {e}", file=sys.stderr)
    # Exit with a success code even if downloads fail, to not block the build
    sys.exit(0) 