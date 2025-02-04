# Shakespearean Text Generator

This project implements a text generation model that imitates Shakespeare's writing style using n-gram language models (bigrams, trigrams, and quadgrams).

## Overview

The text generator uses the NLTK Gutenberg corpus to access Shakespeare's works and implements three different n-gram models:
- Bigram model (2-gram)
- Trigram model (3-gram)
- Quadgram model (4-gram)

## Requirements

- Python 3.7+
- NLTK library
- Required NLTK data (Gutenberg corpus)

## Installation

1. Clone this repository
2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Download required NLTK data:

3. The script will automatically download required NLTK data on first run

## Usage

Run the main script:

```bash
python homework.py
```

To run the tests:

```bash
python -m unittest test_shakespeare_generator.py -v
```

The script will:
1. Load Shakespeare's texts from NLTK Gutenberg corpus
2. Build n-gram models
3. Generate sample texts using different n-gram sizes
4. Display statistics and probability distributions

## Implementation Details

- Text Preprocessing: Converts text to lowercase and removes punctuation
- N-gram Building: Creates dictionaries of n-gram counts and probabilities
- Text Generation: Uses weighted random sampling based on n-gram probabilities
- Fallback Mechanism: Handles unseen n-grams with common words

## Model Architecture

The implementation uses nested defaultdict structures to store:
- Token counts following each n-gram
- Probability distributions for next tokens
- Separate models for bigrams, trigrams, and quadgrams

## Output Examples

The generator produces three types of text:
1. Bigram-based (less coherent but faster)
2. Trigram-based (balance of coherence and variety)
3. Quadgram-based (more coherent but may repeat longer phrases)
