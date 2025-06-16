# Text Generator Language Model using LSTM
This project implements a text generator using an LSTM-based language model in PyTorch. The model is trained on a combined corpus of "Alice in Wonderland" and "Macbeth" from the NLTK Gutenberg dataset, enabling it to generate coherent text sequences. Key features include data preprocessing with spaCy, model training with PyTorch, and evaluation using perplexity, along with visualizations of training loss and test perplexity.

## Project Overview
The goal of this project is to build a language model that can generate text by learning patterns from classic literature. The model uses an LSTM architecture to predict the next word in a sequence, trained on a dataset of 244,747 characters. After preprocessing, the model achieves a test perplexity of 71.96 with a 1-layer LSTM, demonstrating its ability to generate meaningful text sequences.

## Key Features
- Dataset:
    - Combined "Alice in Wonderland" and "Macbeth" texts from NLTK Gutenberg (244,747 characters, 56,703 tokens, scaled to 100,020 tokens).
    - Vocabulary size of 5,000 tokens.
    - Split into 90,000 training sequences and 10,000 test sequences (sequence length of 20).
- Model Architecture:
    - LSTM-based language model implemented in PyTorch.
    - Compared 1 and 2-layer LSTM configurations to optimize performance.
- Training and Evaluation:
    - Achieved a test perplexity of 71.96 with the 1-layer LSTM model.
    - Visualized training loss and test perplexity over epochs using Matplotlib.
- Text Generation:
    - Generates 50-word text sequences using top-k sampling (k=5).

## Results
- Performance:
   - Final test perplexity of 71.96 with the 1-layer LSTM model, outperforming deeper architectures (2 and 3 layers).
   - Final training loss of 3.62 after multiple epochs.
- Visualizations:
   - raining loss and test perplexity plots for 1 and 2-layer models.

## Future Improvements
- Larger Dataset: Incorporate additional texts from the Gutenberg corpus to improve model generalization.
- Advanced Architectures: Experiment with Transformers or GPT-based models for better text generation quality.
- Hyperparameter Tuning: Optimize learning rate, batch size, and top-k sampling parameters to reduce perplexity further.
- GPU Support: Extend the project to leverage GPU acceleration for faster training.
