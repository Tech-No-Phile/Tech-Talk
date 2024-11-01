# Tech-Talk

Welcome to Tech-Talk, a chatbot built to enhance conversations and streamline support tasks. This project combines machine learning, natural language processing, and a bit of flair to bring interactive engagement to life!

## Features

- **Intent Classification**: Understands user intents from various phrases.
- **Response Generation**: Delivers accurate and helpful responses based on trained data.
- **Expandable**: Easily add more intents and responses to refine chatbot behavior.

## Getting Started

### Prerequisites

1. **Python 3.8+**
2. **Virtual Environment**: To keep dependencies tidy, activate a virtual environment.

### Setup Instructions

1. Clone the repository and navigate into it:
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/tech-talk-chatbot.git
   cd tech-talk-chatbot
   ```

2. Activate the virtual environment:
   ```bash
   python -m venv chatbot-env
   chatbot-env\Scripts\activate  # For Windows
   source chatbot-env/bin/activate  # For macOS/Linux
   ```

3. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK resources**:
   - Make sure to download the `punkt` and `wordnet` packages:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   ```

5. Run the chatbot:
   ```bash
   python bot.py
   ```

## Training

The `intents.json` file contains the structure of intents and responses. Use it to add or update patterns and responses.

To retrain the model, update `intents.json` and then run:

```bash
python train.py
```

## Troubleshooting

If you run into persistent issues with NLTK resources, make sure the packages are installed correctly within the virtual environment. Clear old cache or lock files if needed.

## Contributing

Feel free to fork this repository, open issues, or submit PRs for new features or fixes!

