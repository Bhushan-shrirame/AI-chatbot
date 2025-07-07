
## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- [pip](https://pip.pypa.io/en/stable/installation/)
- (Optional) [virtualenv](https://virtualenv.pypa.io/en/latest/) for isolated environments

### Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/Bhushan-shrirame/AI-chatbot.git
   cd AI-chatbot
   ```

2. **Create and activate a virtual environment (recommended):**
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **(Optional) Download or place any required model files or data in the `pdf/` directory.**
   - _Note: Large model files are not included in the repository. Download instructions or links should be provided separately if needed._

### Running the Chatbot

#### Command Line Interface

```sh
python main.py
```

#### Web Interface (if implemented)

```sh
python app.py
```
_Or specify the actual entry point for the web UI._

### Usage

- **CLI:** Type your questions or commands and receive responses in the terminal.
- **Web UI:** Open your browser and navigate to `http://localhost:5000` (or the specified port) to interact with the chatbot.

#### Example Conversation


User: Hello!
Bot: Hi there! How can I assist you today?
User: Can you recommend a good book?
Bot: Sure! What genre are you interested in?



## Customization

- **Add new intents or responses:** Edit the intent files or model training data in the `chatbot/` directory.
- **Integrate external APIs:** Extend the `conversation.py` or similar modules to fetch data from APIs (e.g., weather, news).
- **Change the model:** Swap out the NLP model in `nlp_model.py` for another pre-trained model as needed.

## Troubleshooting

- **Large files not included:** If you see errors about missing `.dylib` or model files, download them from the provided links or follow setup instructions.
- **Dependency issues:** Ensure you are using the correct Python version and have installed all dependencies.

## Contributing

Contributions are welcome!  
To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

Please follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/) in all interactions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions, suggestions, or support, please contact [Bhushan Shrirame](https://github.com/Bhushan-shrirame) or open an issue on GitHub.
