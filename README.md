1. telegram_chatbot.ipynb is used to generate and pickle the a. Intent classifier b. sentence tokenizer to get Tech stack (tags) from input user text if the intent is get StackOverflow answers c. Tech stack (tag) wise encoded threads
2. bot.py is the main program for the API calls. It uses dialogue_manager.py to get the best matching threads based on cosine similarities on the encodings.
