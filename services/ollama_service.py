# import ollama

# # Script: interact_with_mistral.py
# # Description: Interacts with a local Mistral (or other) LLM via Ollama.

# # --- Configuration ---
# # Make sure this model is already pulled using 'ollama pull <MODEL_NAME>'
# MODEL_NAME = "llama3.2:1b"  # Use "llama3.2:1b" or "tinyllama" or "phi3" or "mistral" if you pulled those instead
# OLLAMA_HOST = "http://localhost:11434"  # Default Ollama API address

# # --- Initialize Ollama Client ---
# # It's good practice to explicitly set the host if you're not always on default
# client = ollama.Client(host=OLLAMA_HOST)

# # --- Function to get a single completion ---


# # --- Function for an interactive chat session ---
# def interactive_chat(model: str = MODEL_NAME):
#     """
#     Starts an interactive chat session with the specified Ollama model.
#     Type 'exit' or 'quit' to end the session.
#     """
#     print(f"\n--- Starting interactive chat with {model} ---")
#     print("Type your message and press Enter. Type 'exit' or 'quit' to end.")

#     messages = []

#     while True:
#         user_input = input("\nYou: ")
#         if user_input.lower() in ["exit", "quit"]:
#             print("Ending chat session.")
#             break

#         messages.append({"role": "user", "content": user_input})

#         try:
#             # Using ollama.chat for multi-turn conversation
#             stream_response = client.chat(model=model, messages=messages, stream=True)

#             # Print response token by token
#             print(f"{model}: ", end="", flush=True)
#             full_response_content = ""
#             for chunk in stream_response:
#                 content = chunk["message"]["content"]
#                 print(content, end="", flush=True)
#                 full_response_content += content
#             print()  # Newline after the full response

#             messages.append({"role": "assistant", "content": full_response_content})

#         except ollama.ResponseError as e:
#             print(f"Error during chat: {e}")
#             if e.status_code == 404:
#                 print(f"Model '{model}' not found. Did you run 'ollama pull {model}'?")
#             break
#         except Exception as e:
#             print(f"An unexpected error occurred during chat: {e}")
#             break


# # --- Main execution ---
# if __name__ == "__main__":
#     print("Welcome to the local LLM interaction script!")

#     # Example 1: Get a single completion
#     # get_completion("What is the capital of India?", MODEL_NAME)
#     # get_completion("Write a very short poem about the morning sun.", MODEL_NAME)

#     # Example 2: Start an interactive chat
#     interactive_chat(MODEL_NAME)
