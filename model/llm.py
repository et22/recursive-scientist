import openai
import base64
import re

class LLM():
    def __init__(self, model_name="deepseek-r1:32b", logging=None):
        self.logging = logging
        self.client = openai.Client(
            base_url="http://localhost:11434/v1",
            api_key="ollama"  # Authentication-free private access
        )
    
        # Initialize conversation history
        self.conversation_history = []
        self.model_name = model_name

    def chat_with_model(self, user_input):      
        if self.logging is not None:
            self.logging.info("input: " + user_input)
        # Append user message to history
        user_input = self.construct_content(user_input)
        self.conversation_history.append({"role": "user", "content": user_input})
    
        # Send conversation history to model
        response = self.client.chat.completions.create(
            model=self.model_name,  # Adjust model name as needed
            messages=self.conversation_history,
            temperature=0.6
        )
    
        # Extract assistant response
        assistant_response = response.choices[0].message.content
        assistant_response = re.sub(r'<think>.*?</think>', '', assistant_response, flags=re.DOTALL) # we remove COT from the history/return value
        
        # Append assistant response to history
        self.conversation_history.append({"role": "assistant", "content": assistant_response})
        if self.logging is not None:
            self.logging.info("response: " + assistant_response)
        return assistant_response
        
    def construct_content(self, user_input):
        return user_input

class VisionLLM(LLM):
    def __init__(self, model_name="llama3.2-vision", logging=None):
        super().__init__(
            model_name=model_name,
            logging=logging,
        )
        print("Initialized VisionLLM")

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def construct_content(self, user_input):
        text = user_input["text"]
        image_path = user_input["image_path"]
        
        base64_image = self.encode_image(image_path)
        content = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
        ]
        return content
        
class TextLLM(LLM):
    def __init__(self, model_name="deepseek-r1:32b", logging=None):
        super().__init__(
            model_name=model_name,
            logging=logging,
        )
        print("Initialized TextLLM")