"""AI response generation service using OpenAI GPT API"""

from typing import Optional, List, Dict, Any
from openai import OpenAI
from collections import deque


class ResponseGenerationService:
    """Generates contextually appropriate responses using OpenAI GPT"""
    
    def __init__(self, api_key: str, model: str = "gpt-4.1-mini", max_context: int = 10,
                 debug_logging: bool = False, mock_mode: bool = False):
        """
        Initialize response generation service
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use (default: gpt-4)
            max_context: Maximum number of conversation turns to keep in context
            debug_logging: If True, print debug logs to terminal
            mock_mode: If True, return dummy text instead of making API calls
        """
        if not api_key and not mock_mode:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.model = model
        self.max_context = max_context
        self.debug_logging = debug_logging
        self.mock_mode = mock_mode
        self.conversation_history: deque = deque(maxlen=max_context * 2)  # Store pairs of messages
    
    def add_to_history(self, original_text: str, translated_text: str, 
                      source_language: str, target_language: str):
        """
        Add a conversation turn to history
        
        Args:
            original_text: Original language text
            translated_text: Translated text
            source_language: Source language code
            target_language: Target language code
        """
        self.conversation_history.append({
            'original': original_text,
            'translated': translated_text,
            'source_lang': source_language,
            'target_lang': target_language
        })
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
    
    def generate_response(self, original_text: str, translated_text: str,
                          source_language: str, target_language: str,
                          context: Optional[str] = None) -> Dict[str, str]:
        """
        Generate appropriate responses in both languages
        
        Args:
            original_text: Original language text from the conversation
            translated_text: Translated text
            source_language: Source language code
            target_language: Target language code
            context: Optional additional context
            
        Returns:
            Dictionary with 'original_response' and 'translated_response' keys
        """
        # Mock mode: return dummy text without making API calls
        if self.mock_mode:
            if self.debug_logging:
                print(f"[RESPONSE DEBUG] Mock mode enabled - returning dummy text")
            return {
                'original_response': 'dummy text',
                'translated_response': 'dummy text'
            }
        
        try:
            if self.debug_logging:
                print(f"[RESPONSE DEBUG] generate_response() called:")
                print(f"  Model: {self.model}")
                print(f"  Original text: {original_text[:100]}{'...' if len(original_text) > 100 else ''}")
                print(f"  Translated text: {translated_text[:100]}{'...' if len(translated_text) > 100 else ''}")
                print(f"  Source lang: {source_language}, Target lang: {target_language}")
                print(f"  History size: {len(self.conversation_history)}")
            
            # Build conversation context
            context_messages = []
            
            # Add system message
            system_prompt = f"""You are a helpful interview assistant. Based on the conversation context, generate a brief, natural, and appropriate response.

The conversation is happening between languages: {source_language} (original) and {target_language} (user's language).

Generate responses that are:
- Brief and concise (1-2 sentences)
- Professional and appropriate for an interview context
- Natural and conversational
- Contextually relevant to what was just said

Provide the response in BOTH languages:
1. First in {source_language} (the original language)
2. Then in {target_language} (the user's language)

Format your response as:
ORIGINAL: [response in {source_language}]
TRANSLATED: [response in {target_language}]"""
            
            context_messages.append({
                "role": "system",
                "content": system_prompt
            })
            
            # Add conversation history
            if self.conversation_history:
                history_text = "Recent conversation:\n"
                for turn in list(self.conversation_history)[-5:]:  # Last 5 turns
                    history_text += f"- {turn['translated']}\n"
                context_messages.append({
                    "role": "user",
                    "content": f"Conversation context:\n{history_text}"
                })
            
            # Add current message
            current_message = f"Just said (in {source_language}): {original_text}\nTranslated (to {target_language}): {translated_text}"
            if context:
                current_message += f"\nAdditional context: {context}"
            
            current_message += f"\n\nGenerate an appropriate brief response in both {source_language} and {target_language}."
            
            context_messages.append({
                "role": "user",
                "content": current_message
            })
            
            if self.debug_logging:
                print(f"[RESPONSE DEBUG] Sending request to OpenAI API...")
                print(f"  Messages count: {len(context_messages)}")
            
            # Call GPT API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=context_messages,
                temperature=0.7,
                max_tokens=300
            )
            
            response_text = response.choices[0].message.content.strip()
            
            if self.debug_logging:
                print(f"[RESPONSE DEBUG] Received response from API:")
                print(f"  Raw response: {response_text[:200]}{'...' if len(response_text) > 200 else ''}")
            
            # Parse response
            original_response = ""
            translated_response = ""
            
            # Try to extract responses from formatted output
            if "ORIGINAL:" in response_text and "TRANSLATED:" in response_text:
                parts = response_text.split("TRANSLATED:")
                if len(parts) == 2:
                    original_part = parts[0].replace("ORIGINAL:", "").strip()
                    translated_part = parts[1].strip()
                    original_response = original_part
                    translated_response = translated_part
                    if self.debug_logging:
                        print(f"[RESPONSE DEBUG] Parsed ORIGINAL/TRANSLATED format")
            else:
                # Fallback: try to split by newlines or use the whole response
                lines = response_text.split('\n')
                if len(lines) >= 2:
                    original_response = lines[0].strip()
                    translated_response = lines[1].strip()
                    if self.debug_logging:
                        print(f"[RESPONSE DEBUG] Parsed newline-separated format")
                else:
                    # If we can't parse, use the response for both (not ideal but better than nothing)
                    original_response = response_text
                    translated_response = response_text
                    if self.debug_logging:
                        print(f"[RESPONSE DEBUG] Warning: Could not parse format, using full response for both")
            
            # If parsing failed, try a simpler approach: use GPT to generate separately
            if not original_response or not translated_response:
                if self.debug_logging:
                    print(f"[RESPONSE DEBUG] Parsing failed, generating separate responses...")
                # Generate original language response
                original_response = self._generate_single_response(
                    original_text, source_language, source_language
                )
                # Generate translated response
                translated_response = self._generate_single_response(
                    translated_text, target_language, target_language
                )
            
            if self.debug_logging:
                print(f"[RESPONSE DEBUG] Final parsed responses:")
                print(f"  Original response: {original_response[:100]}{'...' if len(original_response) > 100 else ''}")
                print(f"  Translated response: {translated_response[:100]}{'...' if len(translated_response) > 100 else ''}")
            
            return {
                'original_response': original_response,
                'translated_response': translated_response
            }
        
        except Exception as e:
            if self.debug_logging:
                print(f"[RESPONSE DEBUG] Error in generate_response(): {e}")
                import traceback
                traceback.print_exc()
            print(f"Response generation error: {e}")
            # Return fallback responses
            return {
                'original_response': "I understand.",
                'translated_response': "I understand."
            }
    
    def _generate_single_response(self, text: str, language: str, response_language: str) -> str:
        """Generate a single response in one language"""
        try:
            prompt = f"Based on this statement: '{text}', generate a brief, natural response in {response_language} (1-2 sentences)."
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"You are a helpful assistant. Generate brief, natural responses in {response_language}."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Single response generation error: {e}")
            return "I understand."

