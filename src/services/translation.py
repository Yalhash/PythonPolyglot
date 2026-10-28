"""Translation service using OpenAI or Google Translate API"""

from typing import Optional, Dict, Any
from openai import OpenAI
import requests
from functools import lru_cache


class TranslationService:
    """Handles text translation between languages"""
    
    def __init__(self, openai_api_key: Optional[str] = None, 
                 google_api_key: Optional[str] = None,
                 prefer_google: bool = False,
                 model: str = "gpt-4o-mini",
                 debug_logging: bool = False,
                 mock_mode: bool = False):
        """
        Initialize translation service
        
        Args:
            openai_api_key: OpenAI API key (for OpenAI translation)
            google_api_key: Google Translate API key (for Google translation)
            prefer_google: If True, prefer Google Translate over OpenAI
            model: GPT model to use for OpenAI translation (default: gpt-4o-mini)
            debug_logging: If True, print debug logs to terminal
            mock_mode: If True, return dummy text instead of making API calls
        """
        self.openai_client = None
        self.google_api_key = google_api_key
        self.prefer_google = prefer_google
        self.model = model
        self.debug_logging = debug_logging
        self.mock_mode = mock_mode
        
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Translation cache to reduce API calls
        self._cache: Dict[str, str] = {}
    
    def _get_cache_key(self, text: str, source_lang: str, target_lang: str) -> str:
        """Generate cache key for translation"""
        return f"{source_lang}:{target_lang}:{text}"
    
    def translate_with_openai(self, text: str, target_language: str, 
                              source_language: Optional[str] = None) -> str:
        """
        Translate text using OpenAI API
        
        Args:
            text: Text to translate
            target_language: Target language code (e.g., 'en', 'es', 'fr')
            source_language: Optional source language code
            
        Returns:
            Translated text
        """
        if not self.openai_client:
            raise ValueError("OpenAI API key not provided")
        
        if not text or not text.strip():
            return ""
        
        try:
            # Build prompt for translation
            source_hint = f" from {source_language}" if source_language else ""
            prompt = f"Translate the following text{source_hint} to {target_language}. Only return the translation, no explanations:\n\n{text}"
            
            if self.debug_logging:
                print(f"[TRANSLATION DEBUG] OpenAI translation request:")
                print(f"  Model: {self.model}")
                print(f"  Source: {source_language or 'auto'} -> Target: {target_language}")
                print(f"  Text: {text[:100]}{'...' if len(text) > 100 else ''}")
            
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional translator. Translate the given text accurately and naturally."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            translated = response.choices[0].message.content.strip()
            
            if self.debug_logging:
                print(f"[TRANSLATION DEBUG] OpenAI translation response:")
                print(f"  Translated: {translated[:100]}{'...' if len(translated) > 100 else ''}")
            
            return translated
        
        except Exception as e:
            print(f"OpenAI translation error: {e}")
            raise
    
    def translate_with_google(self, text: str, target_language: str,
                             source_language: Optional[str] = None) -> str:
        """
        Translate text using Google Translate API
        
        Args:
            text: Text to translate
            target_language: Target language code (e.g., 'en', 'es', 'fr')
            source_language: Optional source language code
            
        Returns:
            Translated text
        """
        if not self.google_api_key:
            raise ValueError("Google Translate API key not provided")
        
        if not text or not text.strip():
            return ""
        
        try:
            if self.debug_logging:
                print(f"[TRANSLATION DEBUG] Google translation request:")
                print(f"  Source: {source_language or 'auto'} -> Target: {target_language}")
                print(f"  Text: {text[:100]}{'...' if len(text) > 100 else ''}")
            
            # Google Translate API v2
            url = "https://translation.googleapis.com/language/translate/v2"
            params = {
                'key': self.google_api_key,
                'q': text,
                'target': target_language
            }
            if source_language:
                params['source'] = source_language
            
            response = requests.post(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            if 'data' in data and 'translations' in data['data']:
                translated = data['data']['translations'][0]['translatedText']
                
                if self.debug_logging:
                    print(f"[TRANSLATION DEBUG] Google translation response:")
                    print(f"  Translated: {translated[:100]}{'...' if len(translated) > 100 else ''}")
                
                return translated
            else:
                raise ValueError("Unexpected response format from Google Translate API")
        
        except Exception as e:
            print(f"Google translation error: {e}")
            raise
    
    def translate(self, text: str, target_language: str,
                  source_language: Optional[str] = None) -> str:
        """
        Translate text using available service
        
        Args:
            text: Text to translate
            target_language: Target language code
            source_language: Optional source language code
            
        Returns:
            Translated text
        """
        if not text or not text.strip():
            return ""
        
        # Mock mode: return dummy text without making API calls
        if self.mock_mode:
            if self.debug_logging:
                print(f"[TRANSLATION DEBUG] Mock mode enabled - returning dummy text")
            return "dummy text"
        
        # Check cache
        cache_key = self._get_cache_key(text, source_language or 'auto', target_language)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Try to translate
        translated = ""
        error = None
        
        # Determine which service to use
        use_google = self.prefer_google and self.google_api_key
        use_openai = self.openai_client is not None
        print("Use google:", use_google)
        
        if self.debug_logging:
            print(f"[TRANSLATION DEBUG] translate() called:")
            print(f"  Text: {text[:100]}{'...' if len(text) > 100 else ''}")
            print(f"  Source: {source_language or 'auto'} -> Target: {target_language}")
            print(f"  Using: {'Google' if use_google else 'OpenAI' if use_openai else 'None'}")
        
        if use_google:
            try:
                translated = self.translate_with_google(text, target_language, source_language)
            except Exception as e:
                error = e
                if self.debug_logging:
                    print(f"[TRANSLATION DEBUG] Google translation failed: {e}")
                # Fallback to OpenAI if available
                if use_openai:
                    try:
                        translated = self.translate_with_openai(text, target_language, source_language)
                    except Exception as e2:
                        error = e2
                        if self.debug_logging:
                            print(f"[TRANSLATION DEBUG] OpenAI fallback also failed: {e2}")
        
        elif use_openai:
            try:
                translated = self.translate_with_openai(text, target_language, source_language)
            except Exception as e:
                error = e
                if self.debug_logging:
                    print(f"[TRANSLATION DEBUG] OpenAI translation failed: {e}")
        
        else:
            raise ValueError("No translation service available. Provide OpenAI or Google API key.")
        
        if error and not translated:
            raise error
        
        # Cache the result
        if translated:
            self._cache[cache_key] = translated
            # Limit cache size
            if len(self._cache) > 1000:
                # Remove oldest entries (simple approach: clear half)
                keys_to_remove = list(self._cache.keys())[:500]
                for key in keys_to_remove:
                    del self._cache[key]
        
        return translated
    
    def clear_cache(self):
        """Clear translation cache"""
        self._cache.clear()

