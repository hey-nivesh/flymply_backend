import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

class GemmaService:
    def __init__(self):
        api_key = os.getenv('GOOGLE_API_KEY', 'YOUR_API_KEY')
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')  # Or gemma if available
    
    def analyze_turbulence(self, weather_data, ml_probability):
        """
        Use Gemma to reason about turbulence prediction
        Returns: adjusted probability and reasoning
        """
        prompt = f"""You are an aviation meteorology expert. Analyze this turbulence prediction:

Weather Data:
- Wind Speed: {weather_data['wind_speed']} m/s
- Temperature: {weather_data['temperature']}°C
- Pressure: {weather_data['pressure']} hPa
- Humidity: {weather_data['humidity']}%

ML Model Prediction: {ml_probability * 100:.1f}% turbulence risk

Task: 
1. Assess if this probability seems reasonable given the weather
2. Suggest a small adjustment (+/- 10% max) if warranted
3. Provide brief reasoning (1-2 sentences)

Respond in format:
ADJUSTMENT: +5 (or -3, or 0)
REASONING: Brief explanation"""
        try:
            response = self.model.generate_content(prompt)
            text = response.text
            
            # Parse response
            adjustment = 0
            reasoning = "AI reasoning unavailable"
            
            if "ADJUSTMENT:" in text:
                adj_line = [l for l in text.split('\n') if 'ADJUSTMENT:' in l][0]
                adj_str = adj_line.split(':')[1].strip()
                # Handle both +5 and -3 formats
                adjustment = int(adj_str.replace('+', ''))
            
            if "REASONING:" in text:
                reasoning = text.split('REASONING:')[1].strip()
            
            # Apply adjustment (limit to ±10%)
            adjusted_prob = ml_probability + (adjustment / 100)
            adjusted_prob = max(0, min(1, adjusted_prob))
            
            return {
                'adjusted_probability': adjusted_prob,
                'reasoning': reasoning,
                'raw_adjustment': adjustment
            }
        except Exception as e:
            print(f"Gemma error: {e}")
            return {
                'adjusted_probability': ml_probability,
                'reasoning': "AI analysis unavailable",
                'raw_adjustment': 0
            }

