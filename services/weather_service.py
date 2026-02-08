import requests
import random
from datetime import datetime

class WeatherService:
    def __init__(self):
        # Using OpenWeatherMap as example (can be mocked)
        self.api_key = "AIzaSyBrPn3-N8TV-jCPFEGVsvkkLqZuK6FPo4g"  # Or use mock data
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
    
    def get_weather_data(self, lat=40.7128, lon=-74.0060):
        """
        Fetch weather data. Falls back to mock data if API fails.
        Returns: dict with wind_speed, temperature, pressure, humidity
        """
        try:
            # Attempt real API call (optional)
            # params = {"lat": lat, "lon": lon, "appid": self.api_key}
            # response = requests.get(self.base_url, params=params, timeout=5)
            # data = response.json()
            
            # For hackathon: Use simulated data
            return self._generate_mock_data()
        except:
            return self._generate_mock_data()
    
    def _generate_mock_data(self):
        """Generate realistic mock weather data"""
        return {
            "wind_speed": round(random.uniform(5, 45), 2),  # m/s
            "temperature": round(random.uniform(-40, 30), 2),  # Celsius
            "pressure": round(random.uniform(950, 1050), 2),  # hPa
            "humidity": round(random.uniform(20, 90), 2),  # %
            "timestamp": datetime.now().isoformat()
        }

