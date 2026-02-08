class FeatureEngineer:
    def __init__(self):
        self.previous_weather = None
    
    def extract_features(self, weather_data):
        """
        Extract turbulence-related features from weather data
        Returns: dict of engineered features
        """
        features = {
            'wind_speed': weather_data['wind_speed'],
            'wind_change': 0,
            'temp_gradient': 0,
            'pressure_drop': 0
        }
        
        if self.previous_weather:
            # Calculate changes
            features['wind_change'] = abs(
                weather_data['wind_speed'] - self.previous_weather['wind_speed']
            )
            features['temp_gradient'] = abs(
                weather_data['temperature'] - self.previous_weather['temperature']
            )
            features['pressure_drop'] = (
                self.previous_weather['pressure'] - weather_data['pressure']
            )
        
        self.previous_weather = weather_data.copy()
        return features

