import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class PlayerPointsPredictor:
    def __init__(self):
        self.team_to_number = {
            'ATL': 1, 'BOS': 2, 'BRK': 3, 'CHI': 4, 'CHO': 5, 'CLE': 6, 'DAL': 7,
            'DEN': 8, 'DET': 9, 'GSW': 10, 'HOU': 11, 'IND': 12, 'LAC': 13, 'LAL': 14,
            'MEM': 15, 'MIA': 16, 'MIL': 17, 'MIN': 18, 'NOP': 19, 'NYK': 20, 'OKC': 21,
            'ORL': 22, 'PHI': 23, 'PHO': 24, 'POR': 25, 'SAC': 26, 'SAS': 27, 'TOR': 28,
            'UTA': 29, 'WAS': 30,
        }
        self.model = None
        self.scaler = StandardScaler()
        
    def fetch_player_data(self, player_name):
        try:
            from scraper import GameLogs
            return GameLogs(player_name).fetch_data()
        except ImportError:
            raise ImportError("No class")
    
    def clean_data(self, data):
        dat = data.copy()
        
        dat = dat.fillna(0)
        
        # Remove rows where 'GS' is 'Inactive' or 'Did Not Dress'
        if 'GS' in dat.columns:
            dat = dat[~dat['GS'].isin(['Inactive', 'Did Not Dress'])]
        
        # Drop unnecessary columns if they exist
        columns_to_drop = ['Rk', 'Unnamed: 5', 'Age', 'Tm', 'Unnamed: 7', 'MP', 'G', '+/-', 'GS', 'BLK', 'STL', 'ORB']
        for column in columns_to_drop:
            if column in dat.columns:
                dat = dat.drop(column, axis=1)
        
        # Rename 'Unnamed: 7' to 'W/L' if it exists
        if 'Unnamed: 7' in dat.columns:
            dat = dat.rename(columns={'Unnamed: 7': 'W/L'})
        
        # Add home/away indicator if 'Opp' exists
        if 'Opp' in dat.columns:
            dat['Home'] = ~dat['Opp'].astype(str).str.startswith('@')
            dat['Home'] = dat['Home'].astype(int)
            
            # Clean opponent name (remove @ if present)
            dat['Opp'] = dat['Opp'].astype(str).str.replace('@', '')
            
            # Convert opponent to numeric
            dat['Opp'] = dat['Opp'].map(self.team_to_number).fillna(0).astype(int)
        
        # Convert 'Date' to datetime then calculate rest days
        if 'Date' in dat.columns:
            dat['Date'] = pd.to_datetime(dat['Date'])
            dat['Rest'] = dat['Date'].diff().dt.days
            dat['Rest'] = dat['Rest'].fillna(0)
            
            # Add day of week feature (0-6 for Monday-Sunday)
            dat['DayOfWeek'] = dat['Date'].dt.dayofweek
        
        # Reset index
        dat.reset_index(drop=True, inplace=True)
        
        numeric_cols = ['FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'DRB', 'TRB', 'AST', 'TOV', 'PF', 'PTS', 'GmSc']
        for col in numeric_cols:
            if col in dat.columns:
                dat[col] = pd.to_numeric(dat[col], errors='coerce')
        
        for window in [3, 5, 10]:
            for col in ['PTS', 'FG%', '3P%', 'FT%', 'AST', 'TRB', 'TOV']:
                if col in dat.columns:
                    dat[f'{col}_last_{window}_avg'] = dat[col].rolling(window=window).mean()
                    
                    # Add rolling standard deviation to capture consistency
                    dat[f'{col}_last_{window}_std'] = dat[col].rolling(window=window).std()
        
        if 'PTS' in dat.columns:
            dat['PTS_trend'] = dat['PTS'].diff().rolling(window=3).mean()
            
        if 'MP' in dat.columns:
            dat['MP'] = pd.to_numeric(dat['MP'], errors='coerce')
            dat['MP_last_5_avg'] = dat['MP'].rolling(window=5).mean()
            
        if 'Opp' in dat.columns and 'PTS' in dat.columns:
            # Create opponent-specific average points
            opp_avg = dat.groupby('Opp')['PTS'].mean().to_dict()
            dat['Avg_PTS_vs_Opp'] = dat['Opp'].map(opp_avg)

            home_avg = dat[dat['Home'] == 1]['PTS'].mean()
            away_avg = dat[dat['Home'] == 0]['PTS'].mean()
            dat['Avg_PTS_HomeAway'] = dat['Home'].map({1: home_avg, 0: away_avg})
        
        for col in dat.columns:
            if '_avg' in col and dat[col].isna().any():
                base_col = col.split('_last_')[0]
                if base_col in dat.columns:
                    dat[col] = dat[col].fillna(dat[base_col].mean())
        
        dat = dat.fillna(0)
        
        return dat
    
    def train_model(self, data):
        """Train a Random Forest model to predict points"""
        # Define features with expanded feature set
        features = [
            'Rest', 'Opp', 'Home', 'DayOfWeek',
            'FG%', '3P%', 'FT%', 
            'PTS_last_3_avg', 'PTS_last_5_avg', 'PTS_last_10_avg',
            'PTS_last_3_std', 'PTS_last_5_std', 'PTS_last_10_std',
            'FG%_last_3_avg', 'FG%_last_5_avg', 'FG%_last_10_avg',
            '3P%_last_3_avg', '3P%_last_5_avg', '3P%_last_10_avg',
            'FT%_last_3_avg', 'FT%_last_5_avg', 'FT%_last_10_avg',
            'AST_last_5_avg', 'TRB_last_5_avg', 'TOV_last_5_avg',
            'PTS_trend', 'MP_last_5_avg', 'Avg_PTS_vs_Opp', 'Avg_PTS_HomeAway'
        ]
        
        # Make sure all features exist
        available_features = [f for f in features if f in data.columns]
        X = data[available_features]
        
        if 'PTS' not in data.columns:
            raise ValueError("Target variable 'PTS' not found in data")
        
        y = data['PTS']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        # Create Random Forest model
        rf_model = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(
            estimator=rf_model,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            scoring='neg_mean_absolute_error',
            verbose=0
        )
        
        # Train the model with the best hyperparameters
        grid_search.fit(X_scaled, y)
        self.model = grid_search.best_estimator_
        
        # Save the feature list used for prediction
        self.features = available_features
        
        # Calculate prediction errors for confidence estimation
        self.cv_predictions = cross_val_predict(self.model, X_scaled, y, cv=5)
        self.prediction_errors = np.abs(self.cv_predictions - y)
        self.mean_error = np.mean(self.prediction_errors)
        self.std_error = np.std(self.prediction_errors)
        
        # Calculate feature importances
        self.feature_importance = dict(zip(available_features, self.model.feature_importances_))
        
        return self.model
    
    def calculate_confidence(self, prediction, player_data):
        """
        Calculate confidence level for the prediction
        
        Returns a value between 0 and 100, where higher values indicate higher confidence
        """
        if not hasattr(self, 'mean_error') or not hasattr(self, 'std_error'):
            return None
            
        # Calculate consistency of player's scoring
        if 'PTS' in player_data.columns:
            player_std = player_data['PTS'].std()
            player_mean = player_data['PTS'].mean()
            player_cv = player_std / player_mean if player_mean > 0 else 1
            
            # More consistent players (lower CV) get higher confidence
            consistency_factor = max(0, 1 - player_cv)
        else:
            consistency_factor = 0.5  # Default if we can't calculate
            
        # Calculate how close the prediction is to the player's average
        if 'PTS' in player_data.columns:
            recent_avg = player_data['PTS'].iloc[-10:].mean()
            deviation = abs(prediction - recent_avg) / recent_avg if recent_avg > 0 else 1
            deviation_factor = max(0, 1 - deviation)
        else:
            deviation_factor = 0.5  # Default if we can't calculate
            
        error_factor = max(0, 1 - (self.mean_error / (player_mean if player_mean > 0 else self.mean_error)))
        
        confidence = (0.4 * consistency_factor + 0.3 * deviation_factor + 0.3 * error_factor) * 100
        
        return max(0, min(100, confidence))
    
    def predict_next_game(self, player_name, opponent, days_rest, is_home):
        """
        Predict points for a player's next game
        
        Parameters:
        -----------
        player_name : str
            Name of the player
        opponent : str
            Three-letter code of the opponent team (e.g., 'LAL')
        days_rest : int
            Number of days rest before the game
        is_home : bool
            Whether the game is at home (True) or away (False)
        
        Returns:
        --------
        dict
            Dictionary with prediction and player stats
        """
        # Fetch and prepare player data
        raw_data = self.fetch_player_data(player_name)
        player_data = self.clean_data(raw_data)
        
        # Train model if not already trained
        if self.model is None:
            self.train_model(player_data)
        
        # Convert opponent to numeric
        opponent_num = self.team_to_number.get(opponent, 0)
        
        # Get the most recent values for our features
        feature_values = {}
        feature_values['Rest'] = days_rest
        feature_values['Opp'] = opponent_num
        feature_values['Home'] = 1 if is_home else 0
        
        # Add day of week (assuming prediction is for today)
        import datetime
        feature_values['DayOfWeek'] = datetime.datetime.now().weekday()
        
        # Add basic shooting percentages
        if 'FG%' in player_data.columns:
            feature_values['FG%'] = player_data['FG%'].iloc[-1]
        if '3P%' in player_data.columns:
            feature_values['3P%'] = player_data['3P%'].iloc[-1]
        if 'FT%' in player_data.columns:
            feature_values['FT%'] = player_data['FT%'].iloc[-1]
        
        # Add rolling averages and standard deviations
        for window in [3, 5, 10]:
            for col in ['PTS', 'FG%', '3P%', 'FT%', 'AST', 'TRB', 'TOV']:
                if col in player_data.columns:
                    feature_values[f'{col}_last_{window}_avg'] = player_data[col].iloc[-window:].mean()
                    feature_values[f'{col}_last_{window}_std'] = player_data[col].iloc[-window:].std()
        if 'PTS' in player_data.columns:
            recent_pts = player_data['PTS'].iloc[-3:]
            if len(recent_pts) >= 2:
                feature_values['PTS_trend'] = recent_pts.diff().mean()
            else:
                feature_values['PTS_trend'] = 0
                
        if 'MP' in player_data.columns:
            feature_values['MP_last_5_avg'] = player_data['MP'].iloc[-5:].mean()

        if 'Opp' in player_data.columns and 'PTS' in player_data.columns:
            # Points vs this specific opponent
            pts_vs_opp = player_data[player_data['Opp'] == opponent_num]['PTS']
            if not pts_vs_opp.empty:
                feature_values['Avg_PTS_vs_Opp'] = pts_vs_opp.mean()
            else:
                feature_values['Avg_PTS_vs_Opp'] = player_data['PTS'].mean()
                
        
            if is_home:
                home_pts = player_data[player_data['Home'] == 1]['PTS']
                feature_values['Avg_PTS_HomeAway'] = home_pts.mean() if not home_pts.empty else player_data['PTS'].mean()
            else:
                away_pts = player_data[player_data['Home'] == 0]['PTS']
                feature_values['Avg_PTS_HomeAway'] = away_pts.mean() if not away_pts.empty else player_data['PTS'].mean()
        
    
        for feature in self.features:
            if feature not in feature_values:
                feature_values[feature] = 0
        
    
        features_for_prediction = [feature_values[feature] for feature in self.features]
        
        # Scaling
        features_scaled = self.scaler.transform([features_for_prediction])
        
        prediction = self.model.predict(features_scaled)[0]
        
        confidence = self.calculate_confidence(prediction, player_data)
        
        season_stats = {
            'Season PPG': player_data['PTS'].mean() if 'PTS' in player_data.columns else 0,
            'Last 5 games PPG': player_data['PTS'].iloc[-5:].mean() if 'PTS' in player_data.columns else 0,
            'Last 10 games PPG': player_data['PTS'].iloc[-10:].mean() if 'PTS' in player_data.columns else 0
        }
        
        if hasattr(self, 'feature_importance'):
            top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            important_features = {name: round(importance, 3) for name, importance in top_features}
        else:
            important_features = {}
        
        return {
            'player': player_name,
            'opponent': opponent,
            'home_game': is_home,
            'days_rest': days_rest,
            'predicted_points': round(prediction, 1),
            'confidence': round(confidence, 1) if confidence is not None else None,
            'season_stats': season_stats,
            'important_features': important_features
        }

def main():
    predictor = PlayerPointsPredictor()
    
    player_name = input("Enter player name (e.g., Stephen Curry): ")
    opponent = input("Enter opponent team (3-letter code, e.g., LAL): ")
    days_rest = int(input("Enter days of rest: "))
    is_home = input("Is this a home game? (y/n): ").lower() == 'y'
    
    try:
        result = predictor.predict_next_game(player_name, opponent, days_rest, is_home)
        
        print("\n========== PREDICTION RESULT ==========")
        print(f"Player: {result['player']}")
        print(f"Opponent: {result['opponent']}")
        print(f"Location: {'Home' if result['home_game'] else 'Away'}")
        print(f"Days Rest: {result['days_rest']}")
        print(f"\nPREDICTED POINTS: {result['predicted_points']}")
        
        if result['confidence'] is not None:
            confidence_level = "Low" if result['confidence'] < 40 else "Medium" if result['confidence'] < 70 else "High"
            print(f"Confidence: {result['confidence']}% ({confidence_level})")
        
        print("\nSeason Stats:")
        print(f"  Season Average: {result['season_stats']['Season PPG']:.1f} PPG")
        print(f"  Last 5 Games: {result['season_stats']['Last 5 games PPG']:.1f} PPG")
        print(f"  Last 10 Games: {result['season_stats']['Last 10 games PPG']:.1f} PPG")
        
        if result['important_features']:
            print("\nKey Factors:")
            for feature, importance in result['important_features'].items():
                print(f"  {feature}: {importance}")
                
        print("======================================")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have the necessary data fetching functionality implemented.")

if __name__ == "__main__":
    main()