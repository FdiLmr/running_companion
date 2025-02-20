import numpy as np
import pandas as pd
import numpy.ma as ma
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from datetime import date
import logging
import os
from sql_methods import write_db_replace, read_db
from typing import Tuple, Optional, Dict
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def prepare_features(features_blocks: pd.DataFrame, athlete_id: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare features for model training."""
    try:
        if features_blocks.empty:
            raise ValueError("No features data available")
        
        # Filter for specific athlete if provided
        if athlete_id:
            athlete_data = features_blocks[features_blocks['athlete_id'] == int(athlete_id)]
            if athlete_data.empty:
                logger.warning(f"No features data available for athlete {athlete_id}, using all data instead")
                athlete_data = features_blocks  # Fall back to using all data
            features_blocks = athlete_data
            
        # Get feature columns (excluding athlete_id, block_id, and target variables)
        feature_cols = features_blocks.columns[2:-2]
        
        # Convert features to numeric, replacing non-numeric values with NaN
        X = features_blocks[feature_cols].apply(pd.to_numeric, errors='coerce')
        
        # Add missing columns if they don't exist
        if 'r_proportion_alpine_ski' not in X.columns:
            X['r_proportion_alpine_ski'] = 0
        if 'r_proportion_crossfit' not in X.columns:
            X['r_proportion_crossfit'] = 0
            
        # Fill NaN values with 0 or column mean
        for col in X.columns:
            col_mean = X[col].mean()
            if pd.isna(col_mean):  # If entire column is NaN
                X[col] = X[col].fillna(0)
            else:
                X[col] = X[col].fillna(col_mean)
        
        # Convert to numpy array
        X = X.to_numpy()
        
        # Convert target variables to numeric and handle NaN
        y_absolute = pd.to_numeric(features_blocks.iloc[:, -2], errors='coerce').fillna(0)
        y_change = pd.to_numeric(features_blocks.iloc[:, -1], errors='coerce').fillna(0)
        
        return X, y_absolute, y_change
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        raise

def train_random_forest(X: np.ndarray, y: np.ndarray, n_estimators: int = 50) -> Tuple[RandomForestRegressor, float, np.ndarray, np.ndarray]:
    """Train a random forest model and return model, score, and predictions."""
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        regressor = RandomForestRegressor(n_estimators=n_estimators)
        regressor.fit(X_train, y_train)
        
        y_pred = regressor.predict(X_test)
        model_score = r2_score(y_test, y_pred)
        
        return regressor, model_score, y_test, y_pred
    except Exception as e:
        logger.error(f"Error training random forest: {e}")
        raise

def calculate_feature_importance(regressor: RandomForestRegressor, features: list, y_name: str, 
                               model_score: float, athlete_id: Optional[str] = None) -> pd.DataFrame:
    """Calculate and format feature importance metrics."""
    try:
        importances = regressor.feature_importances_
        indices = np.argsort(importances)
        
        model_outputs = pd.DataFrame()
        model_run_date = str(date.today())
        
        for index in indices:
            row_data = {
                'y_name': y_name,
                'feature_name': features[index],
                'importance': importances[index],
                'model_score': model_score,
                'model_run_date': model_run_date
            }
            if athlete_id:
                row_data['athlete_id'] = athlete_id
                
            model_outputs = pd.concat([model_outputs, pd.DataFrame([row_data])], ignore_index=True)
            
        return model_outputs
    except Exception as e:
        logger.error(f"Error calculating feature importance: {e}")
        raise

def generate_shap_plots(regressor: RandomForestRegressor, X: np.ndarray, feature_names: list, 
                       plot_type: str, athlete_id: Optional[str] = None) -> None:
    """Generate and save SHAP plots."""
    try:
        import shap
        plt.clf()  # Clear current figure
        
        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(X)
        
        # Create static/shap_plots directory if it doesn't exist
        os.makedirs('second_part/static/shap_plots', exist_ok=True)
        
        # Generate and save summary plot
        shap.summary_plot(shap_values, pd.DataFrame(X, columns=feature_names), show=False)
        plt.tight_layout()
        summary_filename = f'second_part/static/shap_plots/{plot_type}_summary_{athlete_id}.png' if athlete_id else f'shap_plots/{plot_type}_summary.png'
        plt.savefig(summary_filename)
        plt.close()
        
        # Generate and save bar plot
        plt.clf()
        shap.summary_plot(shap_values, pd.DataFrame(X, columns=feature_names), plot_type="bar", show=False)
        plt.tight_layout()
        bar_filename = f'second_part/static/shap_plots/{plot_type}_bar_{athlete_id}.png' if athlete_id else f'shap_plots/{plot_type}_bar.png'
        plt.savefig(bar_filename)
        plt.close()
        
    except ImportError:
        logger.warning("SHAP package not installed. Skipping visualization.")
    except Exception as e:
        logger.error(f"Error generating SHAP plots: {e}")
        raise

def train_model(athlete_id: Optional[str] = None) -> Dict:
    """Train model for a specific athlete or all athletes."""
    try:
        # Read features data
        features_blocks = read_db('features_blocks')
        if features_blocks.empty:
            raise ValueError("No features data available in database")
            
        # Prepare features and targets
        X, y_absolute, y_change = prepare_features(features_blocks, athlete_id)
        feature_names = list(features_blocks.columns.values[2:-2])
        
        model_outputs = pd.DataFrame()
        results = {}
        
        # Train and evaluate model for absolute VDOT
        regressor_abs, score_abs, _, _ = train_random_forest(X, y_absolute)
        model_outputs = pd.concat([
            model_outputs,
            calculate_feature_importance(regressor_abs, feature_names, 'absolute_vdot', score_abs, athlete_id)
        ], ignore_index=True)
        results['absolute_vdot_score'] = score_abs
        
        # Train and evaluate model for VDOT change
        regressor_change, score_change, _, _ = train_random_forest(X, y_change)
        model_outputs = pd.concat([
            model_outputs,
            calculate_feature_importance(regressor_change, feature_names, 'vdot_change', score_change, athlete_id)
        ], ignore_index=True)
        results['vdot_change_score'] = score_change
        
        # Save results to database
        write_db_replace(model_outputs, 'model_outputs')
        
        # Generate SHAP plots
        try:
            generate_shap_plots(regressor_abs, X, feature_names, 'absolute_vdot', athlete_id)
            generate_shap_plots(regressor_change, X, feature_names, 'vdot_change', athlete_id)
        except Exception as e:
            logger.warning(f"Could not generate SHAP plots: {e}")
            results['shap_error'] = str(e)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in model training pipeline: {e}")
        raise

if __name__ == "__main__":
    train_model()