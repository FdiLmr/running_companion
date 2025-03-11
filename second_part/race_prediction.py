import math
import logging
import datetime
from typing import Dict, List, Tuple, Optional
from sqlalchemy import desc
import numpy as np

from models import MetadataPB, RacePrediction
from sql_methods import db, read_db
from running_functions import calculate_vdot

logger = logging.getLogger(__name__)

# Common race distances in meters
RACE_DISTANCES = {
    "400m": 400,
    "800m": 800,
    "1km": 1000,
    "1 Mile": 1609.34,
    "3km": 3000,
    "2 Mile": 3218.69,
    "5km": 5000,
    "10km": 10000,
    "15km": 15000,
    "10 Mile": 16093.4,
    "Half Marathon": 21097.5,
    "25km": 25000,
    "30km": 30000,
    "Marathon": 42195,
    "50km": 50000,
    "50 Mile": 80467.2,
    "100km": 100000
}

# Default Riegel exponent if we can't calculate a personalized one
DEFAULT_RIEGEL_EXPONENT = 1.06

def calculate_riegel_exponent(time1: float, dist1: float, time2: float, dist2: float) -> float:
    """
    Calculate personalized Riegel exponent using two race performances
    
    Args:
        time1: Time in seconds for first distance
        dist1: First distance in meters
        time2: Time in seconds for second distance
        dist2: Second distance in meters
        
    Returns:
        Calculated exponent value
    """
    try:
        if time1 <= 0 or time2 <= 0 or dist1 <= 0 or dist2 <= 0:
            logger.warning("Invalid input values for Riegel exponent calculation")
            return DEFAULT_RIEGEL_EXPONENT
            
        # Ensure dist1 is shorter than dist2
        if dist1 > dist2:
            dist1, dist2 = dist2, dist1
            time1, time2 = time2, time1
            
        exponent = math.log(time2/time1) / math.log(dist2/dist1)
        
        # Validate the calculated exponent - typical range is 1.05 to 1.15
        if exponent < 0.95 or exponent > 1.20:
            logger.warning(f"Calculated exponent {exponent} is outside normal range, using default")
            return DEFAULT_RIEGEL_EXPONENT
            
        return exponent
        
    except (ValueError, ZeroDivisionError) as e:
        logger.error(f"Error calculating Riegel exponent: {e}")
        return DEFAULT_RIEGEL_EXPONENT

def predict_race_time(base_distance: float, base_time: float, target_distance: float, exponent: float = DEFAULT_RIEGEL_EXPONENT) -> float:
    """
    Predict race time using Riegel's formula
    
    Args:
        base_distance: Distance in meters of the known performance
        base_time: Time in seconds of the known performance
        target_distance: Distance in meters for the prediction
        exponent: Riegel exponent, default is 1.06
        
    Returns:
        Predicted time in seconds
    """
    if base_distance <= 0 or base_time <= 0 or target_distance <= 0:
        logger.warning("Invalid input values for race time prediction")
        return 0
        
    predicted_time = base_time * (target_distance / base_distance) ** exponent
    return predicted_time

def get_prediction_ranges(base_distance: float, base_time: float, target_distance: float, exponent: float) -> Dict[str, float]:
    """
    Generate optimistic, realistic, and conservative predictions
    
    Args:
        base_distance: Distance in meters of the known performance
        base_time: Time in seconds of the known performance
        target_distance: Distance in meters for the prediction
        exponent: Calculated Riegel exponent for the athlete
        
    Returns:
        Dictionary with optimistic, realistic, and conservative predictions
    """
    # Adjust exponent for different prediction ranges
    optimistic_exponent = max(exponent - 0.05, 1.06)  # More optimistic (smaller exponent)
    realistic_exponent = exponent  # Personalized exponent
    conservative_exponent = min(exponent + 0.05, 1.22)  # More conservative (larger exponent)
    
    return {
        "optimistic": predict_race_time(base_distance, base_time, target_distance, optimistic_exponent),
        "realistic": predict_race_time(base_distance, base_time, target_distance, realistic_exponent),
        "conservative": predict_race_time(base_distance, base_time, target_distance, conservative_exponent)
    }

def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable format (HH:MM:SS or MM:SS)
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}:{minutes:02d}:{secs:02d}"

def calculate_athlete_predictions(athlete_id: str) -> Optional[Dict]:
    """
    Calculate race predictions for an athlete based on their best performances
    
    Args:
        athlete_id: Athlete's ID
        
    Returns:
        Dictionary with prediction data or None if insufficient data
    """
    # Get athlete's 5K and 10K personal bests
    try:
        # Instead of a SQL query, we'll read the entire metadata_pbs table and filter in Python
        pbs_df = read_db('metadata_pbs')
        
        # Filter for this athlete's 5K and 10K PRs
        if not pbs_df.empty:
            # Filter for this athlete's data
            athlete_pbs = pbs_df[pbs_df['athlete_id'] == athlete_id]
            
            # Get all 5K performances and sort by elapsed_time to find the fastest
            pb_5k_all = athlete_pbs[athlete_pbs['distance_category'] == '5K'].sort_values('elapsed_time')
            # Get all 10K performances and sort by elapsed_time to find the fastest
            pb_10k_all = athlete_pbs[athlete_pbs['distance_category'] == '10K'].sort_values('elapsed_time')
            
            if len(pb_5k_all) == 0 or len(pb_10k_all) == 0:
                logger.warning(f"Missing 5K or 10K PB for athlete {athlete_id}")
                return None
                
            # Get the fastest 5K and 10K
            pb_5k = pb_5k_all.iloc[0].to_dict()
            pb_10k = pb_10k_all.iloc[0].to_dict()
        else:
            logger.warning(f"No PB data found for athlete {athlete_id}")
            return None
            
        # Calculate VDOT for both 5K and 10K performances
        # First convert to minutes for the calculate_vdot function
        vdot_5k, marathon_time_5k = calculate_vdot(
            pb_5k['distance'],  # distance in meters
            pb_5k['elapsed_time'] / 60  # convert seconds to minutes
        )
        
        vdot_10k, marathon_time_10k = calculate_vdot(
            pb_10k['distance'],  # distance in meters
            pb_10k['elapsed_time'] / 60  # convert seconds to minutes
        )
        
        logger.info(f"Calculated VDOTs: 5K={vdot_5k}, 10K={vdot_10k}")
        
        # Calculate personalized Riegel exponent
        exponent = calculate_riegel_exponent(
            pb_5k['elapsed_time'], pb_5k['distance'],
            pb_10k['elapsed_time'], pb_10k['distance']
        )
        
        # Determine which PB has better VDOT and use that as base for predictions
        if vdot_5k >= vdot_10k:
            base_distance = pb_5k['distance']
            base_time = pb_5k['elapsed_time']
            logger.info(f"Using 5K as base performance (better VDOT={vdot_5k})")
        else:
            base_distance = pb_10k['distance']
            base_time = pb_10k['elapsed_time']
            logger.info(f"Using 10K as base performance (better VDOT={vdot_10k})")
        
        # Generate predictions for all standard race distances
        predictions = {}
        for race_name, distance in RACE_DISTANCES.items():
            predictions[race_name] = get_prediction_ranges(base_distance, base_time, distance, exponent)
        
        # Store the prediction in the database
        try:
            prediction = RacePrediction(
                athlete_id=athlete_id,
                riegel_exponent=exponent,
                best_distance=base_distance,
                best_time=base_time,
                created_at=datetime.datetime.now()
            )
            db.session.add(prediction)
            db.session.commit()
            logger.info(f"Stored race prediction for athlete {athlete_id}")
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error storing race prediction: {e}")
        
        return {
            "exponent": exponent,
            "base_performance": {
                "distance": base_distance,
                "time": base_time,
                "formatted_time": format_time(base_time)
            },
            "predictions": predictions
        }
    except Exception as e:
        logger.error(f"Error calculating race predictions: {e}")
        return None

def get_latest_prediction(athlete_id: str) -> Optional[Dict]:
    """
    Get the latest prediction for an athlete from the database
    
    Args:
        athlete_id: Athlete's ID
        
    Returns:
        Prediction data or None if no prediction exists
    """
    try:
        prediction = RacePrediction.query.filter_by(athlete_id=athlete_id).order_by(desc(RacePrediction.created_at)).first()
        
        if not prediction:
            logger.info(f"No existing prediction for athlete {athlete_id}")
            return None
            
        # Generate predictions for all standard race distances
        predictions = {}
        for race_name, distance in RACE_DISTANCES.items():
            predictions[race_name] = get_prediction_ranges(
                prediction.best_distance, 
                prediction.best_time, 
                distance, 
                prediction.riegel_exponent
            )
        
        return {
            "exponent": prediction.riegel_exponent,
            "base_performance": {
                "distance": prediction.best_distance,
                "time": prediction.best_time,
                "formatted_time": format_time(prediction.best_time)
            },
            "predictions": predictions,
            "created_at": prediction.created_at
        }
        
    except Exception as e:
        logger.error(f"Error retrieving prediction: {e}")
        return None 