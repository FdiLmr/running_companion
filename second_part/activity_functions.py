import numpy
from statistics import stdev 
from scipy import signal
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Activity type mapping
ACTIVITY_TYPES: Dict[str, int] = {
    'Ride': 1, 'Run': 2, 'Swim': 3, 'Walk': 4,
    'Hike': 5, 'Alpine Ski': 6, 'Backcountry Ski': 7,
    'Canoe': 8, 'Crossfit': 9, 'E-BikeRide': 10,
    'Elliptical': 11, 'Handcycle': 12, 'IceSkate': 13,
    'InlineSkate': 14, 'Kayak': 15, 'KitesurfSession': 16,
    'Nordic Ski': 17, 'RockClimb': 18, 'RollerSki': 19,
    'Row': 20, 'Snowboard': 21, 'Snowshoe': 23,
    'StairStepper': 24, 'StandUpPaddle': 25, 'Surf': 26,
    'VirtualRide': 27, 'VirtualRun': 28, 'WeightTraining': 29,
    'WindsurfSession': 30, 'Wheelchair': 31, 'Workout': 32,
    'Yoga': 33, 'Other': 34
}

def get_activity_type(activity_name: str) -> int:
    """Convert activity name to type ID."""
    return ACTIVITY_TYPES.get(activity_name, 34)  # Default to 'Other'

def safe_get(activity: dict, key: str, default: Optional[any] = None) -> any:
    """Safely get value from activity dictionary."""
    try:
        value = activity.get(key)
        return value if value is not None and not numpy.isnan(value) else default
    except Exception:
        return default

def calculate_time_in_zones(hr_values: List[float], zones: List[int]) -> List[float]:
    """Calculate percentage of time spent in each heart rate zone."""
    time_in_zones = [0] * 5
    total_samples = len(hr_values)
    
    if total_samples == 0:
        return time_in_zones
        
    for hr in hr_values:
        if hr < zones[0]:
            time_in_zones[0] += 1
        elif zones[0] <= hr < zones[1]:
            time_in_zones[1] += 1
        elif zones[1] <= hr < zones[2]:
            time_in_zones[2] += 1
        elif zones[2] <= hr < zones[3]:
            time_in_zones[3] += 1
        else:
            time_in_zones[4] += 1
    
    return [round(count / total_samples, 2) for count in time_in_zones]

def calculate_signal_metrics(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Calculate standard deviation and frequency of peaks in signal."""
    try:
        std_dev = stdev(values) if len(values) > 1 else None
        peaks = signal.find_peaks(values)[0]
        freq = round(len(peaks) / len(values), 2) if peaks.size > 0 else None
        return std_dev, freq
    except Exception as e:
        logger.debug(f"Error calculating signal metrics: {e}")
        return None, None

def get_non_run_activity_data(activity: dict, zones: List[int]) -> Tuple:
    """Extract basic activity data common to all activity types."""
    return (
        activity['id'],
        get_activity_type(activity.get('type', 'Other')),
        safe_get(activity, 'elapsed_time'),
        safe_get(activity, 'distance'),
        safe_get(activity, 'average_heartrate')
    )

def get_run_activity_data(activity: dict, zones: List[int], hr_regressor) -> Tuple:
    """Extract detailed running activity data including HR zones and signal analysis."""
    # Get basic activity data
    basic_data = list(get_non_run_activity_data(activity, zones))
    
    # Additional running-specific metrics
    additional_data = {
        'cadence': safe_get(activity, 'average_cadence'),
        'elevation': safe_get(activity, 'elev_high', 0) - safe_get(activity, 'elev_low', 0),
        'pace': safe_get(activity, 'average_speed'),
        'athlete_count': safe_get(activity, 'athlete_count', 1)
    }
    
    # Initialize signal metrics
    signal_metrics = {
        'hr': (None, None),
        'elevation': (None, None),
        'pace': (None, None)
    }
    
    time_in_zones = [0] * 5
    
    try:
        laps = activity.get('laps', [])
        if laps:
            # Collect lap data
            lap_data = {
                'elevation': [],
                'pace': [],
                'hr': []
            }
            
            for lap in laps:
                lap_data['elevation'].append(safe_get(lap, 'total_elevation_gain', 0))
                lap_data['pace'].append(safe_get(lap, 'average_speed', 0))
                
                # Get or estimate heart rate
                hr = safe_get(lap, 'average_heartrate')
                if hr is None and hr_regressor is not None:
                    hr = hr_regressor.predict([[lap['average_speed']]])[0][0]
                if hr is not None:
                    lap_data['hr'].append(hr)
            
            # Calculate metrics for each signal type
            for signal_type in ['hr', 'elevation', 'pace']:
                if lap_data[signal_type]:
                    signal_metrics[signal_type] = calculate_signal_metrics(lap_data[signal_type])
            
            # Calculate time in zones if we have heart rate data
            if lap_data['hr']:
                time_in_zones = calculate_time_in_zones(lap_data['hr'], zones)
                # Update mean HR if we estimated it
                if basic_data[4] is None and hr_regressor is not None:
                    basic_data[4] = hr_regressor.predict([[additional_data['pace']]])[0][0]
    
    except Exception as e:
        logger.error(f"Error processing run data: {e}")
    
    # Combine all data
    return tuple(basic_data + [
        signal_metrics['hr'][0],      # HR stdev
        signal_metrics['hr'][1],      # HR frequency
        additional_data['elevation'],
        signal_metrics['elevation'][0],  # elevation stdev
        signal_metrics['elevation'][1],  # elevation frequency
        additional_data['pace'],
        signal_metrics['pace'][0],     # pace stdev
        signal_metrics['pace'][1],     # pace frequency
        additional_data['cadence'],
        additional_data['athlete_count'],
        time_in_zones
    ])

def get_run_hr_pace(activity: dict, zones: List[int]) -> Tuple[Optional[float], Optional[float]]:
    """Get heart rate and pace data for a running activity."""
    basic_data = get_non_run_activity_data(activity, zones)
    return basic_data[4], safe_get(activity, 'average_speed')


