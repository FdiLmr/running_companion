import math
import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.linear_model import LinearRegression
import logging
from activity_functions import get_non_run_activity_data, get_run_activity_data, get_run_hr_pace
from models import MetadataPB
from sklearn.metrics import mean_squared_error, r2_score  # Import necessary metrics

logger = logging.getLogger(__name__)

# Constants for PB calculations
PB_CONSTANTS = {
    'MIN_DISTANCE': 4900,
    'MAX_DISTANCE': 45000,
    'MIN_VDOT_INCREASE': 0.1,
    'MIN_DAYS_BETWEEN_PB': 7
}

def calculate_vdot(distance: float, time_minutes: float) -> Tuple[float, float]:
    """Calculate VDOT and predicted marathon time using Daniels' formula."""
    c = -4.6 + 0.182258 * (distance/time_minutes) + 0.000104 * (distance/time_minutes)**2
    i = 0.8 + 0.1894393 * math.exp(-0.012778 * time_minutes) + 0.2989558 * math.exp(-0.1932605 * time_minutes)
    vdot = c/i

    # Calculate marathon prediction
    d = 42200
    t = d * .004
    n = 0
    e = 1.0
    
    while n < 50 and e > 0.1:
        c = -4.6 + 0.182258 * (d/t) + 0.000104 * (d/t)**2
        i = 0.8 + 0.1894393 * math.exp(-0.012778 * t) + 0.2989558 * math.exp(-0.1932605 * t)
        e = abs(c - i*vdot)
        dc = -0.182258*d/t/t - 2*.000104*d*d/t/t/t
        di = -.012778*.1894393*math.exp(-.012778*t) - .1932605*.2989558*math.exp(-.1932605*t)
        dt = (c-i*vdot)/(dc-di*vdot)
        t -= dt
        n += 1

    return vdot, t*60  # Return VDOT and marathon time in seconds


def get_pbs(activities: List[dict]) -> List[List]:
    """
    Get a chronological list of significant personal bests (PBs) for target distance categories.
    
    For each activity that is a Run or Trail Run with a 'best_efforts' list, we check each effort
    whose name is in our target categories (5K, 10K, or 15K). If an effort's predicted marathon time
    (computed via Daniels' formula) is lower than the current best for that category, we record that
    effort as a new PB (and update the current best) without discarding previous PBs.
    
    Returns:
        List[List]: Each sublist contains:
            [vdot, predicted marathon time (in hours), pb_date, activity_id, distance_category].
    """
    import datetime
    significant_pbs = []
    # Maintain current best predicted marathon time for each category.
    current_best = {}  
    target_categories = {"5K", "10K", "15K"}
    default_best = 10 * 60 * 60  # 10 hours in seconds

    # Initialize current best for each target category.
    for cat in target_categories:
        current_best[cat] = default_best

    logger.info("Starting get_pbs computation to record PB history with detailed logging.")

    for activity in activities:
        if not isinstance(activity, dict):
            logger.debug("Skipping non-dictionary activity.")
            continue

        activity_type = activity.get('type')
        if activity_type not in ['Run', 'Trail Run']:
            #logger.debug(f"Skipping activity id {activity.get('id')}: type '{activity_type}' is not Run/Trail Run.")
            continue

        if 'best_efforts' not in activity:
            #logger.debug(f"Skipping activity id {activity.get('id')}: missing 'best_efforts' field.")
            continue

        if activity.get('id') == 9009284547:
            #logger.debug("Skipping known non-useful activity with id 9009284547.")
            continue

        # Parse the activity's start_date to use as pb_date.
        try:
            pb_date = datetime.datetime.strptime(activity.get('start_date')[:10], '%Y-%m-%d')
            #logger.debug(f"Parsed pb_date for activity id {activity.get('id')}: {pb_date}.")
        except Exception as e:
            #logger.error(f"Error parsing start_date for activity id {activity.get('id')}: {e}")
            pb_date = None

        #logger.debug(f"Processing activity id {activity.get('id')} with start_date {activity.get('start_date')}.")

        for effort in activity.get('best_efforts', []):
            effort_name = effort.get('name')
            if effort_name not in target_categories:
                #logger.debug(f"Skipping effort with name '{effort_name}' (not in target categories) in activity id {activity.get('id')}.")
                continue

            try:
                elapsed = float(effort.get('elapsed_time'))
                distance = float(effort.get('distance', 0))
                #logger.debug(f"Effort '{effort_name}' in activity id {activity.get('id')}: elapsed_time={elapsed}, distance={distance}.")
            except Exception as e:
                #logger.error(f"Error parsing elapsed_time or distance for effort '{effort_name}' in activity id {activity.get('id')}: {e}")
                continue

            if not (PB_CONSTANTS['MIN_DISTANCE'] <= distance <= PB_CONSTANTS['MAX_DISTANCE']):
                #logger.debug(f"Effort '{effort_name}' in activity id {activity.get('id')} has distance {distance} outside valid range.")
                continue

            try:
                vdot, marathon_pred_secs = calculate_vdot(
                    distance=distance,
                    time_minutes=elapsed / 60
                )
                #logger.debug(f"Calculated for effort '{effort_name}' in activity id {activity.get('id')}: vdot={vdot:.2f}, predicted marathon time={marathon_pred_secs:.2f} sec.")
            except Exception as e:
                #logger.error(f"Error calculating vdot for effort '{effort_name}' in activity id {activity.get('id')}: {e}")
                continue

            # If the effort beats the current best, record it as a new PB.
            if marathon_pred_secs < current_best[effort_name]:
                logger.info(
                    f"New PB for {effort_name} found in activity id {activity.get('id')}: "
                    f"predicted marathon time {marathon_pred_secs/3600:.2f}h (previous best: "
                    f"{current_best[effort_name]/3600:.2f}h), vdot {vdot:.2f}."
                )
                significant_pbs.append([
                    vdot,
                    marathon_pred_secs / 3600,  # Convert seconds to hours.
                    pb_date,
                    activity.get('id'),
                    effort_name
                ])
                # Update current best so that later efforts must be even better.
                current_best[effort_name] = marathon_pred_secs
            #else:
                #logger.debug(
                #    f"Effort '{effort_name}' in activity id {activity.get('id')} did not beat current best "
                #    f"(current best: {current_best[effort_name]/3600:.2f}h, computed: {marathon_pred_secs/3600:.2f}h)."
                #)

    logger.info("Completed get_pbs computation; returning PB history.")
    return significant_pbs



def get_run_outliers(all_activities: pd.DataFrame, block_id: int, athlete_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Identify distance, intensity, and interval outliers in running activities."""
    runs = all_activities[all_activities['activity_type'] == 2]
    block_activities = runs[runs['block_id'] == block_id]
    athlete_activities = runs[runs['athlete_id'] == athlete_id]

    # Initialize empty DataFrames
    distance_outliers = pd.DataFrame()
    intensity_outliers = pd.DataFrame()
    interval_outliers = pd.DataFrame()

    try:
        # Distance outliers
        athlete_mean_distance = athlete_activities['distance'].mean()
        distance_outliers = athlete_activities[
            (np.abs(stats.zscore(athlete_activities["distance"])) >= 1.2) &
            (athlete_activities["distance"] >= athlete_mean_distance) &
            (athlete_activities['block_id'] == block_id)
        ]

        # Intensity outliers
        intensity_outliers = pd.DataFrame()
        
        # Heart rate based intensity
        if athlete_activities['mean_hr'].sum() != 0:
            hr_activities = athlete_activities[athlete_activities['mean_hr'].notna()]
            athlete_mean_hr = hr_activities['mean_hr'].mean()
            hr_outliers = hr_activities[
                (np.abs(stats.zscore(hr_activities["mean_hr"])) >= 1.2) &
                (hr_activities["distance"] >= athlete_mean_distance) &
                (hr_activities["mean_hr"] >= athlete_mean_hr) &
                (hr_activities['block_id'] == block_id)
            ]
            intensity_outliers = pd.concat([intensity_outliers, hr_outliers])

        # Pace based intensity
        if athlete_activities['pace'].sum() != 0:
            pace_activities = athlete_activities[athlete_activities['pace'].notna()]
            athlete_mean_pace = pace_activities['pace'].mean()
            pace_outliers = pace_activities[
                (np.abs(stats.zscore(pace_activities["pace"])) >= 1.5) &
                (pace_activities["distance"] >= 1000) &
                (pace_activities["pace"] >= athlete_mean_pace) &
                (pace_activities['block_id'] == block_id)
            ]
            intensity_outliers = pd.concat([intensity_outliers, pace_outliers])

        # Interval outliers
        if not athlete_activities['stdev_hr'].isna().all():
            stdev_hr_threshold = athlete_activities['stdev_hr'].mean()
            interval_outliers = block_activities[
                (block_activities['stdev_hr'].notna()) &
                (block_activities["distance"] >= 1000) &
                (block_activities["stdev_hr"] >= stdev_hr_threshold)
            ]

    except Exception as e:
        logger.error(f"Error identifying outliers: {e}")

    return distance_outliers.drop_duplicates(), intensity_outliers.drop_duplicates(), interval_outliers.drop_duplicates()

def extract_activity_features(
    activities: pd.DataFrame, 
    activity: dict, 
    zones: List[int], 
    activity_type: str, 
    athlete_id: str, 
    block_id: int, 
    week_id: int, 
    hr_regressor
) -> pd.DataFrame:
    """Extract features from an activity and add them to the activities DataFrame."""
    
    base_features = {
        'athlete_id': athlete_id,
        'block_id': block_id,
        'week_id': week_id
    }
    
    try:
        if activity_type not in ['Run', 'TrailRun']:
            # Handle non-run activities
            basic_data = get_non_run_activity_data(activity, zones)
            features = {
                **base_features,
                'activity_type': basic_data[1],
                'activity_id': basic_data[0],
                'elapsed_time': basic_data[2],
                'distance': basic_data[3],
                'mean_hr': basic_data[4]
            }
        else:
            # Handle run activities
            run_data = get_run_activity_data(activity, zones, hr_regressor)
            features = {
                **base_features,
                'activity_type': run_data[1],
                'activity_id': run_data[0],
                'elapsed_time': run_data[2],
                'distance': run_data[3],
                'mean_hr': run_data[4],
                'stdev_hr': run_data[5],
                'freq_hr': run_data[6],
                'time_in_z1': run_data[-1][0],
                'time_in_z2': run_data[-1][1],
                'time_in_z3': run_data[-1][2],
                'time_in_z4': run_data[-1][3],
                'time_in_z5': run_data[-1][4],
                'elevation': run_data[7],
                'stdev_elevation': run_data[8],
                'freq_elevation': run_data[9],
                'pace': run_data[10],
                'stdev_pace': run_data[11],
                'freq_pace': run_data[12],
                'cadence': run_data[13],
                'athlete_count': run_data[14]
            }
        
        new_features_df = pd.DataFrame([features]).dropna(axis=1, how='all')
        return pd.concat([activities, new_features_df], ignore_index=True)
        
    except Exception as e:
        logger.error(f"Error extracting features from activity: {e}")
        return activities

def calculate_week_stats(df: pd.DataFrame, columns: List[str]) -> Dict[str, float]:
    """Calculate basic statistics for a set of columns in a DataFrame."""
    stats = {}
    for col in columns:
        series = df[col]
        if not series.empty:
            stats.update({
                f"total_{col}": series.sum(),
                f"avg_{col}": series.mean(),
                f"stdev_{col}": series.std()
            })
    return stats

def extract_week_features(
    week_runs: pd.DataFrame, 
    week_non_runs: pd.DataFrame, 
    athlete_id: str, 
    block_id: int, 
    week_id: int, 
    total_runs: int
) -> Dict:
    """Extract features from a week of training data."""
    
    features = {
        'athlete_id': athlete_id,
        'block_id': block_id,
        'week_id': week_id,
        'f_total_runs': total_runs
    }
    
    # Calculate run statistics
    run_stats = calculate_week_stats(week_runs, ['distance', 'elapsed_time'])
    features.update({f"f_run_{k}": v for k, v in run_stats.items()})
    
    # Calculate non-run statistics
    non_run_stats = calculate_week_stats(week_non_runs, ['distance', 'elapsed_time'])
    features.update({f"f_non_run_{k}": v for k, v in non_run_stats.items()})
    
    # Calculate heart rate zones for runs and non-runs
    for activity_type in ['runs', 'non_runs']:
        df = week_runs if activity_type == 'runs' else week_non_runs
        for zone in range(1, 6):
            zone_col = f"time_in_z{zone}"
            # Only calculate mean if the column exists, otherwise use 0
            if zone_col in df.columns:
                features[f"f_time_in_z{zone}_{activity_type}"] = df[zone_col].mean()
            else:
                features[f"f_time_in_z{zone}_{activity_type}"] = 0.0
    
    # Count non-run activity types
    activity_counts = week_non_runs['activity_type'].value_counts()
    for activity_type, count in activity_counts.items():
        features[f"f_activity_type_{int(activity_type)}"] = count
    
    # Calculate running-specific metrics safely
    run_metrics = {
        'run_mean_hr': ('mean_hr', 'mean'),
        'run_stdev_hr': ('stdev_hr', 'mean'),
        'run_freq_hr': ('freq_hr', 'mean'),
        'elevation': ('elevation', ['sum', 'mean']),
        'stdev_elevation': ('stdev_elevation', 'mean'),
        'freq_elevation': ('freq_elevation', 'mean'),
        'pace': ('pace', 'mean'),
        'stdev_pace': ('stdev_pace', 'mean'),
        'freq_pace': ('freq_pace', 'mean'),
        'cadence': ('cadence', 'mean'),
        'athlete_count': ('athlete_count', 'mean')
    }
    
    # Add non-run metrics
    non_run_metrics = {
        'non_run_mean_hr': ('mean_hr', 'mean'),
        'non_run_stdev_hr': ('stdev_hr', 'mean'),
        'non_run_freq_hr': ('freq_hr', 'mean')
    }
    
    # Process run metrics
    for metric, (col, aggs) in run_metrics.items():
        if col in week_runs.columns:
            try:
                if isinstance(aggs, list):
                    for agg in aggs:
                        features[f"f_{agg}_{metric}"] = week_runs[col].agg(agg)
                else:
                    features[f"f_{metric}"] = week_runs[col].agg(aggs)
            except Exception as e:
                logger.warning(f"Could not calculate {metric}: {e}")
                # Set default values if calculation fails
                if isinstance(aggs, list):
                    for agg in aggs:
                        features[f"f_{agg}_{metric}"] = 0.0
                else:
                    features[f"f_{metric}"] = 0.0
        else:
            # Set default values for missing columns
            if isinstance(aggs, list):
                for agg in aggs:
                    features[f"f_{agg}_{metric}"] = 0.0
            else:
                features[f"f_{metric}"] = 0.0
    
    # Process non-run metrics
    for metric, (col, aggs) in non_run_metrics.items():
        if col in week_non_runs.columns:
            try:
                features[f"f_{metric}"] = week_non_runs[col].agg(aggs)
            except Exception as e:
                logger.warning(f"Could not calculate non-run {metric}: {e}")
                features[f"f_{metric}"] = 0.0
        else:
            features[f"f_{metric}"] = 0.0
    
    return features

def build_pace_to_hr_regressor(activities: List[dict], athlete_id: str, zones: List[int]) -> Tuple[Optional[LinearRegression], Optional[pd.DataFrame]]:
    """Build a regression model to predict heart rate from pace."""
    
    paces_and_hrs = pd.DataFrame([{
        'athlete_id': athlete_id,
        'mean_hr': hr,
        'pace': pace
    } for hr, pace in [get_run_hr_pace(act, zones) for act in activities if act.get('type') == 'Run']])
    
    if paces_and_hrs.empty:
        return None, None
    
    valid_data = paces_and_hrs.dropna(subset=['mean_hr', 'pace'])
    
    if len(valid_data) <= 5:  # Need more than 5 points for meaningful regression
        return None, valid_data
    
    try:
        X = valid_data['pace'].values.reshape(-1, 1)
        y = valid_data['mean_hr'].values.reshape(-1, 1)
        
        regressor = LinearRegression()
        regressor.fit(X, y)

        # Log the coefficients and intercept of the regression model
        logger.info(f"Regressor coefficients: {regressor.coef_}, Intercept: {regressor.intercept_}")

        # Calculate predictions and metrics
        predictions = regressor.predict(X)
        mse = mean_squared_error(y, predictions)
        rmse = mse ** 0.5
        r2 = r2_score(y, predictions)

        # Log the metrics
        logger.info(f"R²: {r2:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")
        
        return regressor, valid_data
    except Exception as e:
        logger.error(f"Error building pace-to-HR regressor: {e}")
        return None, valid_data