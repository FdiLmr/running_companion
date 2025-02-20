"""
Initiating the three main dataframes:
    athlete_metadata - stores constants for the athlete, e.g. HR zones
    athlete_training_blocks - stores info about a training block between races
    athlete_training_week - stores info about a training week within a training block
"""

import pandas as pd 
import ast
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from sql_methods import write_db_replace, read_db, db
from search_functions import get_weeks, get_block
from running_functions import (
    build_pace_to_hr_regressor, 
    extract_activity_features,
    extract_week_features,
    get_pbs
)
from scipy.stats import linregress
from datetime import datetime
from models import MetadataPB
import os
import json

logger = logging.getLogger(__name__)

# Constants
MIN_ACTIVITIES_PER_BLOCK = 3

# Activity type groupings
OTHER_ACTIVITIES = {7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,33,34}
WALK_HIKE_ACTIVITIES = {4,5}

def load_file_data(athlete_id: int) -> dict:
    """Load athlete data from file if requested."""
    try:
        with open(f'./data/{athlete_id}.txt', 'r', encoding="utf8") as f:
            return ast.literal_eval(f.read())
    except Exception as e:
        logger.error(f"Error loading file data for athlete {athlete_id}: {e}")
        return None

def load_latest_athlete_data(athlete_id: int) -> dict:
    """Load the most recent data for an athlete from individual activity files."""
    try:
        activity_dir = f'./data/{athlete_id}'
        logger.debug(f"Looking for activity directory: {activity_dir}")
        if not os.path.exists(activity_dir):
            logger.error(f"No activity directory found for athlete {athlete_id}")
            return None

        all_activities = []
        file_list = os.listdir(activity_dir)
        logger.debug(f"Found {len(file_list)} files in {activity_dir}.")
        for filename in file_list:
            # We assume that individual activity files are named with a numeric activity ID.
            if filename.endswith(".json") and filename[:-5].isdigit():
                file_path = os.path.join(activity_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        activity_data = json.load(f)
                        all_activities.append(activity_data)
                except Exception as e:
                    logger.error(f"Error loading file {file_path}: {e}", exc_info=True)
        
        logger.info(f"Loaded {len(all_activities)} activity files for athlete {athlete_id}.")

        # Sort activities by start_date (assuming ISO format strings)
        try:
            all_activities.sort(key=lambda x: x.get('start_date', ''), reverse=True)
            logger.debug("Sorted activities by start_date (descending).")
        except Exception as e:
            logger.error(f"Error sorting activities for athlete {athlete_id}: {e}", exc_info=True)

        # Define file paths for athlete metadata, zones, and stats.
        athlete_file = f'./data/athlete_{athlete_id}_athlete.json'
        zones_file   = f'./data/athlete_{athlete_id}_zones.json'
        stats_file   = f'./data/athlete_{athlete_id}_stats.json'
        logger.debug(f"Expecting athlete metadata file at: {athlete_file}")
        logger.debug(f"Expecting zones file at: {zones_file}")
        logger.debug(f"Expecting stats file at: {stats_file}")

        with open(athlete_file, 'r', encoding='utf-8') as f:
            athlete_data = json.load(f)
        logger.debug("Athlete metadata loaded successfully.")

        with open(zones_file, 'r', encoding='utf-8') as f:
            zones_data = json.load(f)
        logger.debug("Zones data loaded successfully.")

        with open(stats_file, 'r', encoding='utf-8') as f:
            stats_data = json.load(f)
        logger.debug("Stats data loaded successfully.")

        final_data = {
            **athlete_data,       # Base athlete metadata.
            '_Zones': zones_data, # HR zones data.
            '_Stats': stats_data, # Stats data.
            '_Activities': all_activities  # List of all activities.
        }
        logger.info(f"Successfully loaded latest data for athlete {athlete_id}. Total activities: {len(all_activities)}")
        return final_data
    except Exception as e:
        logger.error(f"Error loading data for athlete {athlete_id}: {e}", exc_info=True)
        return None


def get_athlete_zones(athlete_data: dict) -> List[int]:
    """Extract heart rate zones from athlete data."""
    try:
        zones_raw = athlete_data['_Zones']['heart_rate']['zones']
        return [
            zones_raw[0]['max'],
            zones_raw[1]['max'],
            zones_raw[2]['max'],
            zones_raw[3]['max']
        ]
    except Exception:
        # Default zones based on max HR of 190
        return [
            round(190 * 0.6),
            round(190 * 0.7),
            round(190 * 0.8),
            round(190 * 0.9)
        ]

def get_error_free_activities(activities: List[dict]) -> List[dict]:
    """Filter out activities with errors."""
    error_free = []
    for activity in activities:
        try:
            if 'errors' in activity and activity['errors']:
                continue
            error_free.append(activity)
        except Exception:
            error_free.append(activity)
    return error_free

def process_activity_block(
    activities: List[dict],
    athlete_data: dict,
    athlete_id: str,
    zones: List[int],
    hr_regressor,
    block_id: str = '0'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process a block of activities and return activity and week features."""
    activities_df = pd.DataFrame()
    weeks_df = pd.DataFrame()
    
    weeks = get_weeks(activities, duration_days=0)
    
    for week_num, week in enumerate(weeks):
        week_id = f"{block_id}_{week_num}"
        
        # Process activities in week
        for activity in week:
            activities_df = extract_activity_features(
                activities_df, 
                activity, 
                zones, 
                activity['type'],
                athlete_data['id'],
                block_id,
                week_id,
                hr_regressor
            )
        
        # Process week features
        week_activities = activities_df[activities_df['week_id'] == week_id]
        week_runs = week_activities[week_activities['activity_type'] == 2]
        week_non_runs = week_activities[week_activities['activity_type'] != 2]
        
        week_features = extract_week_features(
            week_runs,
            week_non_runs,
            athlete_id,
            block_id,
            week_id,
            len(week_runs)
        )
        weeks_df = pd.concat([weeks_df, pd.DataFrame([week_features])], ignore_index=True)
    
    return activities_df, weeks_df

def process_pb_blocks(activities: List[dict], athlete_id: str, zones: List[int], hr_regressor) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process personal best blocks and extract features."""
    metadata_blocks = pd.DataFrame()
    features_activities = pd.DataFrame()
    features_weeks = pd.DataFrame()
    
    # significant_pbs is assumed to return lists of
    # [vdot, predicted marathon time (hrs), pb_date, activity_id, distance_category]
    significant_pbs = get_pbs(activities)
    
    for i, pb in enumerate(significant_pbs):
        activity_date, block_id = pb[2], pb[3]
        block = get_block(activities, activity_date)
        
        # Skip if the block is too short
        if len(block) < MIN_ACTIVITIES_PER_BLOCK:
            continue
        
        # For the current PB, search backwards for the most recent PB with the same distance category.
        prev_vdot = None
        for j in range(i - 1, -1, -1):
            if significant_pbs[j][4] == pb[4]:
                prev_vdot = significant_pbs[j][0]
                break
        
        vdot_delta = pb[0] - prev_vdot if prev_vdot is not None else 0
        
        # Add block metadata
        block_metadata = {
            'athlete_id': athlete_id,
            'vdot': pb[0],
            'vdot_delta': vdot_delta,
            'predicted_marathon_time': pb[1],
            'pb_date': pb[2],
            'block_id': block_id
        }
        metadata_blocks = pd.concat([metadata_blocks, pd.DataFrame([block_metadata])], ignore_index=True)
        
        # Process activities by week
        block_activities, block_weeks = process_activity_block(
            block,                  # activities list
            {'id': athlete_id},     # mock athlete_data dict with required 'id' field
            athlete_id,             # athlete_id
            zones,                  # zones
            hr_regressor,           # hr_regressor
            block_id                # block_id
        )
        
        features_activities = pd.concat([features_activities, block_activities], ignore_index=True)
        features_weeks = pd.concat([features_weeks, block_weeks], ignore_index=True)
    
    return metadata_blocks, features_activities, features_weeks


def calculate_activity_proportions(activities_df: pd.DataFrame, activity_type: Union[int, List[int]]) -> float:
    """Calculate proportion of activities of given type(s)."""
    if isinstance(activity_type, int):
        mask = activities_df['activity_type'] == activity_type
    else:
        mask = activities_df['activity_type'].isin(activity_type)
    
    try:
        return round(len(activities_df[mask]) / len(activities_df), 2)
    except ZeroDivisionError:
        return 0.0

def calculate_relative_proportion(block_prop: float, total_prop: float) -> Optional[float]:
    """Calculate relative proportion, handling division by zero."""
    try:
        return round(block_prop / total_prop, 2) if total_prop > 0 else None
    except Exception:
        return None

def get_activity_type_metrics(block_activities: pd.DataFrame, all_activities: pd.DataFrame) -> Dict[str, float]:
    """Calculate proportions for different activity types."""
    metrics = {}
    
    # Activity types to analyze
    activity_types = {
        'rides': 1,
        'swims': 3,
        'walks_hikes': WALK_HIKE_ACTIVITIES,
        'alpine_ski': 6,
        'workout': 32,
        'yoga': 34,
        'crossfit': 10,
        'other': OTHER_ACTIVITIES
    }
    
    for activity_name, activity_type in activity_types.items():
        block_prop = calculate_activity_proportions(block_activities, activity_type)
        total_prop = calculate_activity_proportions(all_activities, activity_type)
        rel_prop = calculate_relative_proportion(block_prop, total_prop)
        
        metrics.update({
            f'f_proportion_{activity_name}': block_prop,
            f'proportion_{activity_name}': total_prop,
            f'r_proportion_{activity_name}': rel_prop
        })
    
    return metrics

def get_outlier_metrics(
    features_activities: pd.DataFrame,
    all_athlete_activities: pd.DataFrame,
    block_id: str,
    athlete_id: str,
    block_activities: pd.DataFrame
) -> Dict[str, float]:
    """Calculate metrics for outlier activities."""
    from running_functions import get_run_outliers
    
    # Get outliers for block and overall
    f_outliers = get_run_outliers(features_activities, block_id, athlete_id)
    total_outliers = get_run_outliers(all_athlete_activities, '0', athlete_id)
    total_activities = len(all_athlete_activities)
    
    metrics = {}
    outlier_types = ['distance', 'intense', 'varying']
    
    for i, outlier_type in enumerate(outlier_types):
        # Calculate block proportions
        f_prop = round(len(f_outliers[i]) / len(block_activities), 2)
        total_prop = round(len(total_outliers[i]) / total_activities, 2)
        rel_prop = calculate_relative_proportion(f_prop, total_prop)
        
        metrics.update({
            f'f_proportion_{outlier_type}_activities': f_prop,
            f'proportion_{outlier_type}_activities': total_prop,
            f'r_proportion_{outlier_type}_activities': rel_prop
        })
    
    return metrics

def calculate_block_metrics(block_weeks: pd.DataFrame, athlete_weeks: pd.DataFrame, 
                          features_activities: pd.DataFrame, all_athlete_activities: pd.DataFrame,
                          block_id: str, athlete_id: str) -> Dict:
    """Calculate all block-level metrics."""
    metrics = {}
    
    # Get run statistics from appropriate column names
    if block_weeks.empty or 'f_total_runs' not in block_weeks.columns:
        return metrics
        
    # Check for required columns and use correct names
    run_distance_col = 'f_run_total_distance'
    
    # Skip blocks without sufficient data
    if (block_weeks['f_total_runs'].mean() == 0 or 
        (run_distance_col in block_weeks.columns and block_weeks[run_distance_col][:-2].mean() == 0)):
        return metrics
    
    # Get base metrics (existing code)
    metrics.update(calculate_training_metrics(block_weeks, athlete_weeks))
    
    # Get activity type proportions
    block_activities = features_activities[features_activities['block_id'] == block_id]
    metrics.update(get_activity_type_metrics(block_activities, all_athlete_activities))
    
    # Get outlier metrics
    metrics.update(get_outlier_metrics(
        features_activities,
        all_athlete_activities,
        block_id,
        athlete_id,
        block_activities
    ))
    
    return metrics

def calculate_training_metrics(block_weeks: pd.DataFrame, athlete_weeks: pd.DataFrame) -> Dict:
    """Calculate training metrics for a block."""
    metrics = {}
    
    # Map of metrics to their column names
    metric_mapping = {
        'run_distance': 'f_run_total_distance',
        'run_time': 'f_run_total_elapsed_time',
        'mean_hr': 'f_run_mean_hr'
    }
    
    # Calculate ramp rates
    for metric, column_name in metric_mapping.items():
        values = list(block_weeks[column_name][:-2])
        if values:
            slope, *_ = linregress(range(len(values)), values)
            metrics[f'f_slope_{metric}_before_taper'] = slope
            
            mean_value = block_weeks[column_name][:-2].mean()
            mean_taper = block_weeks[column_name][-2:].mean()
            metrics[f'f_taper_factor_{metric}'] = mean_taper / mean_value if mean_value else 0
    
    # Calculate relative metrics
    weekly_metrics = [
        ('run_distance', 'f_run_total_distance'),
        ('non_run_distance', 'f_non_run_total_distance'),
        ('run_time', 'f_run_total_elapsed_time'),
        ('non_run_time', 'f_non_run_total_elapsed_time'),
        ('run_elevation', 'f_mean_elevation'),
        ('athlete_count', 'f_athlete_count')
    ]
    
    for metric_name, col_name in weekly_metrics:
        if col_name in block_weeks.columns:
            block_mean = block_weeks[col_name].mean()
            athlete_mean = athlete_weeks[col_name].mean() if col_name in athlete_weeks.columns else np.nan
            metrics[f'f_avg_weekly_{metric_name}'] = block_mean
            if athlete_mean:
                metrics[f'r_avg_weekly_{metric_name}'] = block_mean / athlete_mean
        else:
            # Set default values or skip this metric
            metrics[f'f_avg_weekly_{metric_name}'] = np.nan
            metrics[f'r_avg_weekly_{metric_name}'] = np.nan

    
    # Calculate heart rate zones
    for zone in range(1, 6):
        # For runs
        zone_col = f'f_time_in_z{zone}_runs'
        block_mean = block_weeks[zone_col].mean()
        athlete_mean = np.nanmean(athlete_weeks[zone_col])
        metrics[f'f_avg_{zone_col}'] = block_mean
        if athlete_mean:
            metrics[f'r_avg_{zone_col}'] = block_mean / athlete_mean
            
        # For non-runs
        non_run_zone_col = f'f_time_in_z{zone}_non_runs'
        if non_run_zone_col in block_weeks.columns:
            block_mean = block_weeks[non_run_zone_col].mean()
            athlete_mean = np.nanmean(athlete_weeks[non_run_zone_col])
            metrics[f'f_avg_{non_run_zone_col}'] = block_mean
            if athlete_mean:
                metrics[f'r_avg_{non_run_zone_col}'] = block_mean / athlete_mean
    
    return metrics

def process_metadata_pbs(activities, athlete_id):
    """
    Process the 'best_efforts' in each activity and insert PB entries
    into the metadata_pbs table for each distance category where a PB exists.
    
    Args:
        activities (list): List of activity dictionaries (raw JSON data).
        athlete_id (int or str): The athlete's ID.
    """
    for activity in activities:
        best_efforts = activity.get("best_efforts", [])
        for effort in best_efforts:
            # Only process efforts where pr_rank is not null (i.e. it's a PB)
            if effort.get("pr_rank") is not None:
                try:
                    pb_date = datetime.strptime(effort.get("start_date")[:10], '%Y-%m-%d')
                except Exception as e:
                    pb_date = None
                pb = MetadataPB(
                    athlete_id=str(athlete_id),
                    distance_category=effort.get("name"),       # e.g., "5K", "10K", etc.
                    elapsed_time=effort.get("elapsed_time"),
                    distance=effort.get("distance"),
                    pr_rank=effort.get("pr_rank"),
                    start_date=pb_date,
                    activity_id=str(effort.get("activity", {}).get("id")),
                    pb_data=effort  # Optionally store the entire effort JSON
                )
                db.session.merge(pb)  # merge will update an existing record or insert a new one
    db.session.commit()

def merge_with_existing_data(new_data: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Instead of merging, just return the new data."""
    return new_data

def save_dataframes_to_db(dataframes: Dict[str, pd.DataFrame]) -> None:
    """Save multiple dataframes to database with proper formatting."""
    string_only_tables = {'metadata_athletes', 'metadata_blocks'}
    
    for table_name, df in dataframes.items():
        try:
            if (table_name in string_only_tables):
                df = df.astype(str)
            write_db_replace(df, table_name)
        except Exception as e:
            logger.error(f"Error saving {table_name} to database: {e}")
            raise

def transform_athlete_data(athlete_id: int, athlete_data: dict = None, populate_all_from_files: int = 0) -> None:
    """Transform athlete data and store in database."""
    try:
        if populate_all_from_files or athlete_data is None:
            athlete_data = load_latest_athlete_data(athlete_id)
            if not athlete_data:
                return
        
        # Validate basic data
        if 'sex' not in athlete_data:
            logger.error(f"Invalid data for athlete {athlete_id}")
            return
            
        # Initialize DataFrames
        metadata_athletes = pd.DataFrame()
        metadata_blocks = pd.DataFrame()
        all_athlete_activities = pd.DataFrame()
        all_athlete_weeks = pd.DataFrame()
        features_activities = pd.DataFrame()
        features_weeks = pd.DataFrame()
        features_blocks = pd.DataFrame()
        average_paces_and_hrs = pd.DataFrame()
        
        # Get clean activities and zones
        activities = get_error_free_activities(athlete_data['_Activities'])
        zones = get_athlete_zones(athlete_data)
        
        # Build HR regressor
        regressor, not_nan_rows = build_pace_to_hr_regressor(activities, athlete_id, zones)
        if not_nan_rows is not None:
            average_paces_and_hrs = pd.concat([average_paces_and_hrs, not_nan_rows], ignore_index=True)
        
        # Save athlete metadata
        metadata_athletes = pd.DataFrame([{
            'id': athlete_data['id'],
            'sex': athlete_data['sex'],
            'weight': athlete_data['weight'],
            'zones': athlete_data['_Zones']['heart_rate']['zones']
        }])
        
        # Check for existing athlete data and update if necessary
        existing_athletes = read_db('metadata_athletes')
        if not existing_athletes.empty:
            existing_athletes = existing_athletes.drop_duplicates(subset=['id'])
            metadata_athletes = metadata_athletes.drop_duplicates(subset=['id'])
            existing_athletes.set_index('id', inplace=True)
            metadata_athletes.set_index('id', inplace=True)
            metadata_athletes.update(existing_athletes)
            metadata_athletes.reset_index(inplace=True)
        
        # Process all activities
        activities.reverse()  # Process in chronological order
        all_athlete_activities, all_athlete_weeks = process_activity_block(
            activities, 
            athlete_data, 
            athlete_id,
            zones,
            regressor
        )
        
        # Process PB blocks
        metadata_blocks, features_activities, features_weeks = process_pb_blocks(
            activities,
            athlete_id,
            zones,
            regressor
        )
        
        # Calculate block-level features
        features_blocks = pd.DataFrame()
        for _, block in metadata_blocks.iterrows():
            block_id = block['block_id']
            block_weeks = features_weeks[features_weeks['block_id'] == block_id]
            athlete_weeks = all_athlete_weeks[all_athlete_weeks['athlete_id'] == athlete_id]
            
            # Skip blocks without sufficient data using correct column names
            run_distance_col = 'f_run_total_distance' if 'f_run_total_distance' in block_weeks.columns else 'f_total_run_distance'
            
            if (block_weeks['f_total_runs'].mean() == 0 or 
                (run_distance_col in block_weeks.columns and block_weeks[run_distance_col][:-2].mean() == 0)):
                continue
            
            # Calculate block metrics
            try:
                block_metrics = calculate_block_metrics(
                    block_weeks=block_weeks, 
                    athlete_weeks=athlete_weeks,
                    features_activities=features_activities,
                    all_athlete_activities=all_athlete_activities,
                    block_id=block_id,
                    athlete_id=athlete_id
                )
                
                block_metrics.update({
                    'athlete_id': athlete_id,
                    'block_id': block_id,
                    'y_vdot_delta': block['vdot_delta'],
                    'y_vdot': block['vdot']
                })
                
                features_blocks = pd.concat([features_blocks, pd.DataFrame([block_metrics])], ignore_index=True)
                
            except Exception as e:
                logger.error(f"Error calculating metrics for block {block_id}: {e}")
                continue
        
        # Save directly without merging
        dataframes_to_save = {
            'metadata_athletes': metadata_athletes,
            'metadata_blocks': metadata_blocks,
            'all_athlete_activities': all_athlete_activities,
            'all_athlete_weeks': all_athlete_weeks,
            'features_activities': features_activities,
            'features_weeks': features_weeks,
            'features_blocks': features_blocks,
            'average_paces_and_hrs': average_paces_and_hrs
        }
        
        # Save all dataframes to database directly
        save_dataframes_to_db(dataframes_to_save)
        
        process_metadata_pbs(athlete_data["_Activities"], athlete_id)
        
    except Exception as e:
        logger.error(f"Error transforming athlete data: {e}")
        raise
