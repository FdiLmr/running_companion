from sql_methods import write_db_replace, write_db_insert, read_db, db
from athlete_data_transformer import transform_athlete_data
import requests
import pandas as pd
import time
import os
import logging
from models import Activity, AthleteStats
from datetime import datetime
from flask import current_app
import json

logger = logging.getLogger(__name__)

DEBUG_MODE = True  # Set to False in production
ACTIVITIES_LIMIT = 25 if DEBUG_MODE else 90

def refresh_tokens():    
    try:
        processing_status = read_db('processing_status')
        
        for index, row in processing_status.iterrows():
            if row['athlete_id'] != 0 and row['status'] == "none":
                params = {
                    "client_id": os.environ.get('CLIENT_ID'),
                    "client_secret": os.environ.get('CLIENT_SECRET'),
                    "refresh_token": row['refresh_token'],
                    "grant_type": "refresh_token"
                }
                
                r = requests.post("https://www.strava.com/oauth/token", data=params)
                r.raise_for_status()
                response_data = r.json()
                
                processing_status.at[index,'bearer_token'] = response_data['access_token']
                processing_status.at[index,'refresh_token'] = response_data['refresh_token']
        
        write_db_replace(processing_status, 'processing_status')
        return 0
    except Exception as e:
        logger.error(f"Error refreshing tokens: {e}")
        return 1

def get_unprocessed_activities(activity_list, existing_ids, limit=ACTIVITIES_LIMIT):
    """Helper function to get activities we haven't processed yet"""
    new_activities = []
    for activity in activity_list:
        if len(new_activities) >= limit:
            break
        if activity['id'] not in existing_ids:
            new_activities.append(activity)
    return new_activities

def save_activity_data(athlete_id: int, activities: list, timestamp: str = None) -> None:
    """Save activities to a JSON file with timestamp."""
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create data directory if it doesn't exist
    data_dir = './data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    filename = f'{data_dir}/athlete_{athlete_id}_activities.json'
    
    # Load existing data if file exists
    existing_data = {}
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Could not read existing file {filename}, creating new one")
    
    # Add new data with timestamp
    existing_data[timestamp] = activities
    
    # Save updated data
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2)
    
    logger.info(f"Saved {len(activities)} activities to {filename}")

def fetch_strava_data():
    """Fetch data from Strava API and store in files and activity table."""
    start_time = time.time()
    total_activities_fetched = 0
    athletes_processed = 0
    
    # Get initial API call count
    daily_limit = read_db('daily_limit')    
    initial_api_calls = int(daily_limit.iloc[0,0])
    logger.info(f"Starting data processing. Initial API calls today: {initial_api_calls}/25000")
    
    if (initial_api_calls > 25000):
        logger.error("API LIMIT EXCEEDED")
        return "api limit exceeded"
    
    processing_status = read_db('processing_status')
    athletes_to_process = len(processing_status[processing_status['status'] == 'none'])
    logger.info(f"Found {athletes_to_process} athletes to process")
    
    current_api_calls = initial_api_calls
    
    for index, row in processing_status.iterrows():
        with current_app.app_context():  # Add application context
            athlete_id = int(row['athlete_id'])
            
            if athlete_id != 0 and row['status'] == "none":
                athlete_start_time = time.time()
                logger.info(f"Processing athlete {athlete_id}")
                
                bearer_token = row['bearer_token']            
                print ('processing athlete ' + str(athlete_id))
                headers = {"Authorization": "Bearer " + bearer_token}
            
                processing_status.at[index, 'status'] = 'processing'
                
                try:
                    
                    """
                    GET ATHLETE DATA
                    ----------
                    """
                    url = 'https://www.strava.com/api/v3/athlete'
                    data = ''
                    headers = {"Authorization": "Bearer " + bearer_token}
                    response = requests.get(url, data=data, headers=headers)
                    athlete_data = response.json()           
                    
                    """
                    GET ATHLETE ZONES
                    -----------
                    """
                    url = 'https://www.strava.com/api/v3/athlete/zones'
                    data = ''
                    response = requests.get(url, data=data, headers=headers)
                    athlete_zones = response.json()                    
                    current_api_calls += 1
                    
                    
                    """
                    GET ATHLETE STATS
                    -----------
                    """
                    url = 'https://www.strava.com/api/v3/athletes/' + str(athlete_id) + '/stats'
                    response = requests.get(url, headers=headers)
                    athlete_stats = response.json()
                    
                    # Store stats in database
                    try:
                        stats = AthleteStats(
                            athlete_id=str(athlete_id),
                            recent_run_totals=athlete_stats.get('recent_run_totals'),
                            all_run_totals=athlete_stats.get('all_run_totals'),
                            all_ride_totals=athlete_stats.get('all_ride_totals')
                        )
                        db.session.merge(stats)
                        db.session.commit()
                    except Exception as e:
                        logger.error(f"Error storing stats: {e}")
                        db.session.rollback()
                    
                    """
                    GET ACTIVITY LIST
                    -----------------
                    """
                    activities_to_process = ACTIVITIES_LIMIT  # Changed from hardcoded 90
                    all_activities = []
                    page = 1
                    
                    # First, get all activities IDs we already have
                    existing_activities = db.session.query(Activity.id).filter_by(athlete_id=str(athlete_id)).all()
                    existing_ids = {a[0] for a in existing_activities}
                    logger.info(f"Found {len(existing_ids)} existing activities")
                    
                    # Keep fetching pages until we have enough new activities
                    while True:
                        url = f'https://www.strava.com/api/v3/athlete/activities?per_page=100&page={page}'
                        response = requests.get(url, headers=headers)
                        this_response = response.json()
                        current_api_calls += 1
                        
                        if not this_response:  # No more activities
                            break
                            
                        all_activities.extend(this_response)
                        
                        # Check if we have enough new activities
                        unprocessed = get_unprocessed_activities(all_activities, existing_ids, activities_to_process)
                        if len(unprocessed) >= activities_to_process:
                            break
                            
                        page += 1
                        
                        # Rate limiting between pages
                        time.sleep(1.5)  # Ensure we don't hit rate limits
                    
                    logger.info(f"Found {len(all_activities)} total activities, {len(unprocessed)} new ones")
                    
                    # Check if there are more activities to process later
                    has_more_activities = len(all_activities) > len(existing_ids) + activities_to_process
                    
                    # Store what we found, even if zero new activities
                    activities = []
                    new_activities_count = 0
                    
                    if len(unprocessed) > 0:
                        """
                        GET DETAILED ACTIVITY DATA
                        ------------------------
                        """
                        for activity in unprocessed[:activities_to_process]:
                            start = time.time()
                            
                            activity_id = activity['id']
                            
                            # Create a directory for this athlete if it doesn't exist
                            athlete_dir = os.path.join('./data', str(athlete_id))
                            if not os.path.exists(athlete_dir):
                                os.makedirs(athlete_dir)
                            
                            # Build the file path using athlete_id and activity_id
                            file_path = os.path.join(athlete_dir, f'{activity_id}.json')
                            
                            if os.path.exists(file_path):
                                # If the file already exists, load the JSON data from disk
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    this_response = json.load(f)
                                logger.info(f"Fetching activity {activity_id} from cached file")
                            else:
                                # Otherwise, fetch the detailed activity data from Strava
                                url = f'https://www.strava.com/api/v3/activities/{activity_id}'
                                response = requests.get(url, headers=headers)
                                this_response = response.json()
                                # Save the JSON data to file
                                with open(file_path, 'w', encoding='utf-8') as f:
                                    json.dump(this_response, f, indent=2)
                                current_api_calls += 1  # Only count an API call if we had to fetch it
                                logger.info(f"Fetching activity {activity_id} from Strava")
                            
                            activities.append(this_response)
                            new_activities_count += 1
                            
                            # Mapping from best effort names (lowercase) to the Activity model column names.
                            best_effort_mapping = {
    "400m": "be_400m",
    "1/2 mile": "be_half_mile",
    "1K": "be_1km",
    "1 mile": "be_1_mile",
    "2 mile": "be_2_miles",
    "5K": "be_5km",
    "10K": "be_10km",
    "15K": "be_15km",
    "10 mile": "be_10_miles",
    "20K": "be_20km",
    "Half-Marathon": "be_half_marathon",
    "30K": "be_30km",
    "Marathon": "be_marathon",
    "50K": "be_50km"
}
                            
                            # Initialize a dictionary with keys for each best effort column and default value None.
                            be_columns = {col: None for col in best_effort_mapping.values()}

                            # Loop over the best efforts in the response (if any)
                            for effort in this_response.get('best_efforts', []):
                                name = effort.get('name', '')
                                elapsed = effort.get('elapsed_time')
                                if name in best_effort_mapping and elapsed is not None:
                                    col_name = best_effort_mapping[name]
                                    be_columns[col_name] = elapsed
                            
                            # Store activity in database
                            try:
                                activity = Activity(
                                    id=this_response['id'],
                                    athlete_id=str(athlete_id),
                                    name=this_response.get('name'),
                                    distance=this_response.get('distance'),
                                    moving_time=this_response.get('moving_time'),
                                    elapsed_time=this_response.get('elapsed_time'),
                                    total_elevation_gain=this_response.get('total_elevation_gain'),
                                    type=this_response.get('type'),
                                    start_date=datetime.strptime(this_response.get('start_date'), '%Y-%m-%dT%H:%M:%SZ'),
                                    average_speed=this_response.get('average_speed'),
                                    max_speed=this_response.get('max_speed'),
                                    average_heartrate=this_response.get('average_heartrate'),
                                    max_heartrate=this_response.get('max_heartrate'),
                                    activity_data=this_response,
                                    suffer_score=this_response.get('suffer_score'),
                                    map_data=this_response.get('map'),
                                    laps_data=this_response.get('laps'),
                                    # Best efforts columns (unchanged)
                                    be_400m=be_columns.get('be_400m'),
                                    be_half_mile=be_columns.get('be_half_mile'),
                                    be_1km=be_columns.get('be_1km'),
                                    be_1_mile=be_columns.get('be_1_mile'),
                                    be_2_miles=be_columns.get('be_2_miles'),
                                    be_5km=be_columns.get('be_5km'),
                                    be_10km=be_columns.get('be_10km'),
                                    be_15km=be_columns.get('be_15km'),
                                    be_10_miles=be_columns.get('be_10_miles'),
                                    be_20km=be_columns.get('be_20km'),
                                    be_half_marathon=be_columns.get('be_half_marathon'),
                                    be_30km=be_columns.get('be_30km'),
                                    be_marathon=be_columns.get('be_marathon'),
                                    be_50km=be_columns.get('be_50km'),
                                    # New fields
                                    resource_state=this_response.get('resource_state'),
                                    sport_type=this_response.get('sport_type'),
                                    workout_type=this_response.get('workout_type'),
                                    start_date_local=datetime.strptime(this_response.get('start_date_local'), '%Y-%m-%dT%H:%M:%SZ') if this_response.get('start_date_local') else None,
                                    timezone=this_response.get('timezone'),
                                    utc_offset=this_response.get('utc_offset'),
                                    location_city=this_response.get('location_city'),
                                    location_state=this_response.get('location_state'),
                                    location_country=this_response.get('location_country'),
                                    achievement_count=this_response.get('achievement_count'),
                                    kudos_count=this_response.get('kudos_count'),
                                    comment_count=this_response.get('comment_count'),
                                    athlete_count=this_response.get('athlete_count'),
                                    photo_count=this_response.get('photo_count'),
                                    trainer=this_response.get('trainer'),
                                    commute=this_response.get('commute'),
                                    manual=this_response.get('manual'),
                                    private=this_response.get('private'),
                                    visibility=this_response.get('visibility'),
                                    flagged=this_response.get('flagged'),
                                    gear_id=this_response.get('gear_id'),
                                    start_latlng=this_response.get('start_latlng'),
                                    end_latlng=this_response.get('end_latlng'),
                                    average_cadence=this_response.get('average_cadence'),
                                    average_temp=this_response.get('average_temp'),
                                    average_watts=this_response.get('average_watts'),
                                    max_watts=this_response.get('max_watts'),
                                    weighted_average_watts=this_response.get('weighted_average_watts'),
                                    device_watts=this_response.get('device_watts'),
                                    kilojoules=this_response.get('kilojoules'),
                                    has_heartrate=this_response.get('has_heartrate'),
                                    heartrate_opt_out=this_response.get('heartrate_opt_out'),
                                    display_hide_heartrate_option=this_response.get('display_hide_heartrate_option'),
                                    elev_high=this_response.get('elev_high'),
                                    elev_low=this_response.get('elev_low'),
                                    upload_id=this_response.get('upload_id'),
                                    upload_id_str=this_response.get('upload_id_str'),
                                    external_id=this_response.get('external_id'),
                                    from_accepted_tag=this_response.get('from_accepted_tag'),
                                    pr_count=this_response.get('pr_count'),
                                    total_photo_count=this_response.get('total_photo_count'),
                                    has_kudoed=this_response.get('has_kudoed'),
                                    description=this_response.get('description'),
                                    calories=this_response.get('calories'),
                                    perceived_exertion=this_response.get('perceived_exertion'),
                                    prefer_perceived_exertion=this_response.get('prefer_perceived_exertion'),
                                    device_name=this_response.get('device_name'),
                                    embed_token=this_response.get('embed_token'),
                                    private_note=this_response.get('private_note'),
                                    similar_activities=this_response.get('similar_activities'),
                                    available_zones=this_response.get('available_zones'),
                                    splits_metric=this_response.get('splits_metric'),
                                    splits_standard=this_response.get('splits_standard'),
                                    laps=this_response.get('laps'),
                                    photos=this_response.get('photos'),
                                    stats_visibility=this_response.get('stats_visibility'),
                                    hide_from_home=this_response.get('hide_from_home')
)
                                db.session.merge(activity)
                            except Exception as e:
                                logger.error(f"Error storing activity {activity_id}: {e}")
                                continue
                                
                            # Rate limiting
                            end = time.time()
                            remain = start + 1.5 - end
                            if remain > 0:
                                time.sleep(remain)
                        
                        try:
                            db.session.commit()
                            logger.info(f"Successfully stored {new_activities_count} new activities")
                        except Exception as e:
                            logger.error(f"Error committing activities to database: {e}")
                            db.session.rollback()
                            raise
                    else:
                        logger.info(f"No new activities to process for athlete {athlete_id}")
                    
                    # Always continue with metadata and stats
                    athlete_data["_Zones"] = athlete_zones
                    athlete_data["_Stats"] = athlete_stats
                    athlete_data["_Activities"] = activities
                    
                    # After processing activities, save athlete metadata, zones, and stats to fixed files.
                    data_dir = './data'
                    if not os.path.exists(data_dir):
                        os.makedirs(data_dir)
    
                    athlete_file = f'{data_dir}/athlete_{athlete_id}_athlete.json'
                    zones_file   = f'{data_dir}/athlete_{athlete_id}_zones.json'
                    stats_file   = f'{data_dir}/athlete_{athlete_id}_stats.json'
    
                    try:
                        with open(athlete_file, 'w', encoding='utf-8') as f:
                            json.dump(athlete_data, f, indent=2)
                        logger.info(f"Saved athlete metadata to {athlete_file}")
    
                        with open(zones_file, 'w', encoding='utf-8') as f:
                            json.dump(athlete_zones, f, indent=2)
                        logger.info(f"Saved zones data to {zones_file}")
    
                        with open(stats_file, 'w', encoding='utf-8') as f:
                            json.dump(athlete_stats, f, indent=2)
                        logger.info(f"Saved stats data to {stats_file}")
                    except Exception as e:
                        logger.error(f"Error saving athlete files: {e}")
                        daily_limit.at[0, 'daily'] = current_api_calls
                        write_db_replace(daily_limit, 'daily_limit')
                        processing_status.at[index, 'status'] = 'none'
                        return f'failure processing athlete {athlete_id}: {str(e)}'

                    
                except Exception as ex:                    
                    daily_limit.at[0, 'daily'] = current_api_calls
                    write_db_replace(daily_limit,'daily_limit')                                
                    processing_status.at[index, 'status'] = 'none'
                    return ('failure processing athlete ' + str(row['athlete_id']) + ': ' + str(ex))          
                                                
                # Always set status to 'none' so the athlete can be re-fetched
                processing_status.at[index, 'status'] = 'none'
                write_db_replace(processing_status, 'processing_status')
                
                daily_limit.at[0, 'daily'] = current_api_calls
                write_db_replace(daily_limit, 'daily_limit')       

                print ('successfully processed athlete ' + str(athlete_id))     
                if has_more_activities:
                    print(f'There are more activities to process for athlete {athlete_id}')

                # After successfully processing activities
                activities_count = len(activities)
                total_activities_fetched += activities_count
                athletes_processed += 1
                processing_time = time.time() - athlete_start_time
                
                avg_time_per_activity = processing_time / activities_count if activities_count > 0 else 0
                logger.info(f"""
                    Athlete {athlete_id} processing complete:
                    - Activities processed: {activities_count}
                    - Processing time: {processing_time:.2f} seconds
                    - Average time per activity: {avg_time_per_activity:.2f} seconds
                    - Current API calls: {current_api_calls}/25000
                """)
    
    total_time = time.time() - start_time
    api_calls_made = current_api_calls - initial_api_calls
    
    # Fix division by zero
    avg_time_per_athlete = total_time / athletes_processed if athletes_processed > 0 else 0
    avg_time_per_activity = total_time / total_activities_fetched if total_activities_fetched > 0 else 0
    
    summary = f"""
    Data Fetch Complete:
    ===================
    Athletes processed: {athletes_processed}/{athletes_to_process if athletes_to_process > 0 else 'None'}
    Total activities: {total_activities_fetched}
    Total processing time: {total_time:.2f} seconds
    Average time per athlete: {avg_time_per_athlete:.2f} seconds
    Average time per activity: {avg_time_per_activity:.2f} seconds
    API calls made: {api_calls_made}
    Initial API calls: {initial_api_calls}
    Final API calls: {current_api_calls}
    Remaining API calls: {25000 - current_api_calls}
    """
    
    logger.info(summary)
    return summary

def process_stored_data():
    """Process stored data files into analytics tables."""
    start_time = time.time()
    total_activities_processed = 0
    athletes_processed = 0

    logger.debug("Reading processing_status table from database.")
    processing_status = read_db('processing_status')
    athletes_to_process = len(processing_status[processing_status['status'] == 'none'])
    logger.info(f"Found {athletes_to_process} athletes to process (status == 'none').")

    for index, row in processing_status.iterrows():
        athlete_id = int(row['athlete_id'])
        logger.debug(f"Processing athlete with ID: {athlete_id}, current status: {row['status']}")
        if athlete_id != 0 and row['status'] in ["none", "processing"]:
            try:
                logger.info(f"Starting transformation for athlete {athlete_id}.")
                transform_athlete_data(athlete_id, populate_all_from_files=1)
                athletes_processed += 1
                logger.info(f"Transformation successful for athlete {athlete_id}.")
                # Only mark as processed if transform was successful
                processing_status.at[index, 'status'] = 'none'
                write_db_replace(processing_status, 'processing_status')
                logger.debug(f"Updated processing_status for athlete {athlete_id} to 'none'.")
            except Exception as e:
                logger.error(f"Error processing athlete {athlete_id}: {e}", exc_info=True)
                continue

    total_time = time.time() - start_time
    summary = f"""
    Processing Complete:
    ===================
    Athletes processed: {athletes_processed}
    Total processing time: {total_time:.2f} seconds
    """
    logger.info("process_stored_data() completed. " + summary)
    return summary
