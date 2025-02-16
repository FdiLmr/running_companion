""" 
Basics:
    
To get data on athletes, you will have to make an application and request that
athletes sign in with Strava, and grant your application certain permission
using OAuth 2.0. You can get data on yourself without authentication for testing purposes.

Strava API usage is limited on a per-application basis using both a 
15-minute and daily request limit. 
The default rate limit allows 
- 200 requests every 15 minutes
- with up to 2000 requests per day. 
- aka 13 req/min for 2h30min 
- to a maximum of 6 hours per access token
- this script uses x requests for me, an athlete with many activities. 
so one ingestion per day is possible, worst case
An application’s 15-minute limit is reset at natural 15-minute intervals corresponding to 0, 15, 30 and 45 minutes after the hour. The daily limit resets at midnight UTC.


"""
import os
import requests
import logging
import pandas as pd  # Add pandas import
from sql_methods import read_db, write_db_replace

logger = logging.getLogger(__name__)
CLIENT_ID = os.environ.get('CLIENT_ID')

def get_athlete(bearer_token):
    url = 'https://www.strava.com/api/v3/athlete'
    headers = {"Authorization": f"Bearer {bearer_token}"}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises HTTPError for bad responses
        athlete_data = response.json()
        athlete_id = athlete_data['id']
        logger.info(f"Successfully retrieved data for athlete {athlete_id}")
        return athlete_data
    except requests.exceptions.RequestException as e:
        logger.error(f"Error requesting athlete data from Strava: {e}")
        return None
    except (KeyError, ValueError) as e:
        logger.error(f"Error parsing athlete data: {e}")
        return None

def get_athlete_data_status(athlete_id):
    try:
        processing_status = read_db('processing_status')
        athlete_id_str = str(athlete_id)
        
        if processing_status.empty:
            logger.info("Processing status table is empty")
            return "none"
            
        if athlete_id_str in processing_status["athlete_id"].values:
            mask = processing_status["athlete_id"] == athlete_id_str
            ingest_status = processing_status.loc[mask, "status"].iloc[0]
            logger.info(f"Found status '{ingest_status}' for athlete {athlete_id}")
            return ingest_status
            
        logger.info(f"No status found for athlete {athlete_id}")
        return "none"
        
    except Exception as e:
        logger.error(f"Error checking athlete status: {e}")
        return "none"

def queue_athlete_for_processing(athlete_id, bearer_token, refresh_token):
    try:
        logger.info(f"Starting to queue athlete {athlete_id}")
        
        # Create DataFrame with the new athlete data
        new_row = pd.DataFrame([{
            'athlete_id': str(athlete_id),
            'status': 'none',
            'bearer_token': bearer_token,
            'refresh_token': refresh_token
        }])
        
        try:
            # Try to read existing processing status
            processing_status = read_db('processing_status')
            logger.info(f"Current processing status entries: {len(processing_status)}")
            
            # Check if athlete already exists
            if str(athlete_id) in processing_status['athlete_id'].values:
                mask = processing_status['athlete_id'] == str(athlete_id)
                processing_status.loc[mask, 'status'] = 'none'
                processing_status.loc[mask, 'bearer_token'] = bearer_token
                processing_status.loc[mask, 'refresh_token'] = refresh_token
                logger.info("Updated existing athlete entry")
            else:
                processing_status = pd.concat([processing_status, new_row], ignore_index=True)
                logger.info("Added new athlete entry")
                
        except Exception as e:
            logger.warning(f"Could not read processing_status table: {e}")
            logger.info("Creating new processing status table")
            processing_status = new_row
        
        # Write back to database
        logger.info(f"Writing processing status with {len(processing_status)} entries")
        write_db_replace(processing_status, 'processing_status')
        logger.info(f"Successfully queued athlete {athlete_id}")
        return "none"
        
    except Exception as e:
        logger.error(f"Error queueing athlete {athlete_id}: {e}", exc_info=True)
        return None