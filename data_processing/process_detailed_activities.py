import json
import logging
import pandas as pd
import boto3
from io import StringIO

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    logging.info("Starting the script...")

    # Create S3 client. Specify region if your bucket is in eu-west-3
    s3 = boto3.client('s3', region_name='eu-west-3')

    # Configure the correct bucket and key
    bucket_name = "runningcompanionbucket"
    key = "user_data/user_id_1/detailed_activities.json"

    logging.info(f"Fetching {key} from bucket {bucket_name}...")

    try:
        # Step 1: Read JSON from S3
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        json_string = obj['Body'].read().decode('utf-8')
        data = json.loads(json_string)

        logging.info("Successfully fetched and decoded JSON data from S3.")
    except Exception as e:
        logging.error(f"Error fetching file from S3: {e}")
        return  # Stop script if we can't fetch the file

    # If the JSON is a single activity object, wrap in a list for consistent handling
    activities = data if isinstance(data, list) else [data]
    logging.info(f"Number of activities loaded: {len(activities)}")

    # ------------------------------
    # Top-Level Fields -> CSV
    # ------------------------------
    df_activities = pd.json_normalize(activities)
    logging.info(f"Flattened top-level fields into df_activities with {len(df_activities)} rows.")

    df_activities.to_csv('detailed_activities_top_level.csv', index=False)
    logging.info("Saved detailed_activities_top_level.csv locally.")

    # For convenience, define a helper to flatten a nested list if it exists:
    def export_nested_list(activities, record_path_key, output_csv_name, record_prefix):
        """Flatten the given record_path_key (nested list) into a separate CSV."""
        # Filter: only include activities that actually have this key
        items_with_key = [act for act in activities if record_path_key in act]

        if len(items_with_key) == 0:
            logging.warning(f"No '{record_path_key}' found in any of the activities. Skipping {output_csv_name}.")
            return

        # Now flatten only those items_with_key
        df_nested = pd.json_normalize(
            data=items_with_key,
            record_path=record_path_key,
            meta=['id', 'name', 'type', 'distance', 'start_date', 'elapsed_time', 'moving_time'],
            record_prefix=record_prefix
        )
        
        logging.info(f"Flattened {record_path_key} -> {len(df_nested)} rows.")
        df_nested.to_csv(output_csv_name, index=False)
        logging.info(f"Saved {output_csv_name} locally.")


    # ------------------------------
    # Nested Fields -> CSV
    # ------------------------------
    export_nested_list(activities, record_path_key='segment_efforts',
                       output_csv_name='data/detailed_activities_segments.csv',
                       record_prefix='segment_efforts_')

    export_nested_list(activities, record_path_key='laps',
                       output_csv_name='data/detailed_activities_laps.csv',
                       record_prefix='laps_')

    export_nested_list(activities, record_path_key='splits_metric',
                       output_csv_name='data/detailed_activities_splits.csv',
                       record_prefix='splits_metric_')

    export_nested_list(activities, record_path_key='highlighted_kudosers',
                       output_csv_name='data/detailed_activities_kudosers.csv',
                       record_prefix='kudosers_')

    export_nested_list(activities, record_path_key='splits_standard',
                       output_csv_name='data/detailed_activities_splits_standard.csv',
                       record_prefix='splits_standard_')

    export_nested_list(activities, record_path_key='best_efforts',
                       output_csv_name='data/detailed_activities_best_efforts.csv',
                       record_prefix='best_efforts_')

    logging.info("Script finished successfully.")

if __name__ == '__main__':
    main()
