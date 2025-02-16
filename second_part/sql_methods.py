from flask import current_app
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from sqlalchemy import text, create_engine, inspect
import os
import logging

logger = logging.getLogger(__name__)
db = SQLAlchemy()

def init_db(app):
    db.init_app(app)

def get_db_connection():
    return create_engine(f'mysql+pymysql://{os.environ.get("DB_USER")}:{os.environ.get("DB_PASS")}@{os.environ.get("DB_HOST")}/{os.environ.get("DB_NAME")}')

def read_db(table_name):
    try:
        engine = get_db_connection()
        logger.info(f"Reading from table {table_name}")
        
        # Check if table exists first
        inspector = inspect(engine)
        if table_name not in inspector.get_table_names():
            logger.info(f"Table {table_name} does not exist yet, returning empty DataFrame")
            return pd.DataFrame()
        
        df = pd.read_sql_table(table_name, engine)
        logger.info(f"Read {len(df)} rows from {table_name}")
        return df
    except Exception as e:
        logger.error(f"Error reading from database: {e}", exc_info=True)
        # Return empty DataFrame instead of raising error
        return pd.DataFrame()

def write_db_replace(df, table_name):
    """Write DataFrame to database, creating table if needed."""
    try:
        # Don't try to write empty DataFrames without schema
        if df.empty and len(df.columns) == 0:
            logger.info(f"Skipping write for empty DataFrame without schema: {table_name}")
            return True
            
        engine = get_db_connection()
        logger.info(f"Writing {len(df)} rows to table {table_name}")
        
        # Define default schemas for empty DataFrames
        default_schemas = {
            'metadata_athletes': {
                'id': 'str',
                'sex': 'str',
                'weight': 'float',
                'zones': 'str'
            },
            'metadata_blocks': {
                'athlete_id': 'str',
                'vdot': 'float',
                'vdot_delta': 'float',
                'predicted_marathon_time': 'float',
                'pb_date': 'datetime64[ns]',
                'block_id': 'str'
            },
            'all_athlete_activities': {
                'athlete_id': 'str',
                'block_id': 'str',
                'week_id': 'str',
                'activity_type': 'int',
                'activity_id': 'str',
                'elapsed_time': 'float',
                'distance': 'float',
                'mean_hr': 'float',
                'stdev_hr': 'float',
                'freq_hr': 'float',
                'time_in_z1': 'float',
                'time_in_z2': 'float',
                'time_in_z3': 'float',
                'time_in_z4': 'float',
                'time_in_z5': 'float',
                'elevation': 'float',
                'stdev_elevation': 'float',
                'freq_elevation': 'float',
                'pace': 'float',
                'stdev_pace': 'float',
                'freq_pace': 'float',
                'cadence': 'float',
                'athlete_count': 'float'
            },
            'all_athlete_weeks': {
                'athlete_id': 'str',
                'block_id': 'str',
                'week_id': 'str',
                'f_total_runs': 'int',
                'f_total_run_distance': 'float',
                'f_total_run_time': 'float',
                'f_total_non_run_distance': 'float',
                'f_total_non_run_time': 'float'
            },
            'features_activities': {
                'athlete_id': 'str',
                'block_id': 'str',
                'week_id': 'str',
                'activity_type': 'int',
                'activity_id': 'str',
                'elapsed_time': 'float',
                'distance': 'float',
                'mean_hr': 'float'
            },
            'features_weeks': {
                'athlete_id': 'str',
                'block_id': 'str',
                'week_id': 'str',
                'f_total_runs': 'int',
                'f_run_distance': 'float',
                'f_run_time': 'float',
                'f_non_run_distance': 'float',
                'f_non_run_time': 'float'
            },
            'features_blocks': {
                'athlete_id': 'str',
                'block_id': 'str',
                'y_vdot_delta': 'float',
                'y_vdot': 'float',
                'f_slope_run_distance': 'float',
                'f_slope_run_time': 'float',
                'f_slope_mean_run_hr': 'float',
                'f_taper_factor_run_distance': 'float',
                'f_taper_factor_run_time': 'float',
                'f_taper_factor_mean_run_hr': 'float'
            },
            'average_paces_and_hrs': {
                'athlete_id': 'str',
                'mean_hr': 'float',
                'pace': 'float'
            }
        }
        
        # If DataFrame is empty but we have a schema, create with schema
        if df.empty and table_name in default_schemas:
            df = pd.DataFrame(columns=default_schemas[table_name].keys())
            for col, dtype in default_schemas[table_name].items():
                df[col] = pd.Series(dtype=dtype)
        
        # Convert DataFrame to SQL
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists='replace',
            index=False
        )
        
        # Verify write
        with engine.connect() as conn:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
            logger.info(f"Written {count} rows to {table_name}")
        
        return True
    except Exception as e:
        logger.error(f"Error writing to database: {e}", exc_info=True)
        raise

def write_db_insert(df, table_name):
    try:
        engine = get_db_connection()
        df.to_sql(name=table_name, con=engine, if_exists='append', index=False)
        return True
    except Exception as e:
        logger.error(f"Error inserting into database: {e}")
        raise

def delete_rows(df_name):    
    try:
        with current_app.app_context():
            with db.engine.connect() as connection:
                result = connection.execute(text(f'DELETE FROM {df_name};'))
                return True
    except Exception as e:
        logger.error(f"Error deleting rows from {df_name}: {e}")
        return False

def test_conn_new():    
    try:
        engine = get_db_connection()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return "Connection successful!"
    except Exception as e:
        return f"Connection failed: {str(e)}"

def reset_database():
    """Safely reset all tables"""
    try:
        db.session.execute(text('SET FOREIGN_KEY_CHECKS = 0'))
        db.session.commit()
        
        # List of tables to truncate
        tables = [
            'activities',
            'athlete_stats',
            'metadata_athletes',
            'metadata_blocks',
            'all_athlete_activities',
            'all_athlete_weeks',
            'features_activities',
            'features_weeks',
            'features_blocks',
            'average_paces_and_hrs',
            'processing_status',
            'daily_limit'
        ]
        
        # Truncate all tables
        for table in tables:
            try:
                db.session.execute(text(f'TRUNCATE TABLE {table}'))
                logger.info(f"Truncated table: {table}")
            except Exception as e:
                logger.warning(f"Could not truncate {table}: {e}")
        
        # Reinitialize daily_limit with 0
        db.session.execute(text("INSERT INTO daily_limit (daily) VALUES (0)"))
        
        db.session.execute(text('SET FOREIGN_KEY_CHECKS = 1'))
        db.session.commit()
        logger.info("Database reset completed successfully")
        return True
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error resetting database: {e}")
        return False

"""
processing_status = read_db('processing_status')
processing_status_index = processing_status[processing_status['athlete_id']==str(int(athlete_id))].index.values.astype(int)[0]    
processing_status.at[processing_status_index, 'status'] = 'processed'
write_db_replace(processing_status, 'processing_status')


SHOW / DROP TABLES

rs = engine.execute('show tables')

for row in rs:
    print (row)

rs = engine.execute('drop table metadata_athletes')
rs = engine.execute('drop table metadata_blocks')
rs = engine.execute('drop table all_athlete_activities')
rs = engine.execute('drop table all_athlete_weeks')
rs = engine.execute('drop table features_activities')
rs = engine.execute('drop table features_weeks')
rs = engine.execute('drop table features_blocks')
rs = engine.execute('drop table average_paces_and_hrs')


"""