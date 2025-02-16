from sql_methods import db, get_db_connection
import datetime
import pandas as pd
import io
import math
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import logging
from models import FeaturesBlock, MetadataBlock, ModelOutput
from sqlalchemy import select, text
from flask import current_app

# Configure logging
logger = logging.getLogger(__name__)
# Set matplotlib and PIL loggers to WARNING level only
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

feature_labels = {
    'f_slope_run_distance_before_taper': 'Week on week distance increase\n(metres)',
    'f_taper_factor_run_distance': 'Distance tapering\n(ratio decreased for last two weeks)',
    'f_slope_run_time_before_taper': 'Week on week running time increase\n(seconds)',
    'f_taper_factor_run_time': 'Decrease in time when tapering\n(ratio)',
    'f_slope_mean_hr_before_taper': 'Week on week intensity increase\n(average HR)',
    'f_taper_factor_mean_hr': 'Heart rate (intensity) tapering\n(ratio)',
    'f_avg_weekly_run_distance': 'Weekly run distance\n(metres)',
    'r_avg_weekly_run_distance': 'RELATIVE: Average weekly run distance\n(ratio: this period vs all your data)',
    'f_avg_weekly_non_run_distance': 'Weekly non-run distance\n(metres)',
    'r_avg_weekly_non_run_distance': 'RELATIVE: Weekly non-run distance\n(ratio: this period vs all your data)',
    'f_avg_weekly_run_time': 'Weekly run time\n(seconds)',
    'r_avg_weekly_run_time': 'RELATIVE: Weekly run time\n(ratio: this period vs all your data)',
    'f_avg_weekly_non_run_time': 'Weekly time spent on other activities\n(seconds)',
    'r_avg_weekly_non_run_time': 'RELATIVE: Weekly time spent on other activities\n(ratio: this period vs all your data)',
    'f_avg_weekly_run_elevation': 'Weekly run elevation\n(metres)',
    'r_avg_weekly_run_elevation': 'RELATIVE: Weekly run elevation\n(ratio: this period vs all your data)',
    'f_avg_weekly_athlete_count': 'How many athletes you trained with on average\n(people)',
    'r_avg_weekly_athlete_count': 'RELATIVE: Athletes trained with\n(ratio: this period vs all your data)',
    'f_avg_f_time_in_z1_runs': 'Weekly time in Z1 for runs\n(out of 1)',
    'f_avg_f_time_in_z2_runs': 'Weekly time in Z2 for runs\n(out of 1)',
    'f_avg_f_time_in_z3_runs': 'Weekly time in Z3 for runs\n(out of 1)',
    'f_avg_f_time_in_z4_runs': 'Weekly time in Z4 for runs\n(out of 1)',
    'f_avg_f_time_in_z5_runs': 'Weekly time in Z5 for runs\n(out of 1)',
    'r_avg_f_time_in_z1_runs': 'RELATIVE: Weekly time in Z1 for runs\n(ratio: this period vs all your data)',
    'r_avg_f_time_in_z2_runs': 'RELATIVE: Weekly time in Z2 for runs\n(ratio: this period vs all your data)',
    'r_avg_f_time_in_z3_runs': 'RELATIVE: Weekly time in Z3 for runs\n(ratio: this period vs all your data)',
    'r_avg_f_time_in_z4_runs': 'RELATIVE: Weekly time in Z4 for runs\n(ratio: this period vs all your data)',
    'f_proportion_distance_activities': 'Proportion of long runs\n(ratio)',
    'f_proportion_intense_activities': 'Proportion of intense runs\n(ratio)',
    'f_proportion_varying_activities': 'Proportion of interval runs\n(ratio)',  
    'r_proportion_distance_activities': 'RELATIVE: Proportion of long runs\n(ratio: this period vs all your data)',
    'r_proportion_intense_activities': 'RELATIVE: Proportion of intense runs\n(ratio: this period vs all your data)',
    'r_proportion_varying_activities': 'RELATIVE: Proportion of interval runs\n(ratio: this period vs all your data)',  
    'f_proportion_rides': 'Proportion of rides\n(ratio)',  
    'f_proportion_swims': 'Proportion of swims\n(ratio)',  
    'f_proportion_walks_hikes': 'Proportion of walks or hikes\n(ratio)',  
    'f_proportion_alpine_ski': 'Proportion of alpine skiing\n(ratio)',  
    'f_proportion_workout': 'Proportion of workouts\n(ratio)',  
    'f_proportion_yoga': 'Proportion of yoga sessions\n(ratio)',  
    'f_proportion_crossfit': 'Proportion of crossfit sessions\n(ratio)'
}

def athletevsbest(athlete_id):
    """Generate visualization comparing athlete's performance to best athletes."""
    try:
        athlete_id = str(athlete_id)  # Convert to string to match DB schema
        logger.info(f"Starting visualization for athlete {athlete_id}")
        
        with current_app.app_context():
            engine = get_db_connection()
            
            # Get features blocks for this athlete
            features_blocks = pd.read_sql(
                "SELECT * FROM features_blocks",
                engine
            )
            logger.info(f"Found {len(features_blocks)} total feature blocks")
            logger.info(f"Features blocks columns: {features_blocks.columns.tolist()}")
            logger.info(f"Sample athlete_ids in features_blocks: {features_blocks['athlete_id'].head().tolist()}")
            logger.info(f"athlete_id type in features_blocks: {features_blocks['athlete_id'].dtype}")
            
            # Convert athlete_id column to string for comparison
            features_blocks['athlete_id'] = features_blocks['athlete_id'].astype(str)
            
            if len(features_blocks) == 0:
                logger.error(f"No feature blocks found in database")
                return None
                
            # Get metadata blocks
            metadata_blocks = pd.read_sql(
                "SELECT * FROM metadata_blocks",
                engine
            )
            logger.info(f"Found {len(metadata_blocks)} metadata blocks")
            
            # Get model outputs (this can be empty, we'll handle it)
            try:
                model_outputs = pd.read_sql(
                    "SELECT * FROM model_outputs",
                    engine
                )
            except:
                logger.warning("Could not read model_outputs table, using default values")
                model_outputs = pd.DataFrame(columns=['y_name', 'feature_name', 'importance'])
            
            logger.info(f"Found {len(model_outputs)} model outputs")
            
            # Get this athlete's last block
            this_athlete_blocks = features_blocks[features_blocks['athlete_id']==athlete_id]
            logger.info(f"Found {len(this_athlete_blocks)} blocks for athlete {athlete_id}")
            if len(this_athlete_blocks) > 0:
                logger.info(f"Sample of athlete's block data: {this_athlete_blocks.iloc[-1].to_dict()}")
            
            if len(this_athlete_blocks) == 0:
                logger.warning(f"No blocks found for athlete {athlete_id}, using empty values")
                this_athlete_last_block = pd.Series(0.0, index=features_blocks.columns)
            else:
                this_athlete_last_block = this_athlete_blocks.iloc[-1]
            
            # Get block dates
            try:
                block_metadata = metadata_blocks[
                    metadata_blocks['block_id'] == this_athlete_last_block['block_id']
                ].iloc[0]
                end_date = block_metadata['pb_date']
                start_date = end_date - datetime.timedelta(days=91)
            except (KeyError, IndexError):
                logger.warning("Could not find block metadata, using current date")
                end_date = datetime.datetime.now()
                start_date = end_date - datetime.timedelta(days=91)
            
            # Get feature importance for vdot
            if 'y_name' in model_outputs.columns and len(model_outputs) > 0:
                features = model_outputs[model_outputs['y_name']=='y_vdot']
                features = features.sort_values(['importance'], ascending=[0])
            else:
                # If no model outputs, create features from feature_labels
                logger.info("No model outputs found, using all features with equal importance")
                features = pd.DataFrame({
                    'feature_name': [
                        'f_avg_weekly_run_distance',
                        'f_avg_weekly_run_time',
                        'f_avg_weekly_run_elevation',
                        'f_avg_weekly_athlete_count',
                        'f_avg_f_time_in_z1_runs',
                        'f_avg_f_time_in_z2_runs',
                        'f_avg_f_time_in_z3_runs',
                        'f_avg_f_time_in_z4_runs',
                        'f_avg_f_time_in_z5_runs',
                        'f_proportion_distance_activities',
                        'f_proportion_intense_activities',
                        'f_proportion_varying_activities'
                    ],
                    'importance': [1.0] * 12
                })
            
            logger.info(f"Processing features: {list(features['feature_name'])[:5]}")  # Show first 5 features
            
            # Get top and bottom performers
            if 'y_vdot' in features_blocks.columns:
                # Sort by vdot and ensure we have valid values
                valid_blocks = features_blocks[features_blocks['y_vdot'].notna()]
                if len(valid_blocks) > 0:
                    top_ten_percent = valid_blocks.sort_values(['y_vdot'], ascending=[False]).head(max(1, round(0.1*len(valid_blocks))))
                    bottom_ten_percent = valid_blocks.sort_values(['y_vdot'], ascending=[True]).head(max(1, round(0.1*len(valid_blocks))))
                else:
                    # If no valid vdot values, use empty DataFrames
                    top_ten_percent = features_blocks.head(0)
                    bottom_ten_percent = features_blocks.head(0)
            else:
                # If no y_vdot column, use empty DataFrames with same columns
                top_ten_percent = features_blocks.head(0)
                bottom_ten_percent = features_blocks.head(0)
            
            logger.info(f"Found {len(top_ten_percent)} blocks in top 10% and {len(bottom_ten_percent)} blocks in bottom 10%")
            
            visualisation_outputs = pd.DataFrame()
            processed_features = 0
            
            for index, feature in features.head(20).iterrows():
                feature_name = feature['feature_name']
                feature_importance = feature.get('importance', 1.0)
                
                # Skip if feature not in labels
                if feature_name not in feature_labels:
                    logger.debug(f"Skipping feature {feature_name} - not in labels")
                    continue
                    
                # Get athlete's score and percentiles with safe fallbacks
                try:
                    athlete_score = round(float(this_athlete_last_block.get(feature_name, 0.0)), 2)
                    logger.debug(f"Processing feature {feature_name} with score {athlete_score}")
                except (ValueError, TypeError):
                    athlete_score = 0.0
                    logger.debug(f"Error getting score for {feature_name}, using 0.0")
                    
                try:
                    top_ten_percent_value = top_ten_percent[feature_name].mean() if feature_name in top_ten_percent else 0.0
                    bottom_ten_percent_value = bottom_ten_percent[feature_name].mean() if feature_name in bottom_ten_percent else 0.0
                    logger.debug(f"Feature {feature_name} - top: {top_ten_percent_value}, bottom: {bottom_ten_percent_value}")
                except KeyError:
                    top_ten_percent_value = 0.0
                    bottom_ten_percent_value = 0.0
                    logger.debug(f"Error getting percentiles for {feature_name}, using 0.0")
                
                # Skip problematic features
                if feature_name in ["f_proportion_other", "r_proportion_other"]:
                    logger.debug(f"Skipping problematic feature {feature_name}")
                    continue
                    
                # Handle edge cases
                if top_ten_percent_value == 0.0 and bottom_ten_percent_value == 0.0:
                    athlete_percentile = 50  # Default to middle if no data
                elif math.isnan(athlete_score):
                    athlete_score = 0.0
                    athlete_percentile = 5  # Default to low if no score
                elif math.isnan(top_ten_percent_value) or math.isnan(bottom_ten_percent_value):
                    athlete_percentile = 50  # Default to middle if missing percentiles
                else:
                    # Calculate percentiles
                    if bottom_ten_percent_value > top_ten_percent_value:
                        perc_compare_top, perc_compare_bottom = bottom_ten_percent_value, top_ten_percent_value
                        
                        if athlete_score > perc_compare_bottom:
                            athlete_percentile = 5
                        elif athlete_score < perc_compare_top:
                            athlete_percentile = 95
                        else:
                            try:
                                athlete_percentile = 100*((((athlete_score - bottom_ten_percent_value) / 
                                                          (top_ten_percent_value - bottom_ten_percent_value))*0.8) + 0.1)
                            except ZeroDivisionError:
                                athlete_percentile = 50
                    else:
                        perc_compare_top, perc_compare_bottom = top_ten_percent_value, bottom_ten_percent_value
                        
                        if athlete_score < perc_compare_bottom:
                            athlete_percentile = 5
                        elif athlete_score > perc_compare_top:
                            athlete_percentile = 95
                        else:
                            try:
                                athlete_percentile = 100*((((athlete_score - bottom_ten_percent_value) / 
                                                          (top_ten_percent_value - bottom_ten_percent_value))*0.8) + 0.1)
                            except ZeroDivisionError:
                                athlete_percentile = 50
                
                athlete_need = feature_importance * (100-athlete_percentile)
                processed_features += 1
                
                visualisation_outputs = pd.concat([visualisation_outputs, pd.DataFrame([{
                    'feature_name': feature_name,
                    'feature_importance': feature_importance,
                    'athlete_score': athlete_score,
                    'athlete_percentile': athlete_percentile,
                    'athlete_need': athlete_need,
                    'tenth': 10,
                    'ninetieth': 90,
                    'one-hundredth': 100,
                    'value_at_tenth': round(bottom_ten_percent_value,2),
                    'value_at_ninetieth': round(top_ten_percent_value,2)
                }])], ignore_index=True)
            
            logger.info(f"Processed {processed_features} features out of {len(features)} total features")
            
            if len(visualisation_outputs) == 0:
                logger.error("No visualization data could be generated")
                return None
                
            visualisation_outputs = visualisation_outputs.sort_values(by=['athlete_need'],ascending=False)
            visualisation_outputs = visualisation_outputs.iloc[::-1]
            visualisation_outputs = visualisation_outputs.reset_index()
            
            # Create visualization
            plt.style.use('default')  # Using default style instead of seaborn
            plt.title("Your performance relative to the best athletes \n For 3 months before your last PB, between "
                      + str(start_date.date()) + " and " + str(end_date.date())
                      + "\nOrdered by how much each aspect would help your fitness")

            fig = plt.figure(figsize=(12, 12.5))
            
            labels = []
            for index, feature in visualisation_outputs.iterrows():
                labels.append(feature_labels[feature['feature_name']] + "\n" + str(feature['athlete_score']))
                
            ax = fig.add_subplot(111)
            
            ax.barh(labels, visualisation_outputs['one-hundredth'], tick_label=labels, height=0.8, color='#afffd3')
            ax.barh(labels, visualisation_outputs['ninetieth'], tick_label=labels, height=0.8, color='#bbbbc1')
            ax.barh(labels, visualisation_outputs['tenth'], tick_label=labels, height=0.8, color='#ffa4a4')    

            ax.plot(visualisation_outputs['athlete_percentile'], labels, marker=10, markersize=15, linestyle="", label=visualisation_outputs['athlete_score'])

            for index, feature in visualisation_outputs.iterrows():
                ax.text(x=float(11), y=index, s=feature['value_at_tenth'], horizontalalignment="left")
                ax.text(x=float(89), y=index, s=feature['value_at_ninetieth'], horizontalalignment="right")
                
            plt.xlabel('Percentile. 0% = the worst performing athlete. 100% = the best performing athlete.')    
            plt.tight_layout()
            
            bytes_image = io.BytesIO()
            plt.savefig(bytes_image, format='png')
            bytes_image.seek(0)
            plt.clf()
            plt.cla()
            plt.close()
            return bytes_image
            
    except Exception as e:
        logger.error(f"Error generating athlete vs best visualization: {e}")
        return None

def athletevsbestimprovement(athlete_id):
    """Generate visualization comparing athlete's improvement to others."""
    try:
        athlete_id = str(athlete_id)
        logger.info(f"Starting improvement visualization for athlete {athlete_id}")
        
        with current_app.app_context():
            engine = get_db_connection()
            
            # Get features blocks for this athlete
            features_blocks = pd.read_sql(
                "SELECT * FROM features_blocks",
                engine
            )
            logger.info(f"Found {len(features_blocks)} total feature blocks")
            logger.info(f"Features blocks columns: {features_blocks.columns.tolist()}")
            logger.info(f"Sample athlete_ids in features_blocks: {features_blocks['athlete_id'].head().tolist()}")
            logger.info(f"athlete_id type in features_blocks: {features_blocks['athlete_id'].dtype}")
            
            # Convert athlete_id column to string for comparison
            features_blocks['athlete_id'] = features_blocks['athlete_id'].astype(str)
            
            if len(features_blocks) == 0:
                logger.error(f"No feature blocks found in database")
                return None
                
            # Get metadata blocks
            metadata_blocks = pd.read_sql(
                "SELECT * FROM metadata_blocks",
                engine
            )
            logger.info(f"Found {len(metadata_blocks)} metadata blocks")
            
            # Get model outputs (this can be empty, we'll handle it)
            try:
                model_outputs = pd.read_sql(
                    "SELECT * FROM model_outputs",
                    engine
                )
            except:
                logger.warning("Could not read model_outputs table, using default values")
                model_outputs = pd.DataFrame(columns=['y_name', 'feature_name', 'importance'])
            
            logger.info(f"Found {len(model_outputs)} model outputs")
            
            # Get this athlete's last block
            this_athlete_blocks = features_blocks[features_blocks['athlete_id']==athlete_id]
            logger.info(f"Found {len(this_athlete_blocks)} blocks for athlete {athlete_id}")
            if len(this_athlete_blocks) > 0:
                logger.info(f"Sample of athlete's block data: {this_athlete_blocks.iloc[-1].to_dict()}")
            
            if len(this_athlete_blocks) == 0:
                logger.warning(f"No blocks found for athlete {athlete_id}, using empty values")
                this_athlete_last_block = pd.Series(0.0, index=features_blocks.columns)
            else:
                this_athlete_last_block = this_athlete_blocks.iloc[-1]
            
            # Get block dates
            try:
                block_metadata = metadata_blocks[
                    metadata_blocks['block_id'] == this_athlete_last_block['block_id']
                ].iloc[0]
                end_date = block_metadata['pb_date']
                start_date = end_date - datetime.timedelta(days=91)
            except (KeyError, IndexError):
                logger.warning("Could not find block metadata, using current date")
                end_date = datetime.datetime.now()
                start_date = end_date - datetime.timedelta(days=91)
            
            # Get feature importance for vdot_delta
            if 'y_name' in model_outputs.columns and len(model_outputs) > 0:
                features = model_outputs[model_outputs['y_name']=='y_vdot_delta']
                features = features.sort_values(['importance'], ascending=[0])
            else:
                # If no model outputs, create features from feature_labels
                logger.info("No model outputs found, using all features with equal importance")
                features = pd.DataFrame({
                    'feature_name': [
                        'f_avg_weekly_run_distance',
                        'f_avg_weekly_run_time',
                        'f_avg_weekly_run_elevation',
                        'f_avg_weekly_athlete_count',
                        'f_avg_f_time_in_z1_runs',
                        'f_avg_f_time_in_z2_runs',
                        'f_avg_f_time_in_z3_runs',
                        'f_avg_f_time_in_z4_runs',
                        'f_avg_f_time_in_z5_runs',
                        'f_proportion_distance_activities',
                        'f_proportion_intense_activities',
                        'f_proportion_varying_activities'
                    ],
                    'importance': [1.0] * 12
                })
            
            logger.info(f"Processing features: {list(features['feature_name'])[:5]}")  # Show first 5 features
            
            # Get top and bottom improvers
            if 'y_vdot_delta' in features_blocks.columns:
                # Sort by vdot_delta and ensure we have valid values
                valid_blocks = features_blocks[features_blocks['y_vdot_delta'].notna()]
                if len(valid_blocks) > 0:
                    top_ten_percent = valid_blocks.sort_values(['y_vdot_delta'], ascending=[False]).head(max(1, round(0.1*len(valid_blocks))))
                    bottom_ten_percent = valid_blocks.sort_values(['y_vdot_delta'], ascending=[True]).head(max(1, round(0.1*len(valid_blocks))))
                else:
                    # If no valid vdot_delta values, use empty DataFrames
                    top_ten_percent = features_blocks.head(0)
                    bottom_ten_percent = features_blocks.head(0)
            else:
                # If no y_vdot_delta column, use empty DataFrames with same columns
                top_ten_percent = features_blocks.head(0)
                bottom_ten_percent = features_blocks.head(0)
            
            logger.info(f"Found {len(top_ten_percent)} blocks in top 10% and {len(bottom_ten_percent)} blocks in bottom 10%")
            
            visualisation_outputs = pd.DataFrame()
            
            for index, feature in features.head(20).iterrows():
                feature_name = feature['feature_name']
                feature_importance = feature.get('importance', 1.0)
                
                # Skip if feature not in labels
                if feature_name not in feature_labels:
                    continue
                    
                # Get athlete's score and percentiles with safe fallbacks
                try:
                    athlete_score = round(float(this_athlete_last_block.get(feature_name, 0.0)), 2)
                except (ValueError, TypeError):
                    athlete_score = 0.0
                    
                try:
                    top_ten_percent_value = top_ten_percent[feature_name].mean() if feature_name in top_ten_percent else 0.0
                except KeyError:
                    top_ten_percent_value = 0.0
                    
                try:
                    bottom_ten_percent_value = bottom_ten_percent[feature_name].mean() if feature_name in bottom_ten_percent else 0.0
                except KeyError:
                    bottom_ten_percent_value = 0.0
                
                # Skip problematic features
                if feature_name in ["f_proportion_other", "r_proportion_other"]:
                    continue
                    
                # Handle edge cases
                if top_ten_percent_value == 0.0 and bottom_ten_percent_value == 0.0:
                    athlete_percentile = 50  # Default to middle if no data
                elif math.isnan(athlete_score):
                    athlete_score = 0.0
                    athlete_percentile = 5  # Default to low if no score
                elif math.isnan(top_ten_percent_value) or math.isnan(bottom_ten_percent_value):
                    athlete_percentile = 50  # Default to middle if missing percentiles
                else:
                    # Calculate percentiles
                    if bottom_ten_percent_value > top_ten_percent_value:
                        perc_compare_top, perc_compare_bottom = bottom_ten_percent_value, top_ten_percent_value
                        
                        if athlete_score > perc_compare_bottom:
                            athlete_percentile = 5
                        elif athlete_score < perc_compare_top:
                            athlete_percentile = 95
                        else:
                            try:
                                athlete_percentile = 100*((((athlete_score - bottom_ten_percent_value) / 
                                                          (top_ten_percent_value - bottom_ten_percent_value))*0.8) + 0.1)
                            except ZeroDivisionError:
                                athlete_percentile = 50
                    else:
                        perc_compare_top, perc_compare_bottom = top_ten_percent_value, bottom_ten_percent_value
                        
                        if athlete_score < perc_compare_bottom:
                            athlete_percentile = 5
                        elif athlete_score > perc_compare_top:
                            athlete_percentile = 95
                        else:
                            try:
                                athlete_percentile = 100*((((athlete_score - bottom_ten_percent_value) / 
                                                          (top_ten_percent_value - bottom_ten_percent_value))*0.8) + 0.1)
                            except ZeroDivisionError:
                                athlete_percentile = 50
                
                athlete_need = feature_importance * (100-athlete_percentile)
                
                visualisation_outputs = pd.concat([visualisation_outputs, pd.DataFrame([{
                    'feature_name': feature_name,
                    'feature_importance': feature_importance,
                    'athlete_score': athlete_score,
                    'athlete_percentile': athlete_percentile,
                    'athlete_need': athlete_need,
                    'tenth': 10,
                    'ninetieth': 90,
                    'one-hundredth': 100,
                    'value_at_tenth': round(bottom_ten_percent_value,2),
                    'value_at_ninetieth': round(top_ten_percent_value,2)
                }])], ignore_index=True)
            
            if len(visualisation_outputs) == 0:
                logger.error("No visualization data could be generated")
                return None
                
            visualisation_outputs = visualisation_outputs.sort_values(by=['athlete_need'],ascending=False)
            visualisation_outputs = visualisation_outputs.iloc[::-1]
            visualisation_outputs = visualisation_outputs.reset_index()
            
            # Create visualization
            plt.style.use('default')  # Using default style instead of seaborn
            plt.title("Your ability, relative to others, to IMPROVE your fitness \n For 3 months before your last PB, between "
                      + str(start_date.date()) + " and " + str(end_date.date())
                      + "\nOrdered by how much each item will help you improve")

            fig = plt.figure(figsize=(12, 12.5))
            
            labels = []
            for index, feature in visualisation_outputs.iterrows():
                labels.append(feature_labels[feature['feature_name']] + "\n" + str(feature['athlete_score']))
                
            ax = fig.add_subplot(111)
            
            ax.barh(labels, visualisation_outputs['one-hundredth'], tick_label=labels, height=0.8, color='#afffd3')
            ax.barh(labels, visualisation_outputs['ninetieth'], tick_label=labels, height=0.8, color='#bbbbc1')
            ax.barh(labels, visualisation_outputs['tenth'], tick_label=labels, height=0.8, color='#ffa4a4')    

            ax.plot(visualisation_outputs['athlete_percentile'], labels, marker=10, markersize=15, linestyle="", label=visualisation_outputs['athlete_score'])

            for index, feature in visualisation_outputs.iterrows():
                ax.text(x=float(11), y=index, s=feature['value_at_tenth'], horizontalalignment="left")
                ax.text(x=float(89), y=index, s=feature['value_at_ninetieth'], horizontalalignment="right")
                
            plt.xlabel('Percentile. 0% = the least improvement. 100% = the best improvement.')    
            plt.tight_layout()
            
            bytes_image = io.BytesIO()
            plt.savefig(bytes_image, format='png')
            bytes_image.seek(0)
            plt.clf()
            plt.cla()
            plt.close()
            return bytes_image
            
    except Exception as e:
        logger.error(f"Error generating athlete vs best improvement visualization: {e}")
        return None

def double_to_hours_minutes(time):
    hours = int(math.floor(time))
    minutes = int(round(60*(time - hours),0))
    return str(datetime.time(hours, minutes, 0, 0))