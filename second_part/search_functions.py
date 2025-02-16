from datetime import datetime, timedelta
import math
from itertools import dropwhile
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)

def get_activity(activities: List[Dict], activity_id: int) -> Optional[Dict]:
    """Find an activity by its ID in a list of activities."""
    try:
        return next(
            (activity for activity in activities if int(activity['id']) == activity_id),
            None
        )
    except Exception as e:
        logger.error(f"Error finding activity {activity_id}: {e}")
        return None

def is_valid_activity(activity: Dict) -> bool:
    """Check if an activity has valid essential data."""
    try:
        activity['type']
        return True
    except (KeyError, TypeError):
        return False

def get_block(
    activities: List[Dict], 
    activity_date: datetime, 
    duration_days: int = 91
) -> List[Dict]:
    """
    Get a block of activities leading up to a specific date.
    
    Args:
        activities: List of activities in chronological order
        activity_date: Target date to work backwards from
        duration_days: Number of days to look back (default 91 days / 3 months)
    
    Returns:
        List of activities within the specified time period
    """
    block_activities = []
    
    for activity in activities:
        if not is_valid_activity(activity):
            continue
            
        try:
            current_date = datetime.strptime(
                activity['start_date'][:10], 
                '%Y-%m-%d'
            )
            time_diff = (activity_date - current_date).days
            
            if 0 <= time_diff < duration_days:
                block_activities.append(activity)
                
        except Exception as e:
            logger.debug(f"Error processing activity date: {e}")
            continue

    return block_activities

def get_weeks(
    block_activities: List[Dict], 
    duration_days: int = 91
) -> List[List[Dict]]:
    """
    Split activities into weeks.
    
    Args:
        block_activities: List of activities in chronological order
        duration_days: Optional duration to consider (0 means use full range)
    
    Returns:
        List of lists, where each inner list contains activities for one week
    """
    try:
        if not block_activities:
            return []
            
        # Get date range
        end_date = datetime.strptime(block_activities[-1]['start_date'][:10], '%Y-%m-%d')
        
        if duration_days > 0:
            # Use specified duration
            start_date = end_date - timedelta(days=duration_days-1)
        else:
            # Use full range of activities
            start_date = datetime.strptime(block_activities[0]['start_date'][:10], '%Y-%m-%d')
            duration_days = (end_date - start_date).days
        
        # Initialize weeks
        num_weeks = int(duration_days/7)
        weeks = [[] for _ in range(num_weeks)]
        
        # Distribute activities into weeks
        for activity in block_activities[:-1]:  # Exclude last activity (PB)
            try:
                activity_date = datetime.strptime(activity['start_date'][:10], '%Y-%m-%d')
                current_week = math.floor((activity_date - start_date).days / 7)
                
                if 0 <= current_week < num_weeks:
                    weeks[current_week].append(activity)
            except Exception as e:
                logger.debug(f"Error assigning activity to week: {e}")
                continue
        
        # Remove empty weeks from the start
        return list(dropwhile(lambda x: not x, weeks))
        
    except Exception as e:
        logger.error(f"Error processing weeks: {e}")
        return []