from sql_methods import db

class ProcessingStatus(db.Model):
    __tablename__ = 'processing_status'
    
    athlete_id = db.Column(db.String(100), primary_key=True)
    status = db.Column(db.String(50))
    bearer_token = db.Column(db.String(255))
    refresh_token = db.Column(db.String(255))

    def __repr__(self):
        return f'<ProcessingStatus {self.athlete_id}>'

class AthleteStats(db.Model):
    __tablename__ = 'athlete_stats'
    
    athlete_id = db.Column(db.String(100), primary_key=True)
    recent_run_totals = db.Column(db.JSON)
    all_run_totals = db.Column(db.JSON)
    all_ride_totals = db.Column(db.JSON)
    
    def __repr__(self):
        return f'<AthleteStats {self.athlete_id}>'

class Activity(db.Model):
    __tablename__ = 'activities'
    
    id = db.Column(db.BigInteger, primary_key=True)
    athlete_id = db.Column(db.String(100))
    name = db.Column(db.String(255))
    distance = db.Column(db.Float)
    moving_time = db.Column(db.Integer)
    elapsed_time = db.Column(db.Integer)
    total_elevation_gain = db.Column(db.Float)
    type = db.Column(db.String(50))
    start_date = db.Column(db.DateTime)
    average_speed = db.Column(db.Float)
    max_speed = db.Column(db.Float)
    average_heartrate = db.Column(db.Float)
    max_heartrate = db.Column(db.Float)
    activity_data = db.Column(db.JSON)  # Full raw JSON from Strava
    suffer_score = db.Column(db.Float, nullable=True)
    map_data = db.Column(db.JSON, nullable=True)    
    laps_data = db.Column(db.JSON, nullable=True)     

    # Best effort columns (store, for example, the elapsed_time of the effort)
    be_400m = db.Column(db.Integer, nullable=True)
    be_half_mile = db.Column(db.Integer, nullable=True)
    be_1km = db.Column(db.Integer, nullable=True)
    be_1_mile = db.Column(db.Integer, nullable=True)
    be_2_miles = db.Column(db.Integer, nullable=True)
    be_5km = db.Column(db.Integer, nullable=True)
    be_10km = db.Column(db.Integer, nullable=True)
    be_15km = db.Column(db.Integer, nullable=True)
    be_10_miles = db.Column(db.Integer, nullable=True)
    be_20km = db.Column(db.Integer, nullable=True)
    be_half_marathon = db.Column(db.Integer, nullable=True)
    be_30km = db.Column(db.Integer, nullable=True)
    be_marathon = db.Column(db.Integer, nullable=True)
    be_50km = db.Column(db.Integer, nullable=True)

    def __repr__(self):
        return f'<Activity {self.id}>'

class FeaturesBlock(db.Model):
    __tablename__ = 'features_blocks'
    
    id = db.Column(db.Integer, primary_key=True)
    athlete_id = db.Column(db.String(100))
    block_id = db.Column(db.String(100))
    y_vdot_delta = db.Column(db.Float)
    y_vdot = db.Column(db.Float)
    f_slope_run_distance = db.Column(db.Float)
    f_slope_run_time = db.Column(db.Float)
    f_slope_mean_run_hr = db.Column(db.Float)
    f_taper_factor_run_distance = db.Column(db.Float)
    f_taper_factor_run_time = db.Column(db.Float)
    f_taper_factor_mean_run_hr = db.Column(db.Float)

    def __repr__(self):
        return f'<FeaturesBlock {self.block_id}>'

class MetadataBlock(db.Model):
    __tablename__ = 'metadata_blocks'
    
    id = db.Column(db.Integer, primary_key=True)
    athlete_id = db.Column(db.String(100))
    vdot = db.Column(db.Float)
    vdot_delta = db.Column(db.Float)
    predicted_marathon_time = db.Column(db.Float)
    pb_date = db.Column(db.DateTime)
    block_id = db.Column(db.String(100))

    def __repr__(self):
        return f'<MetadataBlock {self.block_id}>'

class ModelOutput(db.Model):
    __tablename__ = 'model_outputs'
    
    id = db.Column(db.Integer, primary_key=True)
    y_name = db.Column(db.String(100))
    feature_name = db.Column(db.String(100))
    importance = db.Column(db.Float)

    def __repr__(self):
        return f'<ModelOutput {self.y_name}_{self.feature_name}>'
    
class MetadataAthlete(db.Model):
    __tablename__ = 'metadata_athletes'
    id = db.Column(db.String(100), primary_key=True)
    sex = db.Column(db.String(50))
    weight = db.Column(db.Float)
    zones = db.Column(db.String(255))  # You could also store this as JSON if preferred

    def __repr__(self):
        return f'<MetadataAthlete {self.id}>'

class AllAthleteActivities(db.Model):
    __tablename__ = 'all_athlete_activities'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    athlete_id = db.Column(db.String(100))
    block_id = db.Column(db.String(100))
    week_id = db.Column(db.String(100))
    activity_type = db.Column(db.Integer)
    activity_id = db.Column(db.String(100))
    elapsed_time = db.Column(db.Float)
    distance = db.Column(db.Float)
    mean_hr = db.Column(db.Float)
    stdev_hr = db.Column(db.Float)
    freq_hr = db.Column(db.Float)
    time_in_z1 = db.Column(db.Float)
    time_in_z2 = db.Column(db.Float)
    time_in_z3 = db.Column(db.Float)
    time_in_z4 = db.Column(db.Float)
    time_in_z5 = db.Column(db.Float)
    elevation = db.Column(db.Float)
    stdev_elevation = db.Column(db.Float)
    freq_elevation = db.Column(db.Float)
    pace = db.Column(db.Float)
    stdev_pace = db.Column(db.Float)
    freq_pace = db.Column(db.Float)
    cadence = db.Column(db.Float)
    athlete_count = db.Column(db.Float)

    def __repr__(self):
        return f'<AllAthleteActivities {self.id} - {self.athlete_id}>'

class AllAthleteWeeks(db.Model):
    __tablename__ = 'all_athlete_weeks'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    athlete_id = db.Column(db.String(100))
    block_id = db.Column(db.String(100))
    week_id = db.Column(db.String(100))
    f_total_runs = db.Column(db.Integer)
    f_total_run_distance = db.Column(db.Float)
    f_total_run_time = db.Column(db.Float)
    f_total_non_run_distance = db.Column(db.Float)
    f_total_non_run_time = db.Column(db.Float)

    def __repr__(self):
        return f'<AllAthleteWeeks {self.id} - {self.athlete_id}>'

class FeaturesActivities(db.Model):
    __tablename__ = 'features_activities'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    athlete_id = db.Column(db.String(100))
    block_id = db.Column(db.String(100))
    week_id = db.Column(db.String(100))
    activity_type = db.Column(db.Integer)
    activity_id = db.Column(db.String(100))
    elapsed_time = db.Column(db.Float)
    distance = db.Column(db.Float)
    mean_hr = db.Column(db.Float)

    def __repr__(self):
        return f'<FeaturesActivities {self.id} - {self.athlete_id}>'

class FeaturesWeeks(db.Model):
    __tablename__ = 'features_weeks'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    athlete_id = db.Column(db.String(100))
    block_id = db.Column(db.String(100))
    week_id = db.Column(db.String(100))
    f_total_runs = db.Column(db.Integer)
    f_run_distance = db.Column(db.Float)
    f_run_time = db.Column(db.Float)
    f_non_run_distance = db.Column(db.Float)
    f_non_run_time = db.Column(db.Float)

    def __repr__(self):
        return f'<FeaturesWeeks {self.id} - {self.athlete_id}>'

class AveragePacesAndHrs(db.Model):
    __tablename__ = 'average_paces_and_hrs'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    athlete_id = db.Column(db.String(100))
    mean_hr = db.Column(db.Float)
    pace = db.Column(db.Float)

    def __repr__(self):
        return f'<AveragePacesAndHrs {self.id} - {self.athlete_id}>'
    

class MetadataPB(db.Model):
    __tablename__ = 'metadata_pbs'
    
    id = db.Column(db.Integer, primary_key=True)
    athlete_id = db.Column(db.String(100), nullable=False)
    distance_category = db.Column(db.String(50), nullable=False)  # e.g., "5K", "10K"
    elapsed_time = db.Column(db.Integer)  # time in seconds for the PB effort
    distance = db.Column(db.Float)          # distance in meters for the effort
    pr_rank = db.Column(db.Integer)         # personal best rank (1, 2, or 3)
    start_date = db.Column(db.DateTime)       # when the PB was set
    activity_id = db.Column(db.String(100))    # the activity ID associated with this PB
    pb_data = db.Column(db.JSON)  # (Optional) store the full best effort JSON

    def __repr__(self):
        return (f'<MetadataPB {self.athlete_id} {self.distance_category} PB: '
                f'{self.elapsed_time}s, rank {self.pr_rank}>')
