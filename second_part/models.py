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

    # Best efforts columns (kept as-is)
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

    # New columns to capture additional JSON data:
    resource_state = db.Column(db.Integer, nullable=True)
    sport_type = db.Column(db.String(50), nullable=True)
    workout_type = db.Column(db.Integer, nullable=True)
    start_date_local = db.Column(db.DateTime, nullable=True)
    timezone = db.Column(db.String(100), nullable=True)
    utc_offset = db.Column(db.Float, nullable=True)
    location_city = db.Column(db.String(100), nullable=True)
    location_state = db.Column(db.String(100), nullable=True)
    location_country = db.Column(db.String(100), nullable=True)
    achievement_count = db.Column(db.Integer, nullable=True)
    kudos_count = db.Column(db.Integer, nullable=True)
    comment_count = db.Column(db.Integer, nullable=True)
    athlete_count = db.Column(db.Integer, nullable=True)
    photo_count = db.Column(db.Integer, nullable=True)
    trainer = db.Column(db.Boolean, nullable=True)
    commute = db.Column(db.Boolean, nullable=True)
    manual = db.Column(db.Boolean, nullable=True)
    private = db.Column(db.Boolean, nullable=True)
    visibility = db.Column(db.String(50), nullable=True)
    flagged = db.Column(db.Boolean, nullable=True)
    gear_id = db.Column(db.String(100), nullable=True)
    start_latlng = db.Column(db.JSON, nullable=True)
    end_latlng = db.Column(db.JSON, nullable=True)
    average_cadence = db.Column(db.Float, nullable=True)
    average_temp = db.Column(db.Float, nullable=True)
    average_watts = db.Column(db.Float, nullable=True)
    max_watts = db.Column(db.Float, nullable=True)
    weighted_average_watts = db.Column(db.Float, nullable=True)
    device_watts = db.Column(db.Boolean, nullable=True)
    kilojoules = db.Column(db.Float, nullable=True)
    has_heartrate = db.Column(db.Boolean, nullable=True)
    heartrate_opt_out = db.Column(db.Boolean, nullable=True)
    display_hide_heartrate_option = db.Column(db.Boolean, nullable=True)
    elev_high = db.Column(db.Float, nullable=True)
    elev_low = db.Column(db.Float, nullable=True)
    upload_id = db.Column(db.BigInteger, nullable=True)
    upload_id_str = db.Column(db.String(50), nullable=True)
    external_id = db.Column(db.String(100), nullable=True)
    from_accepted_tag = db.Column(db.Boolean, nullable=True)
    pr_count = db.Column(db.Integer, nullable=True)
    total_photo_count = db.Column(db.Integer, nullable=True)
    has_kudoed = db.Column(db.Boolean, nullable=True)
    description = db.Column(db.Text, nullable=True)
    calories = db.Column(db.Float, nullable=True)
    perceived_exertion = db.Column(db.Float, nullable=True)
    prefer_perceived_exertion = db.Column(db.Boolean, nullable=True)
    device_name = db.Column(db.String(100), nullable=True)
    embed_token = db.Column(db.String(255), nullable=True)
    private_note = db.Column(db.Text, nullable=True)
    similar_activities = db.Column(db.JSON, nullable=True)
    available_zones = db.Column(db.JSON, nullable=True)
    splits_metric = db.Column(db.JSON, nullable=True)
    splits_standard = db.Column(db.JSON, nullable=True)
    laps = db.Column(db.JSON, nullable=True)
    photos = db.Column(db.JSON, nullable=True)
    stats_visibility = db.Column(db.JSON, nullable=True)
    hide_from_home = db.Column(db.Boolean, nullable=True)

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

    # Add a unique constraint for athlete_id and distance_category
    __table_args__ = (
        db.UniqueConstraint('athlete_id', 'distance_category', name='uix_athlete_id_distance_category'),
    )

    def __repr__(self):
        return (f'<MetadataPB {self.athlete_id} {self.distance_category} PB: '
                f'{self.elapsed_time}s, rank {self.pr_rank}>')

class RacePrediction(db.Model):
    __tablename__ = 'race_predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    athlete_id = db.Column(db.String(100), nullable=False)
    riegel_exponent = db.Column(db.Float, nullable=False)  # Calculated personalized exponent
    best_distance = db.Column(db.Float, nullable=False)    # Base distance in meters
    best_time = db.Column(db.Integer, nullable=False)      # Base time in seconds
    created_at = db.Column(db.DateTime, nullable=False)    # When prediction was calculated
    
    def __repr__(self):
        return f'<RacePrediction {self.id} for athlete {self.athlete_id}>'
