# Analysis of different papers related to this project

## Master's Thesis Maurer


## 1. Introduction

**Motivation:** Running is widely popular, with millions using platforms like Strava to track their activities. Most available race time predictors use basic formulas derived from elite runners and neglect individual athlete specifics and elevation profiles.

**Research Goals:** The thesis aims to develop more accurate methods for predicting race times using historical training data, especially taking into account elevation changes. Two main approaches are proposed: an activity-based model and a segment-based model.

## 2. Related Work

### 2.1 Power models, scoring tables, and other formulas

- **Riegel (1981):** Developed a simple power formula \( t = ax^b \) for predicting race times based on historical data of elite athletes. Found fatigue factor \( b \) generally around 1.08.
- **Vickers et al. (2016):** Improved on Riegel's formula specifically for recreational marathon runners by adding factors like weekly mileage, significantly reducing prediction errors for non-elite runners.
- **Gardner & Purdy (1970)** and **Purdy (1974):** Developed scoring tables and equations to standardize race times across different distances using world record data.
- **Elliott (2012):** Included elevation change in marathon pace predictions by analyzing energy cost variations on hilly marathon courses, refining previous flat-course analyses.
- **Minetti et al. (2002):** Investigated the energy costs of running on extreme slopes and defined an energy cost function based on slope gradients.
- **Townshend et al. (2010):** Studied pacing strategies over undulating terrain, recommending slightly slower uphill pacing and predicting speed from gradient data using regression.

### Complex Models Using Historical Parameters

- **Tanda (2011):** Explored marathon performance prediction using weekly mileage and average training pace, achieving good accuracy with nonlinear regression.
- **Ruiz-Mayo et al. (2016):** Utilized comprehensive historical training session parameters (e.g., longest distance, elevation gain, fastest pace) and applied machine learning, achieving mean absolute error (MAE) ~7-9 minutes on marathon predictions.
- **Jin (2014):** Used linear regression based on distance, elevation gain, and recent 10k race times with limited data (4 athletes, 447 activities), achieving low errors but questionable generalizability.
- **Millett et al. (2015):** Compared locally weighted linear regression and Hidden Markov Model (HMM) on historical race data. HMM showed better general results (~3.8% error), slightly outperforming Riegel’s formula and coach predictions.
- **Blythe et al. (2017):** Employed Local Matrix Completion (LMC) to model runners’ individual performance using power-law models, achieving excellent results on marathon data (3.6 min MAE).
- **Smyth & Cunningham (2017):** Developed a recommender system using Case-Based Reasoning (CBR) to suggest optimal pacing plans based on similar past performances, achieving high accuracy and similarity in pacing strategies.

## 2.3 Comparison to Thesis Approach:

- Thesis focuses on individualized predictions rather than elite athlete-based predictions and explicitly incorporates elevation data.
- The activity-based model resembles Elliott's method, incorporating features like elevation gain and hilliness.
- The segment-based approach divides activities into gradient segments, providing a detailed pacing plan. This is comparable to Smyth & Cunningham's recommender system but without needing the athlete to have previously completed the same course.

## Survey (Chapter 3)

Conducted an online survey with 330 runners, examining running habits, race preparation, and problems encountered in races. Most participants (~64%) did not have a coach and had problems with pacing, especially starting too fast. They indicated interest in more precise predictors that account for individual training data and environmental conditions (weather, sleep quality, personal motivation, etc.).

## Implementation (Chapter 4)

### Historical training data

- Used Strava API for activity data; due to inaccurate elevation data, replaced it with Google Elevation API data.
- Applied Ramer-Douglas-Peucker algorithm for smoothing elevation profiles.
- Implemented Training Stress Balance (TSB) to quantify fatigue (ATL) and fitness (CTL).
- Computed Functional Threshold Pace (FTP) and Normalized Graded Pace (NGP) for accurate training stress measurement.

### Neural Network (NN)

- Implemented a regression NN using TensorFlow with one hidden layer (5 nodes), ELU activation, and regularization methods (Elastic Net, Dropout).
- Activity-based features included distance, elevation gain, hilliness, climb score, ATL, CTL, average VO2max (6-week window), and a race indicator.

### Activity-based Approach

Uses activities as whole units, achieving an evaluation accuracy of 91.28% (RMSE ~6:45 min).

### Segment-based Approach

Divides activities into gradient-based segments, predicting times for each segment, then reconstructing to overall predictions. Segment features included activity distance, elevation gain, segment-specific distance and gradient, and position within the activity. Achieved an accuracy of 89.82% and similarity (segment accuracy) of 76.41%.

## User Study (Chapter 5)

Conducted real-world evaluation with 12 runners on a hilly 10 km race (~150 m elevation gain). Riegel's formula baseline achieved 96.14% accuracy, which was unexpectedly good. Activity-based NN prediction: 92.48% accuracy (3:16 min MAE). Segment-based NN model achieved 88.2% accuracy and 81.54% similarity. Refined subset (minimum 4 races and 20 km weekly mileage) improved activity-based accuracy to 95.25% and segment-based similarity to 87.82%.

## Conclusion (Chapter 6)

Activity-based approach successfully captures elevation effects and fitness state, achieving solid predictions (~91-95% accuracy). Segment-based approach provided pacing plans but was sensitive to inaccurate or anomalous data, achieving lower accuracy (~88%) but promising similarities (~82-88%). Neither approach consistently outperformed Riegel's formula in the user study but showed high potential and provided valuable individualized predictions. Future work suggested includes improving elevation data accuracy, adding subjective performance ratings, and incorporating additional features (weather, sleep, motivation). Potential also identified for combining general models (like Riegel’s) with individual training data through model fine-tuning. Overall, the thesis made significant progress in personalized race time predictions using historical training data, especially by integrating elevation profiles and training load metrics, paving the way for more precise and individualized performance modeling.