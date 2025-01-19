# Running Companion

A personalized running companion application that integrates with Strava to provide predictive analytics and insights for runners. Hosted entirely on AWS, it offers a seamless experience to track, analyze, and enhance running performance.

## Features
- **User Authentication:** Secure user account creation and login.
- **Strava Integration:** Link your Strava account to import running activity data.
- **Race Predictions:** Predict finishing times for common race distances (5K, 10K, Half Marathon, Marathon).
- **Interactive Dashboard:** View personalized insights, pace trends, and mileage charts.

## Tech Stack
- **Backend:** Python (FastAPI/Django/Flask).
- **Frontend:** React or Vue.js.
- **Database:** PostgreSQL for structured data, S3 for raw data storage.
- **Model Serving:** AWS Lambda/SageMaker.
- **Hosting:** AWS (EC2, API Gateway, Cognito).

## Setup Instructions
### Prerequisites
- Python 3.12
- Node.js
- AWS CLI
- Docker (optional for deployment)

### Clone the Repository
```bash
git clone https://github.com/FdiLmr/running-companion.git
cd running-companion
```

### Backend Setup
1. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the FastAPI server:
   ```bash
   uvicorn app.main:app --reload
   ```

### Frontend Setup
1. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm start
   ```

### Run Tests
- Backend:
  ```bash
  pytest
  ```
- Linting:
  ```bash
  flake8 .
  ```

## Deployment
1. Configure AWS CLI:
   ```bash
   aws configure
   ```
2. Deploy the backend using Docker or AWS Lambda.
3. Deploy the frontend to an S3 bucket or AWS Amplify.

## Contributing
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.