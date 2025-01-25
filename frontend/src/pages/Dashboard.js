import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { getUserProfile, logout } from '../services/auth';
import { stravaService } from '../services/strava';
import './dashboard.css';

function Dashboard() {
  const [userData, setUserData] = useState(null);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [isConnecting, setIsConnecting] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const data = await getUserProfile();
        setUserData(data);
      } catch (err) {
        setError(err.message);
        if (err.message === 'No id token found' || err.message.includes('Invalid token')) {
          navigate('/login');
        }
      } finally {
        setIsLoading(false);
      }
    };

    fetchUserData();
  }, [navigate]);

  // Handle Strava callback
  useEffect(() => {
    const handleStravaCallback = async () => {
      const searchParams = new URLSearchParams(location.search);
      const code = searchParams.get('code');
      const error = searchParams.get('error');
      
      // Clear the URL parameters immediately to prevent double processing
      window.history.replaceState({}, document.title, window.location.pathname);

      if (error) {
        setError('Failed to connect to Strava: ' + error);
        return;
      }

      if (code && !isConnecting) {  
        setIsConnecting(true);
        try {
          await stravaService.handleStravaCallback(code);
          // Refresh user data to get updated Strava status
          const data = await getUserProfile();
          setUserData(data);
        } catch (err) {
          setError('Failed to complete Strava connection: ' + err.message);
        } finally {
          setIsConnecting(false);
        }
      }
    };

    if (location.search) {  
      handleStravaCallback();
    }
  }, [location.search]);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const handleStravaConnect = async () => {
    try {
      const response = await stravaService.getStravaConnectUrl();
      if (response && response.auth_url) {
        window.location.href = response.auth_url;
      } else {
        setError('Invalid response from server');
      }
    } catch (err) {
      setError('Failed to initiate Strava connection. Please try again.');
    }
  };

  if (isLoading) {
    return (
      <div className="dashboard-page">
        <div className="dashboard-container">
          <p>Loading...</p>
        </div>
      </div>
    );
  }

  if (error && error !== 'No id token found' && !error.includes('Invalid token')) {
    return (
      <div className="dashboard-page">
        <div className="dashboard-container">
          <div className="error-message" role="alert">{error}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-page">
      <header className="dashboard-header">
        <h1>Running Companion</h1>
        <div className="header-actions">
          {userData && <span>Welcome, {userData.username}!</span>}
          <button className="logout-button" onClick={handleLogout}>Logout</button>
        </div>
      </header>

      <div className="dashboard-container">
        <section className="welcome-section">
          <h2>Your Running Hub</h2>
          <p>Track your progress and get insights about your running journey.</p>
        </section>

        <section className="strava-section">
          {isConnecting ? (
            <div>
              <h2>Connecting to Strava...</h2>
              <p>Please wait while we complete your connection.</p>
            </div>
          ) : userData?.stravaConnected ? (
            <>
              <h2>Connected to Strava</h2>
              <p>Your Strava account is successfully connected!</p>
            </>
          ) : (
            <>
              <h2>Connect with Strava</h2>
              <p>Link your Strava account to access your running data and analytics.</p>
              <button 
                className="strava-button" 
                onClick={handleStravaConnect}
                disabled={isConnecting}
              >
                Connect with Strava
              </button>
            </>
          )}
          {error && <div className="error-message" role="alert">{error}</div>}
        </section>

        <div className="dashboard-grid">
          <div className="dashboard-card">
            <h3>Recent Activities</h3>
            <p>Connect your Strava account to see your activities</p>
          </div>

          <div className="dashboard-card">
            <h3>Running Goals</h3>
            <p>Set and track your running goals here</p>
          </div>

          <div className="dashboard-card">
            <h3>Weekly Summary</h3>
            <p>Connect your Strava account to see your weekly summary</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
