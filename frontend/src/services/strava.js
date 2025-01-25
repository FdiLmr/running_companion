import { api } from './api';

export const stravaService = {
    // Get the Strava OAuth URL
    getStravaConnectUrl: async () => {
        try {
            const response = await api.get('/strava/connect');
            return response.data;
        } catch (error) {
            console.error('Error getting Strava connect URL:', error);
            throw error;
        }
    },

    // Handle the Strava callback
    handleStravaCallback: async (code) => {
        try {
            const response = await api.get('/strava/callback', {
                params: { code }
            });
            return response.data;
        } catch (error) {
            console.error('Error handling Strava callback:', error);
            if (error.response?.data?.detail) {
                throw new Error(error.response.data.detail);
            }
            throw error;
        }
    },

    // Get user's Strava data
    getStravaData: async () => {
        try {
            const response = await api.get('/strava/data');
            return response.data;
        } catch (error) {
            console.error('Error fetching Strava data:', error);
            throw error;
        }
    }
};
