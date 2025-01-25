import axios from 'axios';

const BASE_URL = "http://127.0.0.1:8000"; // Replace with your backend's URL

// Create axios instance
export const api = axios.create({
  baseURL: BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor to include auth token
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Register user
export const registerUser = async (username, password, email) => {
  try {
    const response = await api.post('/register', {
      username,
      password,
      email,
    });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.message || 'Registration failed');
  }
};

// Login user
export const loginUser = async (username, password) => {
  try {
    const response = await api.post('/login', {
      username,
      password,
    });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.message || 'Login failed');
  }
};

// Fetch user info
export const fetchUserInfo = async () => {
  try {
    const response = await api.get('/user');
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.message || 'Failed to fetch user info');
  }
};
