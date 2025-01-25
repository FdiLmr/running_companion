const API_URL = 'http://localhost:8000'; // Update this with your FastAPI server URL

export const register = async (username, email, password) => {
  const response = await fetch(`${API_URL}/register`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      username,
      email,
      password,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Registration failed');
  }

  return response.json();
};

export const login = async (username, password) => {
  const response = await fetch(`${API_URL}/login`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      username,
      password,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Login failed');
  }

  const data = await response.json();
  // Store tokens in localStorage
  localStorage.setItem('access_token', data.access_token);
  localStorage.setItem('id_token', data.id_token);
  return data;
};

export const getUserProfile = async () => {
  const token = localStorage.getItem('id_token');
  if (!token) {
    throw new Error('No id token found');
  }

  const response = await fetch(`${API_URL}/user`, {
    headers: {
      'Authorization': `Bearer ${token}`,
    },
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to fetch user profile');
  }

  return response.json();
};

export const logout = () => {
  localStorage.removeItem('access_token');
  localStorage.removeItem('id_token');
};
