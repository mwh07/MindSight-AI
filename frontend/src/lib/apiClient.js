// src/lib/apiClient.js
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000';

// Create axios instance
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Response interceptor for debugging
apiClient.interceptors.response.use(
  (response) => {
    console.log('✅ API Response:', response.data);
    return response;
  },
  (error) => {
    console.error('❌ API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// Named export for all API methods
export const mindsightAPI = {
  health: () => apiClient.get('/health'),
  submitAssessment: (data) => apiClient.post('/assess', data),
  trainModels: () => apiClient.post('/train'),
  flushData: (target = 'all') => apiClient.post('/flush', { target }),
  latestReport: () => apiClient.get('/latest-report'),   // ✅ Added here
};

// Default export for the raw client (for custom requests)
export default apiClient; 