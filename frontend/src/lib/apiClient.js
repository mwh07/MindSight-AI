// src/lib/apiClient.js
import axios from 'axios';

// src/lib/apiClient.js
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add response interceptor for debugging
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

export const mindsightAPI = {
  // Health check
  health: () => apiClient.get('/health'),
  
  // Submit assessment
  submitAssessment: (data) => apiClient.post('/assess', data),
  
  // Trigger training
  trainModels: () => apiClient.post('/train'),
  
  // Flush data
  flushData: (target = 'all') => apiClient.post('/flush', { target }),
};

export default apiClient;