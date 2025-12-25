import axios from 'axios';

// Configure your backend URL here
// For iOS simulator: use your computer's local IP address
// Example: http://192.168.1.100:8000
const API_BASE_URL = 'http://192.168.5.14:8000';

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface FoodPrediction {
  name: string;
  confidence: number;
  food_id: string;
}

export interface PortionEstimate {
  food_id: string;
  portion_size: string;
  grams: number;
  confidence: number;
}

export interface AnalysisResponse {
  analysis_id: string;
  timestamp: string;
  foods: FoodPrediction[];
  calories_min: number;
  calories_max: number;
  calories_estimate: number;
  confidence: number;
  portions: PortionEstimate[];
  nutrients: {
    protein: number;
    carbs: number;
    fat: number;
  };
}

export const analyzeImage = async (imageUri: string, userId: string = 'default'): Promise<AnalysisResponse> => {
  try {
    const formData = new FormData();
    
    // For React Native, we need to use a different format
    const filename = imageUri.split('/').pop() || 'meal.jpg';
    const match = /\.(\w+)$/.exec(filename);
    const type = match ? `image/${match[1]}` : 'image/jpeg';
    
    formData.append('file', {
      uri: imageUri,
      name: filename,
      type: type,
    } as any);
    
    formData.append('user_id', userId);

    const result = await api.post('/analyze', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      transformRequest: (data) => data, // Don't let axios transform the data
      timeout: 120000, // 2 minutes for first inference (model warmup)
    });

    return result.data;
  } catch (error) {
    console.error('Error analyzing image:', error);
    throw error;
  }
};

export const setBaseline = async (
  userId: string,
  analysisId: string,
  actualCalories?: number,
  notes?: string
) => {
  try {
    const result = await api.post('/baseline/set', {
      user_id: userId,
      analysis_id: analysisId,
      actual_calories: actualCalories,
      notes: notes,
    });
    return result.data;
  } catch (error) {
    console.error('Error setting baseline:', error);
    throw error;
  }
};

export const compareToBaseline = async (userId: string, analysisId: string) => {
  try {
    const result = await api.post('/baseline/compare', {
      user_id: userId,
      analysis_id: analysisId,
    });
    return result.data;
  } catch (error) {
    console.error('Error comparing to baseline:', error);
    throw error;
  }
};

export const getHistory = async (userId: string = 'default', limit: number = 50) => {
  try {
    const result = await api.get('/history', {
      params: { user_id: userId, limit },
    });
    return result.data;
  } catch (error) {
    console.error('Error fetching history:', error);
    throw error;
  }
};

export const exportHistory = async (userId: string = 'default', format: 'json' | 'csv' = 'json') => {
  try {
    const result = await api.get('/history/export', {
      params: { user_id: userId, format },
    });
    return result.data;
  } catch (error) {
    console.error('Error exporting history:', error);
    throw error;
  }
};

