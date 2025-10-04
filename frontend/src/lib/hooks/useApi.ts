import { useState, useCallback } from 'react';
import axios, { AxiosRequestConfig, AxiosResponse } from 'axios';

interface ApiResponse<T = any> {
  data: T;
  status: number;
  statusText: string;
}

interface UseApiReturn {
  loading: boolean;
  error: string | null;
  get: <T = any>(url: string, config?: AxiosRequestConfig) => Promise<ApiResponse<T>>;
  post: <T = any>(url: string, data?: any, config?: AxiosRequestConfig) => Promise<ApiResponse<T>>;
  put: <T = any>(url: string, data?: any, config?: AxiosRequestConfig) => Promise<ApiResponse<T>>;
  patch: <T = any>(url: string, data?: any, config?: AxiosRequestConfig) => Promise<ApiResponse<T>>;
  delete: <T = any>(url: string, config?: AxiosRequestConfig) => Promise<ApiResponse<T>>;
}

// Create axios instance with default configuration
const apiClient = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for adding auth tokens
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('accessToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for handling common errors
apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401) {
      // Try to refresh token
      const refreshToken = localStorage.getItem('refreshToken');
      if (refreshToken) {
        try {
          const response = await axios.post(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001'}/auth/refresh`, {
            refreshToken,
          });
          
          const { accessToken, refreshToken: newRefreshToken } = response.data;
          localStorage.setItem('accessToken', accessToken);
          localStorage.setItem('refreshToken', newRefreshToken);
          
          // Retry the original request
          error.config.headers.Authorization = `Bearer ${accessToken}`;
          return apiClient.request(error.config);
        } catch (refreshError) {
          // Refresh failed, redirect to login
          localStorage.removeItem('accessToken');
          localStorage.removeItem('refreshToken');
          window.location.href = '/auth/login';
          return Promise.reject(refreshError);
        }
      } else {
        // No refresh token, redirect to login
        window.location.href = '/auth/login';
      }
    }
    return Promise.reject(error);
  }
);

export const useApi = (): UseApiReturn => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleRequest = useCallback(async <T = any>(
    requestPromise: Promise<AxiosResponse<T>>
  ): Promise<ApiResponse<T>> => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await requestPromise;
      return {
        data: response.data,
        status: response.status,
        statusText: response.statusText,
      };
    } catch (err: any) {
      const errorMessage = err.response?.data?.message || err.message || 'An unexpected error occurred';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setLoading(false);
    }
  }, []);

  const get = useCallback(<T = any>(url: string, config?: AxiosRequestConfig) => {
    return handleRequest<T>(apiClient.get(url, config));
  }, [handleRequest]);

  const post = useCallback(<T = any>(url: string, data?: any, config?: AxiosRequestConfig) => {
    return handleRequest<T>(apiClient.post(url, data, config));
  }, [handleRequest]);

  const put = useCallback(<T = any>(url: string, data?: any, config?: AxiosRequestConfig) => {
    return handleRequest<T>(apiClient.put(url, data, config));
  }, [handleRequest]);

  const patch = useCallback(<T = any>(url: string, data?: any, config?: AxiosRequestConfig) => {
    return handleRequest<T>(apiClient.patch(url, data, config));
  }, [handleRequest]);

  const del = useCallback(<T = any>(url: string, config?: AxiosRequestConfig) => {
    return handleRequest<T>(apiClient.delete(url, config));
  }, [handleRequest]);

  return {
    loading,
    error,
    get,
    post,
    put,
    patch,
    delete: del,
  };
};

export default useApi;