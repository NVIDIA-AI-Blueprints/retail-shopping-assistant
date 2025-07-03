/**
 * Utility functions for the Shopping Assistant UI
 */

import { FileUploadResult, UserSession, StreamingChunk, ApiRequest } from '../types';
import { config } from '../config/config';

/**
 * Convert a file to base64 string
 */
export const convertToBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = (error) => reject(error);
    reader.readAsDataURL(file);
  });
};

/**
 * Convert base64 string back to a file blob
 */
export const base64ToBlob = (base64: string): Blob => {
  const base64WithoutPrefix = base64.split(',')[1];
  const binaryString = atob(base64WithoutPrefix);
  const byteArray = new Uint8Array(binaryString.length);
  
  for (let i = 0; i < binaryString.length; i++) {
    byteArray[i] = binaryString.charCodeAt(i);
  }
  
  return new Blob([byteArray], { type: "image/png" });
};

/**
 * Get or create a user ID from session storage
 */
export const getOrCreateUserId = (): number => {
  const storedId = sessionStorage.getItem('shopping_user_id');
  if (storedId) return parseInt(storedId, 10);
  
  const newId = Math.floor(Math.random() * 100000);
  sessionStorage.setItem('shopping_user_id', String(newId));
  return newId;
};

/**
 * Clear user session data
 */
export const clearUserSession = (): void => {
  sessionStorage.removeItem('shopping_user_id');
};

/**
 * Handle file upload and validation
 */
export const handleFileUpload = async (file: File): Promise<FileUploadResult> => {
  // Validate file size
  const maxSizeMB = config.features.imageUpload.maxSize;
  if (file.size > maxSizeMB * 1024 * 1024) {
    throw new Error(`File size must be less than ${maxSizeMB}MB`);
  }

  // Validate file type
  if (!config.features.imageUpload.allowedTypes.includes(file.type)) {
    throw new Error('Invalid file type. Please upload an image file.');
  }

  // Convert to base64
  const base64 = await convertToBase64(file);
  
  // Create preview URL
  const previewUrl = URL.createObjectURL(file);

  return {
    file,
    base64,
    previewUrl,
  };
};

/**
 * Parse streaming response chunks
 */
export const parseStreamingChunk = (rawData: string): StreamingChunk | null => {
  if (rawData === '[DONE]') {
    return null;
  }

  try {
    const { type, payload, timestamp } = JSON.parse(rawData);
    return { type, payload, timestamp };
  } catch (error) {
    console.error('Failed to parse streaming chunk:', error);
    return null;
  }
};

/**
 * Create API request payload
 */
export const createApiRequest = (
  userId: number,
  query: string,
  image: string = '',
  guardrails: boolean = true
): ApiRequest => {
  return {
    user_id: userId,
    query,
    guardrails,
    image,
    image_bool: !!image,
  };
};

/**
 * Sleep utility function
 */
export const sleep = (ms: number): Promise<void> => {
  return new Promise((resolve) => setTimeout(resolve, ms));
};

/**
 * Download messages as JSON file
 */
export const downloadMessages = (messages: any[], filename?: string): void => {
  const jsonStr = JSON.stringify(messages, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json' });
  
  const date = new Date();
  const timestamp = date.toISOString().replace(/[:\-]|\.\d{3}/g, '');
  const defaultFilename = `messages_${timestamp}.json`;
  
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename || defaultFilename;
  link.style.display = 'none';
  
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

/**
 * Validate image file
 */
export const validateImageFile = (file: File): string | null => {
  const maxSizeMB = config.features.imageUpload.maxSize;
  const maxSizeBytes = maxSizeMB * 1024 * 1024;
  
  if (file.size > maxSizeBytes) {
    return `File size must be less than ${maxSizeMB}MB`;
  }
  
  if (!config.features.imageUpload.allowedTypes.includes(file.type)) {
    return 'Please select a valid image file (JPEG, PNG, GIF, or WebP)';
  }
  
  return null;
};

/**
 * Format file size for display
 */
export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}; 