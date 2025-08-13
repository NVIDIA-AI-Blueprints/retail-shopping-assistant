// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
    reader.onerror = (error) => reject(new Error("Failed to convert file to based64."));
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
  
  // Use timestamp + random component to avoid collisions
  const newId = Date.now() * 1000 + Math.floor(Math.random() * 1000);
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
    return 'Please select a valid image file (JPEG or PNG only)';
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

export interface CartOperation {
  type: 'add' | 'remove';
  item: string;
}

/**
 * Clean item name by removing markdown formatting and extra whitespace
 */
const cleanItemName = (item: string): string => {
  return item
    .replace(/\*\*/g, '') // Remove markdown bold markers
    .trim(); // Remove extra whitespace
};

/**
 * Heuristic: ignore second-person narrations (e.g., "you've added", "you added")
 * to avoid false-positive toasts when the model is just describing prior state.
 * Keep it intentionally narrow to minimize side effects.
 */
const isSecondPersonCartNarration = (message: string): boolean => {
  const lower = message.toLowerCase();
  return (
    /\byou(?:'ve| have)?\s+added\b/.test(lower) ||
    /\byou\s+added\b/.test(lower) ||
    /\byou(?:'ve| have)?\s+removed\b/.test(lower) ||
    /\byou\s+removed\b/.test(lower)
  );
};

/**
 * Detect cart operations from response messages - simplified to focus only on product names
 */
export const detectCartOperation = (message: string): CartOperation | null => {
  // Early exit on second-person narration to reduce false positives
  if (isSecondPersonCartNarration(message)) {
    return null;
  }

  // Pattern for add operations - captures item name from either format
  const addPattern = /(?:added.*?(?:of\s+)?['"]?([^'"]+?)['"]?\s+to.*cart|added.*?\*\*([^*]+)\*\*.*to.*cart)/i;
  
  // Pattern for remove operations - captures item name from either format  
  const removePattern = /(?:removed.*?(?:of\s+)?['"]?([^'"]+?)['"]?\s+from.*cart|removed.*?\*\*([^*]+)\*\*.*from.*cart)/i;
  
  // Check for add operations
  let match = message.match(addPattern);
  if (match) {
    const item = match[1] || match[2]; // Get whichever group matched
    if (item) {
      return {
        type: 'add',
        item: cleanItemName(item)
      };
    }
  }
  
  // Check for remove operations
  match = message.match(removePattern);
  if (match) {
    const item = match[1] || match[2]; // Get whichever group matched
    if (item) {
      return {
        type: 'remove',
        item: cleanItemName(item)
      };
    }
  }
  
  return null;
};

/**
 * Show cart operation notification using the existing toast system
 */
export const showCartNotification = (
  fullResponse: string, 
  shownOperations: Set<string>,
  toast: any
): void => {
  const cartOperation = detectCartOperation(fullResponse);
  
  if (cartOperation) {
    const operationKey = `${cartOperation.type}-${cartOperation.item}`;
    
    if (!shownOperations.has(operationKey)) {
      shownOperations.add(operationKey);
      
      const message = cartOperation.type === 'add'
        ? `Added ${cartOperation.item} to cart`
        : `🗑️ Removed ${cartOperation.item} from cart`;
      
      // Use the same simple approach as file upload notifications
      if (cartOperation.type === 'add') {
        toast.success(message);
      } else {
        toast.info(message);
      }
    }
  }
}; 