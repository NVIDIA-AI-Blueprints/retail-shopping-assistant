// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Utility functions for the Shopping Assistant UI
 */

import {
  ApiRequest,
  FileUploadResult,
  MediaAttachment,
  ShopperProfile,
  StreamingChunk,
  UserSession,
} from '../types';
import { config } from '../config/config';

const SESSION_STORAGE_KEY = 'shopping_session_identity';
const LEGACY_USER_ID_KEY = 'shopping_user_id';
const SHOPPER_PROFILE_STORAGE_KEY = 'shopping_shopper_profile_id';
const SHOPPER_PROFILE_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;
const SHOPPER_TYPE_PATTERN = /^[a-z][a-z0-9_]*$/;
const ZIPCODE_PATTERN = /^[0-9]{5}$/;

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
 * Get or create a user ID from session storage.
 */
export const getOrCreateUserId = (): number => {
  return getOrCreateUserSession().userId;
};

/**
 * Get or create the browser-scoped assistant identity sent to the API.
 */
export const getOrCreateUserSession = (): UserSession => {
  const storedSession = readStoredUserSession();
  if (storedSession) return storedSession;

  const userId = getStoredUserId() || createUserId();
  const session = {
    userId,
    sessionId: createScopedId('session'),
    conversationId: createScopedId('conversation'),
    cartId: createScopedId('cart'),
    isActive: true,
    createdAt: new Date(),
  };
  storeUserSession(session);
  return session;
};

/**
 * Clear user session data
 */
export const clearUserSession = (): void => {
  sessionStorage.removeItem(SESSION_STORAGE_KEY);
  sessionStorage.removeItem(LEGACY_USER_ID_KEY);
};

/**
 * Replace the browser-scoped chat/cart identity immediately.
 */
export const rotateUserSession = (): UserSession => {
  clearUserSession();
  return getOrCreateUserSession();
};

/**
 * Read the representative shopper selected in this browser tab.
 *
 * This key is intentionally separate from the chat identity so resetting or
 * rotating a conversation does not silently change the selected shopper.
 */
export const getSelectedShopperProfileId = (): string | null => {
  const storedProfileId = sessionStorage.getItem(SHOPPER_PROFILE_STORAGE_KEY);
  if (!storedProfileId) return null;
  if (!SHOPPER_PROFILE_ID_PATTERN.test(storedProfileId)) {
    sessionStorage.removeItem(SHOPPER_PROFILE_STORAGE_KEY);
    return null;
  }
  return storedProfileId;
};

/**
 * Persist only the selected profile ID. Profile contents always come from the
 * server-owned read endpoint.
 */
export const setSelectedShopperProfileId = (
  shopperProfileId: string | null
): void => {
  if (shopperProfileId === null) {
    sessionStorage.removeItem(SHOPPER_PROFILE_STORAGE_KEY);
    return;
  }
  if (!SHOPPER_PROFILE_ID_PATTERN.test(shopperProfileId)) {
    throw new Error('Invalid shopper profile ID.');
  }
  sessionStorage.setItem(SHOPPER_PROFILE_STORAGE_KEY, shopperProfileId);
};

/**
 * Validate the closed shopper-profile list returned by the chain server.
 */
export const parseShopperProfiles = (value: unknown): ShopperProfile[] => {
  if (!Array.isArray(value) || value.length !== 5) {
    throw new Error('Shopper profiles returned an invalid response.');
  }
  if (!value.every(isShopperProfile)) {
    throw new Error('Shopper profiles returned an invalid response.');
  }

  const profileIds = value.map((profile) => profile.shopper_profile_id);
  const shopperTypes = value.map((profile) => profile.shopper_type);
  if (
    new Set(profileIds).size !== profileIds.length ||
    new Set(shopperTypes).size !== shopperTypes.length
  ) {
    throw new Error('Shopper profiles returned duplicate identifiers.');
  }
  return value;
};

const isShopperProfile = (value: unknown): value is ShopperProfile => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const profile = value as Record<string, unknown>;
  return (
    isTrimmedString(profile.shopper_profile_id, 64) &&
    SHOPPER_PROFILE_ID_PATTERN.test(profile.shopper_profile_id) &&
    isTrimmedString(profile.display_name, 80) &&
    isTrimmedString(profile.shopper_type, 80) &&
    SHOPPER_TYPE_PATTERN.test(profile.shopper_type) &&
    isTrimmedString(profile.behavior, 512) &&
    typeof profile.zipcode === 'string' &&
    ZIPCODE_PATTERN.test(profile.zipcode)
  );
};

const isTrimmedString = (value: unknown, maxLength: number): value is string => {
  return (
    typeof value === 'string' &&
    value.length > 0 &&
    value.length <= maxLength &&
    value === value.trim()
  );
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
  userSession: UserSession,
  query: string,
  image: string = '',
  guardrails: boolean = true,
  media: MediaAttachment[] = [],
  shopperProfileId: string | null = null
): ApiRequest => {
  const payload: ApiRequest = {
    user_id: userSession.userId,
    session_id: userSession.sessionId,
    conversation_id: userSession.conversationId,
    cart_id: userSession.cartId,
    query,
    guardrails,
    image,
    media,
    image_bool: !!image,
  };
  if (shopperProfileId !== null) {
    payload.shopper_profile_id = shopperProfileId;
  }
  return payload;
};

const getStoredUserId = (): number | null => {
  const storedId = sessionStorage.getItem(LEGACY_USER_ID_KEY);
  if (!storedId) return null;

  const parsed = parseInt(storedId, 10);
  return Number.isFinite(parsed) ? parsed : null;
};

const createUserId = (): number => {
  const userId = Date.now() * 1000 + Math.floor(Math.random() * 1000);
  sessionStorage.setItem(LEGACY_USER_ID_KEY, String(userId));
  return userId;
};

const createScopedId = (prefix: string): string => {
  const browserCrypto =
    typeof crypto !== 'undefined'
      ? (crypto as Crypto & { randomUUID?: () => string })
      : undefined;
  const randomId =
    browserCrypto && typeof browserCrypto.randomUUID === 'function'
      ? browserCrypto.randomUUID()
      : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
  return `${prefix}-${randomId}`;
};

const readStoredUserSession = (): UserSession | null => {
  const storedSession = sessionStorage.getItem(SESSION_STORAGE_KEY);
  if (!storedSession) return null;

  try {
    const parsed = JSON.parse(storedSession);
    if (
      typeof parsed.userId === 'number' &&
      typeof parsed.sessionId === 'string' &&
      typeof parsed.conversationId === 'string' &&
      typeof parsed.cartId === 'string'
    ) {
      return {
        userId: parsed.userId,
        sessionId: parsed.sessionId,
        conversationId: parsed.conversationId,
        cartId: parsed.cartId,
        isActive: true,
        createdAt: parsed.createdAt ? new Date(parsed.createdAt) : new Date(),
      };
    }
  } catch (error) {
    sessionStorage.removeItem(SESSION_STORAGE_KEY);
  }

  return null;
};

const storeUserSession = (session: UserSession): void => {
  sessionStorage.setItem(LEGACY_USER_ID_KEY, String(session.userId));
  sessionStorage.setItem(
    SESSION_STORAGE_KEY,
    JSON.stringify({
      userId: session.userId,
      sessionId: session.sessionId,
      conversationId: session.conversationId,
      cartId: session.cartId,
      createdAt: session.createdAt.toISOString(),
    })
  );
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
  const timestamp = date.toISOString().replace(/[:-]|\.\d{3}/g, '');
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

