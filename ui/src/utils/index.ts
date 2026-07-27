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
const GUEST_SHOPPER_STORAGE_VALUE = '__guest__';
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
 * Read the shopper mode explicitly selected in this browser tab.
 *
 * ``undefined`` means the shopper has not chosen yet, ``null`` means Guest,
 * and a string is a server-owned representative-shopper ID. This key is
 * intentionally separate from the chat identity so Reset can rotate a
 * conversation without silently changing the selected shopper mode.
 */
export const getSelectedShopperProfileId = (): string | null | undefined => {
  const storedProfileId = sessionStorage.getItem(SHOPPER_PROFILE_STORAGE_KEY);
  if (!storedProfileId) return undefined;
  if (storedProfileId === GUEST_SHOPPER_STORAGE_VALUE) return null;
  if (!SHOPPER_PROFILE_ID_PATTERN.test(storedProfileId)) {
    sessionStorage.removeItem(SHOPPER_PROFILE_STORAGE_KEY);
    return undefined;
  }
  return storedProfileId;
};

/**
 * Persist an explicit Guest or representative-shopper choice. Profile
 * contents always come from the server-owned read endpoint.
 */
export const setSelectedShopperProfileId = (
  shopperProfileId: string | null
): void => {
  if (shopperProfileId === null) {
    sessionStorage.setItem(
      SHOPPER_PROFILE_STORAGE_KEY,
      GUEST_SHOPPER_STORAGE_VALUE
    );
    return;
  }
  if (!SHOPPER_PROFILE_ID_PATTERN.test(shopperProfileId)) {
    throw new Error('Invalid shopper profile ID.');
  }
  sessionStorage.setItem(SHOPPER_PROFILE_STORAGE_KEY, shopperProfileId);
};

/**
 * Return the browser tab to the required unchosen shopper state.
 */
export const clearSelectedShopperProfileId = (): void => {
  sessionStorage.removeItem(SHOPPER_PROFILE_STORAGE_KEY);
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
