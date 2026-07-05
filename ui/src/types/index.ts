// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Type definitions for the Shopping Assistant UI
 */

export interface MessageData {
  role: MessageRole;
  content: string | ImageContent | ImageRowContent;
  productName: string;
}

export type MessageRole = 
  | 'user' 
  | 'assistant' 
  | 'system' 
  | 'image' 
  | 'image_row' 
  | 'user_image'
  | 'user_video';

export interface ImageContent {
  productUrl: string;
  productName: string;
}

export interface ImageRowContent extends Array<ImageContent> {}

export interface ChatboxProps {
  setNewRenderImage: (value: string) => void;
}

export interface ApparelProps {
  newRenderImage: string;
}

export interface SafeHTMLProps {
  html: string;
}

export interface ChatMessageProps {
  role: MessageRole;
  content: string | ImageContent | ImageRowContent;
  productName: string;
}

export interface ApiRequest {
  user_id: number;
  query: string;
  guardrails: boolean;
  image: string;
  media?: MediaAttachment[];
  image_bool: boolean;
  session_id?: string;
  conversation_id?: string;
  cart_id?: string;
  context?: string;
  cart?: CartData;
  retrieved?: Record<string, string>;
}

export interface MediaAttachment {
  type: 'image' | 'video';
  data: string;
  mime_type: string;
  filename?: string;
}

export interface MediaCapabilities {
  enabled: boolean;
  allow_mixed_media: boolean;
  max_images_per_turn: number;
  max_videos_per_turn: number;
  image_mime_types: string[];
  video_mime_types: string[];
  max_image_bytes: number;
  max_video_bytes: number;
  max_video_duration_seconds: number;
  vlm_enabled: boolean;
}

export type CatalogCapabilityType = 'enum' | 'number' | 'text';

export interface CatalogFilterCapability {
  type: CatalogCapabilityType;
  operators: string[];
  source_fields: string[];
  values: string[];
  min_value?: number | null;
  max_value?: number | null;
  request_aliases: Record<string, string>;
}

export interface CatalogCapabilities {
  catalog_id: string;
  retrieval_modes: string[];
  image_search_enabled: boolean;
  filters: Record<string, CatalogFilterCapability>;
}

export interface CapabilitiesResponse {
  media_input: MediaCapabilities;
  catalog?: CatalogCapabilities;
}

export interface ApiResponse {
  response: string;
  images: Record<string, string>;
  timings: Record<string, number>;
}

export interface CartData {
  contents: CartItem[];
}

export interface CartItem {
  item: string;
  amount: number;
}

export interface StreamingChunk {
  type: 'content' | 'images' | 'error';
  payload: string | Record<string, string>;
  timestamp: number;
}

export interface UserSession {
  userId: number;
  sessionId: string;
  conversationId: string;
  cartId: string;
  isActive: boolean;
  createdAt: Date;
}

export interface FileUploadResult {
  file: File;
  base64: string;
  previewUrl: string;
}

export interface ErrorState {
  hasError: boolean;
  message: string;
  code?: string;
}
