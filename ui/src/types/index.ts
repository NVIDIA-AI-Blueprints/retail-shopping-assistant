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

export interface ProductPrice {
  amount: number;
  currency?: string;
}

export interface ProductSummary {
  productId?: string;
  productName: string;
  productUrl?: string;
  description?: string;
  category?: string;
  brand?: string;
  price?: ProductPrice | null;
  availability?: string;
  attributes?: Record<string, unknown>;
}

export interface ImageContent extends ProductSummary {
  productUrl: string;
  productName: string;
}

export interface ImageRowContent extends Array<ImageContent> {}

export interface ChatboxProps {
  selectedProduct: ProductSummary | null;
  onProductSelect: (product: ProductSummary | null) => void;
  onProductsUpdate: (products: ProductSummary[]) => void;
}

export interface ProductDetailPanelProps {
  selectedProduct: ProductSummary | null;
  products: ProductSummary[];
  onProductSelect: (product: ProductSummary | null) => void;
}

export interface SafeHTMLProps {
  html: string;
}

export interface ChatMessageProps {
  role: MessageRole;
  content: string | ImageContent | ImageRowContent;
  productName: string;
  selectedProductName?: string;
  onProductSelect?: (product: ProductSummary) => void;
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

export interface ModelCapability {
  label: string;
  model: string | null;
  source: string;
  enabled: boolean;
}

export interface ModelCapabilities {
  app_llm?: ModelCapability;
  vlm?: ModelCapability;
  text_embedding?: ModelCapability;
  image_embedding?: ModelCapability;
  content_safety?: ModelCapability;
  topic_control?: ModelCapability;
  [role: string]: ModelCapability | undefined;
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
  models?: ModelCapabilities;
  catalog?: CatalogCapabilities;
}

export interface ApiResponse {
  response: string;
  images: Record<string, string>;
  timings: Record<string, number>;
  token_usage?: TokenUsage;
}

export interface CartData {
  contents: CartItem[];
}

export interface CartItem {
  item: string;
  amount: number;
}

export interface StreamingChunk {
  type: 'content' | 'images' | 'products' | 'metrics' | 'error';
  payload: string | Record<string, string> | ProductSummary[] | InferenceMetricsPayload;
  timestamp: number;
}

export interface InferenceMetricsPayload {
  timings: Record<string, number>;
  total_seconds?: number;
  token_usage?: TokenUsage;
  model_usage?: ModelUsage;
}

export interface TokenUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  model_calls: number;
}

export type ModelUsageStatus = 'used' | 'failed' | 'disabled' | 'not_used';

export interface ModelUsageEntry {
  status: ModelUsageStatus;
  calls: number;
  detail?: string;
}

export type ModelUsage = Record<string, ModelUsageEntry | undefined>;

export type InferenceCategory = 'vision' | 'language' | 'embedding' | 'safety' | 'memory' | 'system';
export type InferenceStatus = 'queued' | 'running' | 'complete' | 'failed';

export interface InferenceActivity {
  id: string;
  category: InferenceCategory;
  label: string;
  detail: string;
  modelName?: string;
  status: InferenceStatus;
  durationMs?: number;
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
