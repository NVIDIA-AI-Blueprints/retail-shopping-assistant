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
  selectedShopperProfileId: string | null;
  onProductSelect: (product: ProductSummary | null) => void;
  onProductsUpdate: (products: ProductSummary[]) => void;
  onBusyChange: (isBusy: boolean) => void;
  preserveIdentityOnMount: boolean;
}

export interface ShopperProfile {
  shopper_profile_id: string;
  display_name: string;
  shopper_type: string;
  behavior: string;
  zipcode: string;
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
  shopper_profile_id?: string;
  context?: string;
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

export type CatalogCapabilityType = 'enum' | 'enum_list' | 'number' | 'text';
export type CatalogFieldType = CatalogCapabilityType | 'unclassified';

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
  product_count: number;
  retrieval_modes: string[];
  image_search_enabled: boolean;
  filters: Record<string, CatalogFilterCapability>;
  fields: Record<string, CatalogFieldCapability>;
  taxonomy: CatalogTaxonomyCapabilities;
}

export interface CatalogFieldCapability {
  type: CatalogFieldType;
  observed_type?: string | null;
  filterable: boolean;
  searchable: boolean;
  detail: boolean;
  taxonomy: boolean;
  operators: string[];
  source_fields: string[];
  coverage: { present: number; total: number };
  values: Array<{ value: string; count: number }>;
  min_value?: number | null;
  max_value?: number | null;
}

export interface CatalogTaxonomyScope {
  product_count: number;
  filters: Record<string, CatalogFieldCapability>;
  semantic_fields: Record<string, CatalogFieldCapability>;
}

export interface CatalogTaxonomyCategory extends CatalogTaxonomyScope {
  subcategories: Record<string, CatalogTaxonomyScope>;
}

export interface CatalogTaxonomyCapabilities {
  category_field?: string | null;
  subcategory_field?: string | null;
  categories: Record<string, CatalogTaxonomyCategory>;
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
  agent_diagnostics?: AgentDiagnostics;
}

/**
 * A cart line as the server actually returns it.
 *
 * The previous `CartItem` carried only `item` and `amount`, was referenced by
 * nothing, and could not address a line for mutation -- it had no line id.
 */
export interface CartLine {
  cart_line_id: string;
  product_id: string;
  display_name: string;
  quantity: number;
  /** Absent for one-size goods rather than null, so test presence not truth. */
  size?: string | null;
  unit_price?: number | null;
}

/** What the vision model saw, as the server projects it for display. */
export interface MediaAnalysisItem {
  label: string;
  /** How many of the model's own searches chase this item. 0 = seen, not searched. */
  pursued: number;
}

export interface MediaAnalysis {
  summary: string;
  items: MediaAnalysisItem[];
  colors: string[];
  materials: string[];
  style: string[];
  occasion: string[];
  queries: string[];
}

export interface CartSnapshot {
  lines: CartLine[];
  subtotal: number | null;
}

export interface StreamingChunk {
  type: 'content' | 'images' | 'products' | 'metrics' | 'media_analysis' | 'error';
  payload: string | Record<string, string> | ProductSummary[] | InferenceMetricsPayload;
  timestamp: number;
}

export interface InferenceMetricsPayload {
  timings: Record<string, number>;
  total_seconds?: number;
  token_usage?: TokenUsage;
  model_usage?: ModelUsage;
  agent_diagnostics?: AgentDiagnostics;
}

export interface AgentToolCallDiagnostic {
  sequence: number;
  tool_name: string;
  arguments: Record<string, unknown>;
  status: 'completed' | 'rejected' | 'error' | 'pending';
  rejection_reason?: string;
  duplicate?: boolean;
  restored_fields?: string[];
}

export interface AgentPartialMessageDiagnostic {
  type: string;
  content: string;
  name?: string;
  tool_call_id?: string;
  tool_calls?: Array<Record<string, unknown>>;
  truncated?: boolean;
}

export interface AgentProductEvidenceSearchScope {
  taxonomy: Record<string, unknown>;
  confirmed_filters: Record<string, unknown>;
}

export interface AgentProductEvidenceDiagnostic {
  product_ref: string;
  product_name: string;
  source_tool: 'search_catalog_tool' | 'get_product_details_tool';
  evidence_type: 'search_result' | 'product_detail';
  facts: Record<string, unknown>;
  search_scope?: AgentProductEvidenceSearchScope;
}

export interface AgentCatalogScopeOutcomeDiagnostic {
  outcome: 'no_direct_catalog_match' | 'zero_results';
  requested_product_type?: string | null;
  taxonomy?: Record<string, unknown>;
  confirmed_filters?: Record<string, unknown>;
}

export interface AgentDiagnostics {
  skill_files_read: string[];
  tool_calls: AgentToolCallDiagnostic[];
  rejected_tool_calls: number[];
  duplicate_tool_calls: number[];
  product_evidence: AgentProductEvidenceDiagnostic[];
  product_evidence_truncated: boolean;
  catalog_scope_outcomes: AgentCatalogScopeOutcomeDiagnostic[];
  final_termination_reason: string;
  partial_graph_messages: AgentPartialMessageDiagnostic[];
  partial_graph_messages_truncated?: boolean;
  partial_graph_capture_error?: string;
  diagnostic_collection_error?: string;
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
  /** Present only for roles that report token usage: the chat models. */
  tokens?: number;
}

export type ModelUsage = Record<string, ModelUsageEntry | undefined>;

/**
 * Totals across the whole session rather than the last turn.
 *
 * `TokenUsage` is replaced wholesale on every metrics event, so it answers
 * "what did that question cost". This answers "what has this conversation
 * cost", which is the number worth watching during a demo.
 */
export interface SessionUsage {
  modelCalls: number;
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
}

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
