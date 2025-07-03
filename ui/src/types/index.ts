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
  | 'user_image';

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
  image_bool: boolean;
  context?: string;
  cart?: CartData;
  retrieved?: Record<string, string>;
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