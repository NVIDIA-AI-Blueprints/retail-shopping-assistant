/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import React, { useState, useEffect, useRef } from "react";
import { toast } from "react-toastify";
import SendIcon from "@mui/icons-material/Send";
import AttachFileIcon from "@mui/icons-material/AttachFile";
import CloseIcon from "@mui/icons-material/Close";
import Switch from '@mui/material/Switch';
import { styled } from '@mui/material/styles';

import ChatMessage from "./ChatMessage";
import MediaAnalysisCard from "./MediaAnalysisCard";
import ModelStrip from "./ModelStrip";
import {
  CapabilitiesResponse,
  ChatboxProps,
  ImageContent,
  InferenceMetricsPayload,
  MediaAttachment,
  MediaAnalysis,
  MediaCapabilities,
  MessageData,
  MessageRole,
  ModelCapabilities,
  ModelUsage,
  ProductPrice,
  ProductSummary,
  SessionUsage,
} from "../../types";
import { config } from "../../config/config";
import {
  clearUserSession,
  createApiRequest,
  getOrCreateUserSession,
} from "../../utils";
import logo from "../../assets/nvidia-logo.png";

/**
 * Main chatbox component for the shopping assistant
 */

// Custom styled switch component
const CustomSwitch = styled(Switch)(({ theme }) => ({
  '& .MuiSwitch-switchBase.Mui-checked': {
    color: '#76b900',
  },
  '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
    backgroundColor: '#a3bf73',
  },
  '& .MuiSwitch-track': {
    backgroundColor: 'lightgray',
  },
}));

const normalizeProduct = (raw: unknown): ProductSummary | null => {
  if (!raw || typeof raw !== "object") return null;
  const value = raw as Record<string, unknown>;
  const productName = stringValue(value.display_name) || stringValue(value.productName);
  if (!productName) return null;

  return {
    productId: stringValue(value.product_id) || stringValue(value.productId),
    productName,
    productUrl: stringValue(value.image_url) || stringValue(value.productUrl),
    description: stringValue(value.description),
    category: stringValue(value.category),
    brand: stringValue(value.brand),
    price: normalizePrice(value.price),
    availability: stringValue(value.availability),
    attributes: normalizeAttributes(value.attributes),
  };
};

const normalizePrice = (raw: unknown): ProductPrice | null => {
  if (!raw || typeof raw !== "object") return null;
  const value = raw as Record<string, unknown>;
  const amount = Number(value.amount);
  if (!Number.isFinite(amount)) return null;
  return {
    amount,
    currency: stringValue(value.currency) || "USD",
  };
};

const normalizeAttributes = (raw: unknown): Record<string, unknown> | undefined => {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return undefined;
  return raw as Record<string, unknown>;
};

const stringValue = (value: unknown): string => {
  return typeof value === "string" ? value.trim() : "";
};

const productKey = (name: string): string => name.trim().toLowerCase();

/** Add a turn's model usage to the session's running totals. */
const mergeModelUsage = (session: ModelUsage, turn: ModelUsage): ModelUsage => {
  const merged: ModelUsage = { ...session };
  Object.entries(turn).forEach(([role, entry]) => {
    if (!entry) return;
    const prior = merged[role];
    const tokens = (prior?.tokens ?? 0) + (entry.tokens ?? 0);
    merged[role] = {
      // The latest turn's status: a role that failed just now is worth seeing
      // even if earlier turns were fine.
      status: entry.status,
      calls: (prior?.calls ?? 0) + (entry.calls ?? 0),
      detail: entry.detail || prior?.detail,
      // Left undefined rather than zero, so roles that report no tokens show
      // nothing instead of claiming they used none.
      ...(tokens > 0 ? { tokens } : {}),
    };
  });
  return merged;
};

const mimeForFile = (file: File, allowedMimeTypes: string[]): string => {
  if (file.type && allowedMimeTypes.includes(file.type)) return file.type;
  const lowerName = file.name.toLowerCase();
  if (lowerName.endsWith(".mp4") && allowedMimeTypes.includes("video/mp4")) {
    return "video/mp4";
  }
  if ((lowerName.endsWith(".jpg") || lowerName.endsWith(".jpeg")) && allowedMimeTypes.includes("image/jpeg")) {
    return "image/jpeg";
  }
  if (lowerName.endsWith(".png") && allowedMimeTypes.includes("image/png")) {
    return "image/png";
  }
  return file.type || "";
};

const dataUrlWithMime = (dataUrl: string, mimeType: string): string => {
  if (!mimeType || !dataUrl.startsWith("data:")) return dataUrl;
  if (/^data:[^;]+;base64,/i.test(dataUrl)) return dataUrl;
  return dataUrl.replace(/^data:;base64,/i, `data:${mimeType};base64,`);
};

const acceptedMediaTypes = (capabilities: MediaCapabilities): string => {
  const imageTypes = capabilities.image_mime_types.flatMap((mimeType) => {
    if (mimeType === "image/jpeg") return [mimeType, ".jpg", ".jpeg"];
    if (mimeType === "image/png") return [mimeType, ".png"];
    return [mimeType];
  });
  const advertisedVideoTypes = capabilities.video_mime_types.length > 0
    ? capabilities.video_mime_types
    : ["video/mp4"];
  const videoTypes = advertisedVideoTypes.flatMap((mimeType) => (
        mimeType === "video/mp4" ? [mimeType, ".mp4"] : [mimeType]
      ));
  return Array.from(new Set([...imageTypes, ...videoTypes])).join(",");
};

const supportedVideoLabel = (capabilities: MediaCapabilities): string => {
  if (!capabilities.vlm_enabled || capabilities.max_videos_per_turn <= 0) {
    return "Video upload is disabled by the current model configuration.";
  }
  const formats = capabilities.video_mime_types.map((mimeType) => {
    if (mimeType === "video/mp4") return "MP4";
    return mimeType;
  });
  return `Supported video format: ${formats.join(", ")}. Max ${(capabilities.max_video_bytes / (1024 * 1024)).toFixed(0)}MB.`;
};

/**
 * The four openers a shopper is offered.
 *
 * They were previously bullet points inside the welcome text, so nothing could
 * be clicked -- and two of them could not even be typed usefully: one carried a
 * literal "[product name]" placeholder, the other said "add the first item"
 * before any search had happened.
 *
 * Each of these exercises a different skill and every one is answerable from
 * the catalogue as it stands.
 */
const STARTER_PROMPTS: Array<{
  label: string;
  text: string;
  attachesMedia?: boolean;
}> = [
  {
    label: "Shop this look (give me a video or an image)",
    text: "Shop this look",
    attachesMedia: true,
  },
  {
    label: "A work conference outfit under $400",
    text: "A work conference outfit under $400",
  },
  {
    label: "Cancun wedding next week, I need a dress in size 2 and shoes",
    text:
      "Cancun wedding next week, I need a dress in size 2 and shoes",
  },
];


const Chatbox: React.FC<ChatboxProps> = ({
  selectedProduct,
  selectedShopperProfileId,
  onProductSelect,
  onProductsUpdate,
  onBusyChange,
  preserveIdentityOnMount,
}) => {
  const defaultMediaCapabilities: MediaCapabilities = {
    enabled: config.features.imageUpload.enabled,
    allow_mixed_media: true,
    max_images_per_turn: 1,
    max_videos_per_turn: 1,
    image_mime_types: config.features.imageUpload.allowedTypes,
    video_mime_types: ["video/mp4"],
    max_image_bytes: config.features.imageUpload.maxSize * 1024 * 1024,
    max_video_bytes: 50 * 1024 * 1024,
    max_video_duration_seconds: 120,
    vlm_enabled: true,
  };
  // Per-turn usage is replaced on every metrics event; this accumulates so the
  // chrome can answer "what has this conversation cost" rather than "what did
  // that one question cost".
  // Per-model usage arrives per turn and is replaced wholesale. Accumulating
  // it here keeps every figure in the bar on the same footing: without this
  // the per-model counts are this turn's while the total beside them is the
  // session's, which is two different meanings sitting next to each other.
  const sessionModelUsageRef = useRef<ModelUsage>({});
  const sessionUsageRef = useRef<SessionUsage>({
    modelCalls: 0,
    inputTokens: 0,
    outputTokens: 0,
    totalTokens: 0,
  });
  const mediaInputRef = useRef<HTMLInputElement | null>(null);
  const [conversationId, setConversationId] = useState<string>("");
  const [mediaAnalysis, setMediaAnalysis] = useState<MediaAnalysis | null>(null);
  const [modelCapabilities, setModelCapabilities] = useState<ModelCapabilities>({});
  const [modelUsage, setModelUsage] = useState<ModelUsage>({});
  const [sessionUsage, setSessionUsage] = useState<SessionUsage>({
    modelCalls: 0,
    inputTokens: 0,
    outputTokens: 0,
    totalTokens: 0,
  });
  const [isOpen] = useState<boolean>(true);
  const [hasBeenOpened, setHasBeenOpened] = useState<boolean>(false);
  const [newMessage, setNewMessage] = useState<string>("");
  const [isGuardrailsOn, setIsGuardrailsOn] = useState(config.features.guardrails.defaultState);
  const [image, setImage] = useState("");
  const [previewImage, setPreviewImage] = useState("");
  const [video, setVideo] = useState("");
  const [previewVideo, setPreviewVideo] = useState("");
  const [videoMimeType, setVideoMimeType] = useState("");
  const [videoFilename, setVideoFilename] = useState("");
  const [mediaCapabilities, setMediaCapabilities] = useState<MediaCapabilities>(defaultMediaCapabilities);
  const [messages, setMessages] = useState<MessageData[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const messageRefs = useRef<React.RefObject<HTMLDivElement>[]>([]);
  const [lastAssistantIndex, setLastAssistantIndex] = useState<number | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const productsByNameRef = useRef<Map<string, ProductSummary>>(new Map());
  const currentTurnHasMedia = useRef(false);
  const currentTurnGuardrails = useRef(isGuardrailsOn);
  const handleResetRef = useRef<((clearIdentity: boolean) => Promise<void>) | null>(null);
  const initialResetStartedRef = useRef(false);

  useEffect(() => {
    onBusyChange(isLoading);
  }, [isLoading, onBusyChange]);

  useEffect(
    () => () => {
      onBusyChange(false);
    },
    [onBusyChange]
  );

  // Utility functions
  const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

  const convertToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = (error) => reject(new Error("Failed to read the file."));
      reader.readAsDataURL(file);
    });
  };

  const base64ToBlob = (base64: string): Blob => {
    const base64WithoutPrefix = base64.split(',')[1];
    const binaryString = atob(base64WithoutPrefix);
    const byteArray = new Uint8Array(binaryString.length);
    
    for (let i = 0; i < binaryString.length; i++) {
      byteArray[i] = binaryString.charCodeAt(i);
    }
    
    return new Blob([byteArray], { type: "image/png" });
  };

  const getVideoDuration = (file: File): Promise<number> => {
    return new Promise((resolve, reject) => {
      const url = window.URL.createObjectURL(file);
      const element = document.createElement("video");
      element.preload = "metadata";
      element.onloadedmetadata = () => {
        window.URL.revokeObjectURL(url);
        resolve(element.duration || 0);
      };
      element.onerror = () => {
        window.URL.revokeObjectURL(url);
        reject(new Error("Failed to read video metadata."));
      };
      element.src = url;
    });
  };

  // Event handlers
  const toggleGuardrails = () => {
    setIsGuardrailsOn(!isGuardrailsOn);
  };

  const handleNewMessageChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setNewMessage(event.target.value);
  };

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const file = files[0];

    if (!mediaCapabilities.enabled) {
      toast.error("Media uploads are disabled.");
      return;
    }

    const imageMimeType = mimeForFile(file, mediaCapabilities.image_mime_types);
    const videoMimeTypeForFile = mimeForFile(file, mediaCapabilities.video_mime_types);
    const isImage = Boolean(imageMimeType && mediaCapabilities.image_mime_types.includes(imageMimeType));
    const isVideo = Boolean(videoMimeTypeForFile && mediaCapabilities.video_mime_types.includes(videoMimeTypeForFile));
    const videoAllowed = mediaCapabilities.vlm_enabled && mediaCapabilities.max_videos_per_turn > 0;

    if (!isImage && !isVideo) {
      toast.error(`Please select a supported image or video file. ${supportedVideoLabel(mediaCapabilities)}`);
      return;
    }
    if (isImage && mediaCapabilities.max_images_per_turn <= 0) {
      toast.error("Image upload is not available with the current configuration.");
      return;
    }
    if (isVideo && !videoAllowed) {
      toast.error("Video upload is not available with the current configuration.");
      return;
    }
    if (!mediaCapabilities.allow_mixed_media && ((isImage && video) || (isVideo && image))) {
      toast.error("This configuration allows only one media type per turn.");
      return;
    }

    const maxBytes = isImage ? mediaCapabilities.max_image_bytes : mediaCapabilities.max_video_bytes;
    if (file.size > maxBytes) {
      toast.error(`File size must be less than ${(maxBytes / (1024 * 1024)).toFixed(0)}MB`);
      return;
    }

    if (isVideo) {
      try {
        const duration = await getVideoDuration(file);
        if (duration > mediaCapabilities.max_video_duration_seconds) {
          toast.error(`Video must be ${mediaCapabilities.max_video_duration_seconds} seconds or shorter.`);
          return;
        }
      } catch (error) {
        toast.error("Failed to inspect video duration.");
        return;
      }
    }

    try {
      const base64Media = await convertToBase64(file);
      if (isImage) {
        const normalizedImage = dataUrlWithMime(base64Media, imageMimeType);
        setImage(normalizedImage);
        const decodedImage = base64ToBlob(normalizedImage);
        const imageUrl = window.URL.createObjectURL(decodedImage);
        setPreviewImage(imageUrl);
      } else {
        const normalizedVideo = dataUrlWithMime(base64Media, videoMimeTypeForFile);
        setVideo(normalizedVideo);
        setPreviewVideo(window.URL.createObjectURL(file));
        setVideoMimeType(videoMimeTypeForFile);
        setVideoFilename(file.name);
      }
      
      e.target.value = "";
    } catch (error) {
      toast.error('Failed to upload media');
    }
  };

  const clearImage = () => {
    setPreviewImage("");
    setImage("");
  };

  const clearVideo = () => {
    setPreviewVideo("");
    setVideo("");
    setVideoMimeType("");
    setVideoFilename("");
  };

  const addMessage = (
    role: MessageRole,
    content: MessageData["content"],
    productName: string = ""
  ) => {
    setMessages((prevMessages) => {
      const newMessages = [...prevMessages, { role, content, productName }];
      messageRefs.current = newMessages.map((_, i) => 
        messageRefs.current[i] || React.createRef<HTMLDivElement>()
      );
      
      if (role === "assistant" && (lastAssistantIndex === null || lastAssistantIndex < prevMessages.length)) {
        setLastAssistantIndex(prevMessages.length);
      }
      
      return newMessages;
    });
  };

  const updateLastMessage = (newContent: any, role?: MessageRole, appendContent?: boolean) => {
    setMessages((prevMessages) => {
      if (prevMessages.length === 0) return prevMessages;

      const updatedMessages = [...prevMessages];
      const lastMessageIndex = updatedMessages.length - 1;
      
      if (role) {
        updatedMessages[lastMessageIndex].role = role;
      }
      
      if (typeof newContent === "string") {
        updatedMessages[lastMessageIndex] = {
          ...updatedMessages[lastMessageIndex],
          content: (!appendContent) 
            ? updatedMessages[lastMessageIndex].content + newContent 
            : newContent,
        };
      } else {
        updatedMessages[lastMessageIndex] = {
          ...updatedMessages[lastMessageIndex],
          content: newContent,
        };
      }

      return updatedMessages;
    });
  };

  const mergeProductResults = (products: ProductSummary[]): ProductSummary[] => {
    products.forEach((product) => {
      productsByNameRef.current.set(productKey(product.productName), product);
    });

    const nextProducts = Array.from(productsByNameRef.current.values());
    onProductsUpdate(nextProducts);

    const hasSelectedProduct =
      selectedProduct &&
      nextProducts.some((product) => productKey(product.productName) === productKey(selectedProduct.productName));
    if (nextProducts.length > 0 && !hasSelectedProduct) {
      onProductSelect(nextProducts[0]);
    }

    return nextProducts;
  };

  const productsFromImagePayload = (payload: Record<string, unknown>): ImageContent[] => {
    return Object.entries(payload)
      .map(([productName, productUrl]) => {
        const existing = productsByNameRef.current.get(productKey(productName));
        return {
          ...existing,
          productName,
          productUrl: String(productUrl),
        };
      })
      .filter((product) => product.productName && product.productUrl);
  };

  const enrichExistingImageRows = (products: ProductSummary[]) => {
    setMessages((prevMessages) =>
      prevMessages.map((message) => {
        if (message.role !== "image_row" || !Array.isArray(message.content)) {
          return message;
        }

        return {
          ...message,
          content: message.content.map((image) => {
            const product = products.find(
              (candidate) => productKey(candidate.productName) === productKey(image.productName)
            );
            return product ? { ...image, ...product, productUrl: image.productUrl } : image;
          }),
        };
      })
    );
  };

  const showStarters =
    !isLoading && !messages.some((message) => message.role === "user");

  const handleSendMessage = async (overrideText?: string) => {
    const outgoing = (overrideText ?? newMessage).trim();
    if (!outgoing && !image && !video) return;

    const userSession = getOrCreateUserSession();
    setConversationId(userSession.conversationId);
    // The previous turn's reading is not this turn's.
    setMediaAnalysis(null);
    setIsLoading(true);
    currentTurnHasMedia.current = Boolean(image || video);
    currentTurnGuardrails.current = isGuardrailsOn;

    // Will be used to enable submit shortly after the last token
    let enableSubmitTimer: number | undefined;

    try {
      // Enable-submit helper: if no tokens arrive for a short window, consider the stream done
      const scheduleEnableSubmit = () => {
        if (enableSubmitTimer !== undefined) {
          window.clearTimeout(enableSubmitTimer);
        }
        // Short idle threshold so the button enables promptly after the last token
        enableSubmitTimer = window.setTimeout(() => {
          setIsLoading(false);
        }, 400);
      };

      // Add user message
      if (outgoing) {
        addMessage("user", outgoing, "");
      }
      if (image) {
        addMessage("user_image", previewImage, "");
      }
      if (video) {
        addMessage("user_video", previewVideo, "");
      }

      // Add loading message
      addMessage("assistant", "loader", "");
      setNewMessage("");

      // Prepare API request
      const media: MediaAttachment[] = video
        ? [{
            type: "video",
            data: video,
            mime_type: videoMimeType,
            filename: videoFilename,
          }]
        : [];
      const payload = createApiRequest(
        userSession,
        outgoing,
        image || "",
        isGuardrailsOn,
        media,
        selectedShopperProfileId
      );
      
      // Clear media immediately after preparing payload
      setImage("");
      setPreviewImage("");
      setVideo("");
      setPreviewVideo("");
      setVideoMimeType("");
      setVideoFilename("");

      const url = `${config.api.baseUrl}${config.api.endpoints.stream}`;

      // Send request
      const response = await fetch(url, {
        method: "POST",
        mode: "cors",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        let errorMessage = `HTTP error! status: ${response.status}`;
        try {
          const errorPayload = await response.json();
          if (errorPayload?.detail) {
            errorMessage = String(errorPayload.detail);
          } else if (errorPayload?.message) {
            errorMessage = String(errorPayload.message);
          }
        } catch {
          errorMessage = response.statusText || errorMessage;
        }
        throw new Error(errorMessage);
      }

      if (!response.body) {
        throw new Error("No response body received from assistant.");
      }

      // Process streaming response
      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let fullResponse = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n').filter(line => line.startsWith('data:'));

        for (let line of lines) {
          const raw = line.replace(/^data:\s*/, '');
          
          if (raw === '[DONE]') {
            // Stream closed by server; enable submit immediately
            setIsLoading(false);
            return;
          }
          
          try {
            const { type, payload } = JSON.parse(raw);

            if (type === "products" && Array.isArray(payload)) {
              const products = payload
                .map(normalizeProduct)
                .filter((product): product is ProductSummary => product !== null);
              const mergedProducts = mergeProductResults(products);
              enrichExistingImageRows(mergedProducts);
              continue;
            }

            if (type === "media_analysis" && payload && typeof payload === "object") {
              // Arrives while the turn is still running, seconds before any
              // product, so the shopper sees their image being read.
              setMediaAnalysis(payload as MediaAnalysis);
              continue;
            }

            if (type === "metrics" && payload && typeof payload === "object") {
              const metricsPayload = payload as InferenceMetricsPayload;
              const turnTokens = metricsPayload.token_usage ?? null;
              const turnModelUsage = metricsPayload.model_usage ?? {};
              if (turnTokens) {
                const totals = sessionUsageRef.current;
                sessionUsageRef.current = {
                  modelCalls: totals.modelCalls + (turnTokens.model_calls || 0),
                  inputTokens: totals.inputTokens + (turnTokens.input_tokens || 0),
                  outputTokens: totals.outputTokens + (turnTokens.output_tokens || 0),
                  totalTokens: totals.totalTokens + (turnTokens.total_tokens || 0),
                };
              }
              sessionModelUsageRef.current = mergeModelUsage(
                sessionModelUsageRef.current,
                turnModelUsage
              );
              setModelUsage(sessionModelUsageRef.current);
              setSessionUsage(sessionUsageRef.current);
              continue;
            }

            if (type === "error") {
              toast.error(String(payload || "Assistant stream failed."));
              setMessages(prev => prev.filter(msg => msg.content !== "loader"));
              continue;
            }

            if (type === "images" && payload && typeof payload === "object") {
              const images = productsFromImagePayload(payload as Record<string, unknown>);
              mergeProductResults(images);

              setMessages(prev => {
                const updated = [...prev];
                const lastIndex = updated.length - 1;
                const imageRow = {
                  role: "image_row" as MessageRole,
                  content: images,
                  productName: "",
                };

                if (updated[lastIndex]?.content === "loader") {
                  updated[lastIndex] = imageRow;
                } else {
                  updated.push(imageRow);
                }
                return updated;
              });
              continue;
            }

            if (type === "content") {
              fullResponse += String(payload || "");
              const responseSnapshot = fullResponse;


              // Tokens are flowing; schedule enable when they stop
              scheduleEnableSubmit();

              setMessages(prev => {
                const updated = [...prev];
                const last = updated[updated.length - 1];

                if (last?.role === "assistant") {
                  updated[updated.length - 1] = {
                    ...last,
                    content: responseSnapshot
                  };
                } else {
                  updated.push({
                    role: "assistant",
                    content: responseSnapshot,
                    productName: ""
                  });
                  setLastAssistantIndex(updated.length - 1);
                }

                return updated;
              });
            }
          } catch (e) {
            continue;
          }
        }
      }
      
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = error instanceof Error
        ? error.message
        : 'Failed to send message. Please try again.';
      toast.error(errorMessage);
      
      // Remove loading message on error
      setMessages(prev => prev.filter(msg => msg.content !== 'loader'));
    } finally {
      // Clear any pending enable timer and ensure loading is false
      if (enableSubmitTimer !== undefined) window.clearTimeout(enableSubmitTimer);
      setIsLoading(false);
    }
  };

  const handleKeyUp = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter" && !isLoading) {
      handleSendMessage();
    }
  };

  const resetChat = async (clearIdentity: boolean) => {
    setMessages([]);
    setImage("");
    setPreviewImage("");
    setVideo("");
    setPreviewVideo("");
    setVideoMimeType("");
    setVideoFilename("");
    sessionUsageRef.current = {
      modelCalls: 0,
      inputTokens: 0,
      outputTokens: 0,
      totalTokens: 0,
    };
    sessionModelUsageRef.current = {};
    setModelUsage({});
    setSessionUsage(sessionUsageRef.current);
    productsByNameRef.current.clear();
    onProductSelect(null);
    onProductsUpdate([]);
    if (clearIdentity) {
      clearUserSession();
    }
    // Cleared, not reissued. Reading it back here would mint an identity just
    // to display one, and reset is meant to leave storage empty until the
    // shopper says something. It fills in on the first turn.
    setConversationId("");
    setMediaAnalysis(null);

    // Add welcome messages
    addMessage(
      "system",
      "You are an advanced AI assistant helps customers on a Retail e-commerce website. You help answer questions for customers about products. Start the conversation by asking a couple of questions to clarify what the user is looking for. Use emojis but do not use too many. Structure your output using Markdown but do not use nested indentations.",
      ""
    );
    
    await sleep(200);
    addMessage("assistant", "", "");

    const introduction =
      "Hello! \ud83d\udc4b I'm your shopping assistant. Ask me for an outfit, " +
      "a specific piece, or what is in your cart \u2014 or start with one of these.";

    // Fast enough to read as alive, not slow enough to wait through. This runs
    // on every reset as well as every new session.
    const words = introduction.split(" ");
    for (const word of words) {
      await sleep(12);
      updateLastMessage(word + " ");
    }
  };
  handleResetRef.current = resetChat;

  // Effects
  useEffect(() => {
    if (lastAssistantIndex !== null) {
      const messageRef = messageRefs.current[lastAssistantIndex];
      if (messageRef && messageRef.current) {
        messageRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }
    if (!isLoading) {
      inputRef.current?.focus();
    }
  }, [messages, isLoading, lastAssistantIndex]);

  useEffect(() => {
    if (isOpen) {
      setHasBeenOpened(true);
    }
  }, [isOpen]);

  useEffect(() => {
    const loadCapabilities = async () => {
      try {
        const response = await fetch(`${config.api.baseUrl}${config.api.endpoints.capabilities}`);
        if (!response.ok) return;
        const data = await response.json() as CapabilitiesResponse;
        if (data.media_input) {
          setMediaCapabilities(data.media_input);
        }
        setModelCapabilities(data.models ?? {});
      } catch (error) {
        console.warn("Failed to load media capabilities", error);
      }
    };
    loadCapabilities();
  }, []);

  useEffect(() => {
    if (!hasBeenOpened || initialResetStartedRef.current) return;

    initialResetStartedRef.current = true;
    void handleResetRef.current?.(!preserveIdentityOnMount);
  }, [hasBeenOpened, preserveIdentityOnMount]);

  return (
    <section className="chatbox">
      <div className={`chatbox__support ${isOpen ? "chatbox--active" : ""}`}>
        <div className="chatbox__header">
          <div>
            <h4 className="chatbox__heading--header">Retail Shopping Assistant</h4>
            <p className="chatbox__subheading--header">Your Shopping Concierge</p>
          </div>
          <div className="chatbox__brand">
            <span>Powered by</span>
            <img src={logo} alt="NVIDIA" />
          </div>
        </div>

        <div className="chatbox__messages">
          {[...messages].reverse().map((msg, index) => (
            <ChatMessage
              key={index}
              role={msg.role}
              content={msg.content}
              productName={msg.productName}
              selectedProductName={selectedProduct?.productName}
              onProductSelect={onProductSelect}
              ref={messageRefs.current[messages.length - 1 - index]}
            />
          ))}
        </div>

        {mediaAnalysis && (
          <div className="chatbox__media-analysis">
            <MediaAnalysisCard analysis={mediaAnalysis} />
          </div>
        )}

        {showStarters && (
          <div className="chatbox__starters" aria-label="Suggested openers">
            {STARTER_PROMPTS.map((prompt) => (
              <button
                key={prompt.label}
                type="button"
                className="chatbox__starter"
                disabled={isLoading}
                onClick={() => {
                  if (prompt.attachesMedia) {
                    // Needs a picture before it means anything, so fill the
                    // composer and open the picker rather than submitting.
                    setNewMessage(prompt.text);
                    mediaInputRef.current?.click();
                    return;
                  }
                  setNewMessage("");
                  void handleSendMessage(prompt.text);
                }}
              >
                {prompt.label}
              </button>
            ))}
          </div>
        )}

        <div className="chatbox__footer">
          {(previewImage || previewVideo) && (
            <div className="chatbox__preview-strip">
              {previewImage && (
                <div className="chatbox__preview">
                  <img src={previewImage} alt="Preview" />
                  <button type="button" onClick={clearImage} aria-label="Clear image">
                    <CloseIcon fontSize="small" />
                  </button>
                </div>
              )}

              {previewVideo && (
                <div className="chatbox__preview">
                  <video src={previewVideo} muted />
                  <button type="button" onClick={clearVideo} aria-label="Clear video">
                    <CloseIcon fontSize="small" />
                  </button>
                </div>
              )}
            </div>
          )}

          <div className="chatbox__composer">
            <input
              ref={inputRef}
              type="text"
              className="input_test"
              placeholder="Ask about products, outfits, prices, or your cart"
              value={newMessage}
              onChange={handleNewMessageChange}
              onKeyUp={handleKeyUp}
            />

            <button
              type="button"
              className="chatbox__icon-button"
              onClick={isLoading ? undefined : () => void handleSendMessage()}
              disabled={isLoading}
              aria-label="Send message"
            >
              <SendIcon fontSize="small" />
            </button>

            <label className="chatbox__icon-button" aria-label="Attach media">
              <AttachFileIcon fontSize="small" />
              <input
                ref={mediaInputRef}
                className="chatbox__file-input"
                type="file"
                accept={acceptedMediaTypes(mediaCapabilities)}
                name="media"
                onChange={handleImageUpload}
              />
            </label>

          </div>

          <div className="chatbox__guardrail">
            <span>Guardrails</span>
            <CustomSwitch checked={isGuardrailsOn} onChange={toggleGuardrails} size="small" />
          </div>
        </div>

        <ModelStrip
          models={modelCapabilities}
          modelUsage={modelUsage}
          sessionUsage={sessionUsage}
          conversationId={conversationId}
        />
      </div>

    </section>
  );
};

export default Chatbox;
