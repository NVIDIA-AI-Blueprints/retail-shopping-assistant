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
import RestartAltIcon from "@mui/icons-material/RestartAlt";
import Switch from '@mui/material/Switch';
import { styled } from '@mui/material/styles';

import ChatMessage from "./ChatMessage";
import InferenceActivityPanel from "./InferenceActivityPanel";
import {
  CapabilitiesResponse,
  ChatboxProps,
  ImageContent,
  InferenceActivity,
  InferenceMetricsPayload,
  MediaAttachment,
  MediaCapabilities,
  MessageData,
  MessageRole,
  ModelCapabilities,
  ModelUsage,
  ProductPrice,
  ProductSummary,
  TokenUsage,
} from "../../types";
import { config } from "../../config/config";
import {
  clearUserSession,
  createApiRequest,
  getOrCreateUserSession,
  showCartNotification,
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

const modelName = (models: ModelCapabilities, role: string): string | undefined => {
  const model = models[role];
  return model?.enabled && model.model ? model.model : undefined;
};

const joinedModelNames = (models: ModelCapabilities, roles: string[]): string | undefined => {
  const names = roles
    .map((role) => modelName(models, role))
    .filter((name): name is string => Boolean(name));
  return names.length > 0 ? names.join(" / ") : undefined;
};

const createPendingActivities = (
  hasMedia: boolean,
  guardrailsEnabled: boolean,
  models: ModelCapabilities
): InferenceActivity[] => {
  const events: InferenceActivity[] = [
    {
      id: "memory-pending",
      category: "memory",
      label: "Conversation memory",
      detail: "Session and cart context",
      status: "running",
    },
    {
      id: "language-pending",
      category: "language",
      label: "Language reasoning",
      detail: "Planning, tool use, and response generation",
      modelName: modelName(models, "app_llm"),
      status: "running",
    },
  ];

  if (hasMedia) {
    events.splice(1, 0, {
      id: "vision-pending",
      category: "vision",
      label: "Vision-language inference",
      detail: "Attached media understanding",
      modelName: modelName(models, "vlm"),
      status: "running",
    });
  }

  if (guardrailsEnabled) {
    events.push({
      id: "safety-pending",
      category: "safety",
      label: "Safety inference",
      detail: "Input and output checks",
      modelName: joinedModelNames(models, ["content_safety", "topic_control"]),
      status: "queued",
    });
  }

  return events;
};

const activitiesFromMetrics = (
  metrics: InferenceMetricsPayload,
  hasMedia: boolean,
  guardrailsEnabled: boolean,
  models: ModelCapabilities
): InferenceActivity[] => {
  const timings = metrics.timings || {};
  const activities: InferenceActivity[] = [];
  const addTiming = (
    key: string,
    category: InferenceActivity["category"],
    label: string,
    detail: string,
    modelName?: string,
    status: InferenceActivity["status"] = "complete"
  ) => {
    const seconds = Number(timings[key]);
    if (!Number.isFinite(seconds)) return;
    activities.push({
      id: key,
      category,
      label,
      detail,
      modelName,
      status,
      durationMs: Math.max(0, seconds * 1000),
    });
  };

  addTiming("media_perception", "vision", "Vision-language inference", "Attached media understanding", modelName(models, "vlm"));
  addTiming("catalog_search", "embedding", "Embeddings and vector search", "Catalog retrieval workload", joinedModelNames(models, ["text_embedding", "image_embedding"]));
  addTiming("deepagents", "language", "Language reasoning", "Planning, tool use, and response generation", modelName(models, "app_llm"));
  addTiming("deepagents_error", "language", "Language reasoning", "Planning, tool use, and response generation", modelName(models, "app_llm"), "failed");
  addTiming("memory", "memory", "Conversation memory", "Session and cart context");
  addTiming("safety_input", "safety", "Input safety", "Guardrails check", joinedModelNames(models, ["content_safety", "topic_control"]));
  addTiming("safety_output", "safety", "Output safety", "Guardrails check", joinedModelNames(models, ["content_safety", "topic_control"]));

  if (activities.length === 0) {
    return createPendingActivities(hasMedia, guardrailsEnabled, models).map((event) => ({
      ...event,
      status: "complete",
    }));
  }

  return activities;
};

const failedActivities = (events: InferenceActivity[]): InferenceActivity[] => {
  if (events.length === 0) {
    return [
      {
        id: "request-failed",
        category: "system",
        label: "Assistant request",
        detail: "The turn did not complete",
        status: "failed",
      },
    ];
  }
  return events.map((event) => ({
    ...event,
    status: event.status === "complete" ? event.status : "failed",
  }));
};

const Chatbox: React.FC<ChatboxProps> = ({
  selectedProduct,
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
  const [modelCapabilities, setModelCapabilities] = useState<ModelCapabilities>({});
  const [modelUsage, setModelUsage] = useState<ModelUsage>({});
  const [messages, setMessages] = useState<MessageData[]>([]);
  const [inferenceEvents, setInferenceEvents] = useState<InferenceActivity[]>([]);
  const [tokenUsage, setTokenUsage] = useState<TokenUsage | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const messageRefs = useRef<React.RefObject<HTMLDivElement>[]>([]);
  const [lastAssistantIndex, setLastAssistantIndex] = useState<number | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const shownCartOperations = useRef<Set<string>>(new Set());
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

  const handleSendMessage = async () => {
    if (!newMessage.trim() && !image && !video) return;

    // Clear previous cart operation notifications for new message
    shownCartOperations.current.clear();

    const userSession = getOrCreateUserSession();
    setIsLoading(true);
    currentTurnHasMedia.current = Boolean(image || video);
    currentTurnGuardrails.current = isGuardrailsOn;
    setInferenceEvents(
      createPendingActivities(currentTurnHasMedia.current, isGuardrailsOn, modelCapabilities)
    );
    setTokenUsage(null);
    setModelUsage({});

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
      if (newMessage) {
        addMessage("user", newMessage, "");
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
        newMessage,
        image || "",
        isGuardrailsOn,
        media
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

            if (type === "metrics" && payload && typeof payload === "object") {
              const metricsPayload = payload as InferenceMetricsPayload;
              setTokenUsage(metricsPayload.token_usage ?? null);
              setModelUsage(metricsPayload.model_usage ?? {});
              setInferenceEvents(
                activitiesFromMetrics(
                  metricsPayload,
                  currentTurnHasMedia.current,
                  currentTurnGuardrails.current,
                  modelCapabilities
                )
              );
              continue;
            }

            if (type === "error") {
              setInferenceEvents((prev) => failedActivities(prev));
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

              // Check for cart operations and show notifications
              showCartNotification(responseSnapshot, shownCartOperations.current, toast);

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
    setInferenceEvents([]);
    setTokenUsage(null);
    setModelUsage({});
    productsByNameRef.current.clear();
    onProductSelect(null);
    onProductsUpdate([]);
    if (clearIdentity) {
      clearUserSession();
    }

    // Add welcome messages
    addMessage(
      "system",
      "You are an advanced AI assistant helps customers on a Retail e-commerce website. You help answer questions for customers about products. Start the conversation by asking a couple of questions to clarify what the user is looking for. Use emojis but do not use too many. Structure your output using Markdown but do not use nested indentations.",
      ""
    );
    
    await sleep(1000);
    addMessage("assistant", "", "");
    
    await sleep(1000);
    const introduction = "Hello! 👋 I'm your dedicated Shopping Assistant created by NVIDIA. You can ask me anything—from finding the perfect item to learning more about product care.\n\nHere are some questions you could ask me:\n\n• Show me items under $100\n• Does the [product name] require dry cleaning?\n• Do you have anything like this? (upload an image)\n• Add the first item to my cart";
    
    const words = introduction.split(" ");
    for (const word of words) {
      await sleep(40);
      updateLastMessage(word + " ");
    }
  };
  handleResetRef.current = resetChat;

  const handleReset = () => {
    void resetChat(true);
  };

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
        if (data.models) {
          setModelCapabilities(data.models);
        }
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
              onClick={isLoading ? undefined : handleSendMessage}
              disabled={isLoading}
              aria-label="Send message"
            >
              <SendIcon fontSize="small" />
            </button>

            <label className="chatbox__icon-button" aria-label="Attach media">
              <AttachFileIcon fontSize="small" />
              <input
                className="chatbox__file-input"
                type="file"
                accept={acceptedMediaTypes(mediaCapabilities)}
                name="media"
                onChange={handleImageUpload}
              />
            </label>

            <button
              type="button"
              className="chatbox__icon-button"
              onClick={handleReset}
              aria-label="Reset conversation"
            >
              <RestartAltIcon fontSize="small" />
            </button>
          </div>

          <div className="chatbox__guardrail">
            <span>Guardrails</span>
            <CustomSwitch checked={isGuardrailsOn} onChange={toggleGuardrails} size="small" />
          </div>
        </div>
      </div>

      <InferenceActivityPanel
        events={inferenceEvents}
        models={modelCapabilities}
        tokenUsage={tokenUsage}
        modelUsage={modelUsage}
      />
    </section>
  );
};

export default Chatbox;
