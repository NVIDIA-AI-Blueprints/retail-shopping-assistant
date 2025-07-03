/**
 * Custom hook for managing chat state and API interactions
 */

import React, { useState, useCallback, useRef } from 'react';
import { toast } from 'react-toastify';
import { MessageData, ApiRequest, StreamingChunk, MessageRole } from '../types';
import { config } from '../config/config';
import { 
  getOrCreateUserId, 
  createApiRequest, 
  parseStreamingChunk,
  sleep,
  handleFileUpload,
  clearUserSession,
  downloadMessages as downloadMessagesUtil
} from '../utils';

interface UseChatReturn {
  messages: MessageData[];
  isLoading: boolean;
  userId: number;
  isGuardrailsOn: boolean;
  image: string;
  previewImage: string;
  sendMessage: (query: string, image?: string) => Promise<void>;
  addMessage: (role: MessageRole, content: any, productName?: string) => void;
  updateLastMessage: (newContent: any, role?: MessageRole, appendContent?: boolean) => void;
  handleImageUpload: (file: File) => Promise<void>;
  clearImage: () => void;
  toggleGuardrails: () => void;
  resetChat: () => Promise<void>;
  downloadMessages: () => void;
}

export const useChat = (setNewRenderImage: (value: string) => void): UseChatReturn => {
  const [userId, setUserId] = useState(getOrCreateUserId());
  const [messages, setMessages] = useState<MessageData[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isGuardrailsOn, setIsGuardrailsOn] = useState(config.features.guardrails.defaultState);
  const [image, setImage] = useState("");
  const [previewImage, setPreviewImage] = useState("");
  const [lastAssistantIndex, setLastAssistantIndex] = useState<number | null>(null);
  const messageRefs = useRef<React.RefObject<HTMLDivElement>[]>([]);

  const addMessage = useCallback((role: MessageRole, content: any, productName: string = "") => {
    setMessages((prevMessages: MessageData[]) => {
      const newMessages = [...prevMessages, { role, content, productName }];
      messageRefs.current = newMessages.map((_, i) => 
        messageRefs.current[i] || React.createRef<HTMLDivElement>()
      );
      
      if (role === "assistant" && (lastAssistantIndex === null || lastAssistantIndex < prevMessages.length)) {
        setLastAssistantIndex(prevMessages.length);
      }
      
      return newMessages;
    });
  }, [lastAssistantIndex]);

  const updateLastMessage = useCallback((newContent: any, role?: MessageRole, appendContent?: boolean) => {
    setMessages((prevMessages: MessageData[]) => {
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
  }, []);

  const handleImageUpload = useCallback(async (file: File) => {
    try {
      const { base64, previewUrl } = await handleFileUpload(file);
      setImage(base64);
      setPreviewImage(previewUrl);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Failed to upload image');
    }
  }, []);

  const clearImage = useCallback(() => {
    setImage("");
    setPreviewImage("");
  }, []);

  const toggleGuardrails = useCallback(() => {
    setIsGuardrailsOn(!isGuardrailsOn);
  }, [isGuardrailsOn]);

  const sendMessage = useCallback(async (query: string, imageData?: string) => {
    if (!query.trim() && !imageData) return;

    setIsLoading(true);
    
    try {
      // Add user message
      if (query) {
        addMessage("user" as MessageRole, query, "");
      }
      if (imageData) {
        addMessage("user_image" as MessageRole, previewImage, "");
      }

      // Add loading message
      addMessage("assistant" as MessageRole, "loader", "");

      // Prepare API request
      const payload = createApiRequest(userId, query, imageData || image, isGuardrailsOn);
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
          const errorData = await response.json();
          errorMessage = errorData.detail || errorData.message || errorMessage;
        } catch (e) {
          // If we can't parse the error response, use the status text
          errorMessage = response.statusText || errorMessage;
        }
        throw new Error(errorMessage);
      }

      if (!response.body) {
        throw new Error('No response body received');
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
          const parsedChunk = parseStreamingChunk(raw);
          
          if (!parsedChunk) continue;

          if (parsedChunk.type === 'content') {
            fullResponse += parsedChunk.payload as string;
          } else if (parsedChunk.type === 'images') {
            const images = Object.entries(parsedChunk.payload as Record<string, string>)
              .map(([productName, productUrl]) => ({ productUrl, productName }));
            
            setMessages((prev: MessageData[]) => {
              const updated = [...prev];
              updated[updated.length - 1] = {
                ...updated[updated.length - 1],
                role: 'image_row' as MessageRole,
                content: images,
              };
              return updated;
            });
          } else if (parsedChunk.type === 'error') {
            throw new Error(parsedChunk.payload as string);
          }

          // Update assistant message
          setMessages((prev: MessageData[]) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];

            if (last?.role === 'assistant') {
              updated[updated.length - 1] = {
                ...last,
                content: fullResponse
              };
            } else {
              updated.push({
                role: 'assistant' as MessageRole,
                content: fullResponse,
                productName: ""
              });
              setLastAssistantIndex(updated.length - 1);
            }

            return updated;
          });
        }
      }

      // Clear input
      setImage("");
      setPreviewImage("");
      
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to send message. Please try again.';
      
      // Remove loading message and add error message
      setMessages((prev: MessageData[]) => {
        const filtered = prev.filter((msg: MessageData) => msg.content !== 'loader');
        return [...filtered, {
          role: 'assistant' as MessageRole,
          content: `❌ Error: ${errorMessage}`,
          productName: ""
        }];
      });
      
      toast.error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [userId, isGuardrailsOn, image, previewImage, addMessage]);

  const resetChat = useCallback(async () => {
    setMessages([]);
    setImage("");
    setPreviewImage("");
    setNewRenderImage("");
    clearUserSession();
    setUserId(getOrCreateUserId());

    // Add welcome messages
    addMessage(
      "system" as MessageRole,
      "You are an advanced AI assistant helps customers on a Retail e-commerce website. You help answer questions for customers about products. Start the conversation by asking a couple of questions to clarify what the user is looking for. Use emojis but do not use too many. Structure your output using Markdown but do not use nested indentations.",
      ""
    );
    
    await sleep(1000);
    addMessage("assistant" as MessageRole, "", "");
    
    await sleep(1000);
    const introduction = "Hello! 👋 I'm your dedicated Shopping Assistant created by NVIDIA, here to answer any questions you might have and help you find anything you're looking for. What can I help you with today?";
    
    const words = introduction.split(" ");
    for (const word of words) {
      await sleep(40);
      updateLastMessage(word + " ");
    }
  }, [addMessage, updateLastMessage, setNewRenderImage]);

  const downloadMessages = useCallback(() => {
    downloadMessagesUtil(messages);
  }, [messages]);

  return {
    messages,
    isLoading,
    userId,
    isGuardrailsOn,
    image,
    previewImage,
    sendMessage,
    addMessage,
    updateLastMessage,
    handleImageUpload,
    clearImage,
    toggleGuardrails,
    resetChat,
    downloadMessages,
  };
}; 