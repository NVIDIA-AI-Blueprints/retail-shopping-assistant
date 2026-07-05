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
import CancelIcon from "@mui/icons-material/Cancel";
import DownloadIcon from "@mui/icons-material/Download";
import FormGroup from '@mui/material/FormGroup';
import FormControlLabel from '@mui/material/FormControlLabel';
import Switch from '@mui/material/Switch';
import { styled } from '@mui/material/styles';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faTimesCircle } from '@fortawesome/free-solid-svg-icons';

import ChatMessage from "./ChatMessage";
import { CapabilitiesResponse, ChatboxProps, MediaAttachment, MediaCapabilities } from "../../types";
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

const Chatbox: React.FC<ChatboxProps> = ({ setNewRenderImage }) => {
  const defaultMediaCapabilities: MediaCapabilities = {
    enabled: config.features.imageUpload.enabled,
    allow_mixed_media: true,
    max_images_per_turn: 1,
    max_videos_per_turn: 0,
    image_mime_types: config.features.imageUpload.allowedTypes,
    video_mime_types: [],
    max_image_bytes: config.features.imageUpload.maxSize * 1024 * 1024,
    max_video_bytes: 0,
    max_video_duration_seconds: 120,
    vlm_enabled: false,
  };
  const [isOpen, setIsOpen] = useState<boolean>(true);
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
  const [messages, setMessages] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const messageRefs = useRef<React.RefObject<HTMLDivElement>[]>([]);
  const [lastAssistantIndex, setLastAssistantIndex] = useState<number | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const shownCartOperations = useRef<Set<string>>(new Set());

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

    const isImage = mediaCapabilities.image_mime_types.includes(file.type);
    const isVideo = mediaCapabilities.video_mime_types.includes(file.type);
    const videoAllowed = mediaCapabilities.vlm_enabled && mediaCapabilities.max_videos_per_turn > 0;

    if (!isImage && !isVideo) {
      toast.error("Please select a supported image or video file.");
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
        setImage(base64Media);
        const decodedImage = base64ToBlob(base64Media);
        const imageUrl = window.URL.createObjectURL(decodedImage);
        setPreviewImage(imageUrl);
      } else {
        setVideo(base64Media);
        setPreviewVideo(window.URL.createObjectURL(file));
        setVideoMimeType(file.type);
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

  const addMessage = (role: string, content: any, productName: string = "") => {
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

  const updateLastMessage = (newContent: any, role?: string, appendContent?: boolean) => {
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

  const handleSendMessage = async () => {
    if (!newMessage.trim() && !image && !video) return;

    // Clear previous cart operation notifications for new message
    shownCartOperations.current.clear();

    const userSession = getOrCreateUserSession();
    setIsLoading(true);

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

      if (!response.ok || !response.body) {
        throw new Error(`HTTP error! status: ${response.status}`);
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
            
            if (type === 'content') {
              fullResponse += payload;
              
              // Check for cart operations and show notifications
              showCartNotification(fullResponse, shownCartOperations.current, toast);

              // Tokens are flowing; schedule enable when they stop
              scheduleEnableSubmit();
            } else if (type === 'images') {
              const images = Object.entries(payload).map(([productName, productUrl]) => ({ 
                productUrl, 
                productName 
              }));
              
              setMessages(prev => {
                const updated = [...prev];
                updated[updated.length - 1] = {
                  ...updated[updated.length - 1],
                  role: 'image_row',
                  content: images,
                };
                return updated;
              });
            }

            // Update assistant message
            setMessages(prev => {
              const updated = [...prev];
              const last = updated[updated.length - 1];

              if (last?.role === 'assistant') {
                updated[updated.length - 1] = {
                  ...last,
                  content: fullResponse
                };
              } else {
                updated.push({
                  role: 'assistant',
                  content: fullResponse,
                  productName: ""
                });
                setLastAssistantIndex(updated.length - 1);
              }

              return updated;
            });
          } catch (e) {
            continue;
          }
        }
      }
      
    } catch (error) {
      console.error('Error sending message:', error);
      toast.error('Failed to send message. Please try again.');
      
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

  const handleReset = async () => {
    setMessages([]);
    setImage("");
    setPreviewImage("");
    setVideo("");
    setPreviewVideo("");
    setVideoMimeType("");
    setVideoFilename("");
    setNewRenderImage("");
    clearUserSession();

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
  }, [messages, isLoading]);

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
      } catch (error) {
        console.warn("Failed to load media capabilities", error);
      }
    };
    loadCapabilities();
  }, []);

  useEffect(() => {
    if (hasBeenOpened) {
      handleReset();
    }
  }, [hasBeenOpened]);

  return (
    <div>
      <div className="chatbox">
        <div className={`chatbox__support ${isOpen ? "chatbox--active" : ""}`}>
          {/* Header */}
          <div className="chatbox__header">
            <h4 className="chatbox__heading--header">
              Retail Shopping Assistant
            </h4>
          </div>

          {/* Messages */}
          <div className="chatbox__messages">
            {[...messages].reverse().map((msg, index) => (
              <ChatMessage 
                key={index} 
                role={msg.role} 
                content={msg.content} 
                productName={msg.productName} 
                ref={messageRefs.current[messages.length - 1 - index]} 
              />
            ))}
          </div>

          {/* Footer */}
          <div className="chatbox__footer">
             {/* Image preview */}
             {previewImage && (
              <div style={{ position: 'relative', display: 'inline-block' }}>
                <img src={previewImage} alt="Preview" style={{ width: '50px', height: '50px' }} />
                <button
                  type="button"
                  style={{
                    display: 'inline-flex',
                    position: 'absolute',
                    right: '-5px',
                    top: '-5px',
                    cursor: 'pointer',
                    background: 'transparent',
                    border: 'none',
                    padding: 0,
                  }}
                  onClick={clearImage}
                  aria-label="Clear image"
                >
                  <FontAwesomeIcon icon={faTimesCircle} />
                </button>
              </div>
            )}

            {previewVideo && (
              <div style={{ position: 'relative', display: 'inline-block' }}>
                <video src={previewVideo} muted style={{ width: '50px', height: '50px', objectFit: 'cover' }} />
                <button
                  type="button"
                  style={{
                    display: 'inline-flex',
                    position: 'absolute',
                    right: '-5px',
                    top: '-5px',
                    cursor: 'pointer',
                    background: 'transparent',
                    border: 'none',
                    padding: 0,
                  }}
                  onClick={clearVideo}
                  aria-label="Clear video"
                >
                  <FontAwesomeIcon icon={faTimesCircle} />
                </button>
              </div>
            )}

            {/* Input field */}
            <input
              ref={inputRef}
              type="text"
              className="input_test"
              placeholder="Type something here..."
              value={newMessage}
              onChange={handleNewMessageChange}
              onKeyUp={handleKeyUp}
            />

            {/* Action buttons */}
            <div className="button-class">
              <SendIcon
                sx={{ color: isLoading ? "lightgray" : "#76B900", cursor: isLoading ? "not-allowed" : "pointer" }}
                onClick={isLoading ? () => {} : handleSendMessage}
                fontSize="large"
              />
            </div>
            
            <div className="button-class">
              <CancelIcon
                sx={{ color: "#76B900" }}
                onClick={handleReset}
                fontSize="large"
              />
            </div>
            
            <div className="button-class" style={{ transform: "rotate(180deg)" }}>
              <label htmlFor="image-upload" style={{ cursor: "pointer" }}>
                <DownloadIcon
                  sx={{ color: "#76B900" }}
                  fontSize="large"
                />
              </label>
              <input
                style={{ display: "none" }}
                type="file"
                accept={[
                  ...mediaCapabilities.image_mime_types,
                  ...(mediaCapabilities.vlm_enabled ? mediaCapabilities.video_mime_types : []),
                ].join(",")}
                id="image-upload"
                name="media"
                onChange={handleImageUpload}
              />
            </div>
          </div>

          {/* Guardrails toggle */}
          <div className="chatbox__guardrail">
            <FormGroup>
              <FormControlLabel 
                control={
                  <CustomSwitch 
                    checked={isGuardrailsOn}
                    onChange={toggleGuardrails}
                  />
                } 
                label="Guardrails" 
              />
            </FormGroup>
          </div>

          {/* Powered by NVIDIA */}
          <div className="flex relative flex-row items-center justify-center bg-white pb-[15px]">
            <h3 className="text-[16px]">Powered by</h3>
            <img src={logo} alt="NVIDIA" className="h-14" />
          </div>
        </div>

        {/* Chatbox toggle button (hidden) */}
        <div className="chatbox__button" style={{ visibility: "hidden" }}>
          <button onClick={() => setIsOpen(!isOpen)}>
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Chat_icon.svg/44px-Chat_icon.svg.png" alt="Chat" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default Chatbox;
