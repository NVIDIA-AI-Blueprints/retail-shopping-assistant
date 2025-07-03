/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import { ChatboxProps } from "../../types";
import { config } from "../../config/config";
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
  const [isOpen, setIsOpen] = useState<boolean>(true);
  const [hasBeenOpened, setHasBeenOpened] = useState<boolean>(false);
  const [newMessage, setNewMessage] = useState<string>("");
  const [isGuardrailsOn, setIsGuardrailsOn] = useState(config.features.guardrails.defaultState);
  const [image, setImage] = useState("");
  const [previewImage, setPreviewImage] = useState("");
  const [messages, setMessages] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const messageRefs = useRef<React.RefObject<HTMLDivElement>[]>([]);
  const [lastAssistantIndex, setLastAssistantIndex] = useState<number | null>(null);

  // Utility functions
  const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

  const getOrCreateUserId = () => {
    const storedId = sessionStorage.getItem('shopping_user_id');
    if (storedId) return parseInt(storedId, 10);
    
    const newId = Math.floor(Math.random() * 100000);
    sessionStorage.setItem('shopping_user_id', String(newId));
    return newId;
  };

  const convertToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = (error) => reject(error);
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
    
    // Validate file
    const maxSizeMB = config.features.imageUpload.maxSize;
    if (file.size > maxSizeMB * 1024 * 1024) {
      toast.error(`File size must be less than ${maxSizeMB}MB`);
      return;
    }

    if (!config.features.imageUpload.allowedTypes.includes(file.type)) {
      toast.error('Please select a valid image file (JPEG, PNG, GIF, or WebP)');
      return;
    }

    try {
      const base64Image = await convertToBase64(file);
      setImage(base64Image);
      
      const decodedImage = base64ToBlob(base64Image);
      const imageUrl = window.URL.createObjectURL(decodedImage);
      setPreviewImage(imageUrl);
      
      e.target.value = "";
    } catch (error) {
      toast.error('Failed to upload image');
    }
  };

  const clearImage = () => {
    setPreviewImage("");
    setImage("");
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
    if (!newMessage.trim() && !image) return;

    const userId = getOrCreateUserId();
    setIsLoading(true);

    try {
      // Add user message
      if (newMessage) {
        addMessage("user", newMessage, "");
      }
      if (image) {
        addMessage("user_image", previewImage, "");
      }

      // Add loading message
      addMessage("assistant", "loader", "");
      setNewMessage("");

      // Prepare API request
      const payload = {
        user_id: userId,
        query: newMessage,
        guardrails: isGuardrailsOn,
        image: image || "",
        image_bool: !!image
      };

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
            return;
          }
          
          try {
            const { type, payload } = JSON.parse(raw);
            
            if (type === 'content') {
              fullResponse += payload;
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

      // Clear input
      setImage("");
      setPreviewImage("");
      
    } catch (error) {
      console.error('Error sending message:', error);
      toast.error('Failed to send message. Please try again.');
      
      // Remove loading message on error
      setMessages(prev => prev.filter(msg => msg.content !== 'loader'));
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyUp = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter") {
      handleSendMessage();
    }
  };

  const handleReset = async () => {
    setMessages([]);
    setImage("");
    setPreviewImage("");
    setNewRenderImage("");
    sessionStorage.removeItem('shopping_user_id');

    // Add welcome messages
    addMessage(
      "system",
      "You are an advanced AI assistant helps customers on a Retail e-commerce website. You help answer questions for customers about products. Start the conversation by asking a couple of questions to clarify what the user is looking for. Use emojis but do not use too many. Structure your output using Markdown but do not use nested indentations.",
      ""
    );
    
    await sleep(1000);
    addMessage("assistant", "", "");
    
    await sleep(1000);
    const introduction = "Hello! 👋 I'm your dedicated Shopping Assistant created by NVIDIA, here to answer any questions you might have and help you find anything you're looking for. What can I help you with today?";
    
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
  }, [messages]);

  useEffect(() => {
    if (isOpen) {
      setHasBeenOpened(true);
    }
  }, [isOpen]);

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
              <div 
                style={{ 
                  backgroundImage: `url(${previewImage})`, 
                  backgroundSize: "cover", 
                  backgroundPosition: "center", 
                  backgroundRepeat: "no-repeat", 
                  display: "inline-flex", 
                  height: "100px", 
                  width: "100px", 
                  position: "relative" 
                }}
              >
                <div 
                  style={{
                    display: "inline-flex", 
                    position: "absolute", 
                    right: "-5px", 
                    top: "-5px",
                    cursor: 'pointer'
                  }} 
                  onClick={clearImage}
                >
                  <FontAwesomeIcon icon={faTimesCircle} />
                </div>
              </div>
            )}

            {/* Input field */}
            <input
              type="text"
              className="input_test"
              placeholder="Type something here..."
              value={newMessage}
              onChange={handleNewMessageChange}
              onKeyUp={handleKeyUp}
              disabled={isLoading}
            />

            {/* Action buttons */}
            <div className="button-class">
              <SendIcon
                sx={{ color: "#76B900" }}
                onClick={handleSendMessage}
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
                accept="image/*"
                id="image-upload"
                name="image"
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
