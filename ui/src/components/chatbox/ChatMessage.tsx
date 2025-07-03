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

/**
 * Chat message component for displaying different types of messages
 */

import React from "react";
import Showdown from "showdown";
import SafeHTML from "./SafeHTML";
import Loader from "./Loader";
import { ChatMessageProps, MessageRole, ImageContent, ImageRowContent } from "../../types";
import { isFashionMode } from "../../config/config";

const ChatMessage = React.forwardRef<HTMLDivElement, ChatMessageProps>(
  ({ role, content, productName }, ref) => {
    
    // CSS class mapping for markdown elements
    const classMap: Record<string, string> = {
      h1: `messages__item--${role}--h1`,
      h2: `messages__item--${role}--h2`,
      ul: `messages__item--${role}--ul`,
      li: `messages__item--${role}--li`,
      ol: `messages__item--${role}--ol`,
      p: `messages__item--${role}--p`,
    };

    // Create Showdown converter with custom extensions
    const bindings = Object.keys(classMap).map((key) => ({
      type: "output" as const,
      regex: new RegExp(`<${key}(.*)>`, "g"),
      replace: `<${key} class="${classMap[key]}" $1>`,
    }));

    const converter = new Showdown.Converter({
      extensions: [...bindings],
    });

    // Don't render system messages
    if (role === "system") {
      return null;
    }

    // User message
    if (role === "user") {
      return (
        <div className={`messages__item messages__item--${role}`} ref={ref}>
          <SafeHTML html={content as string} />
        </div>
      );
    }

    // Assistant message
    if (role === "assistant") {
      if (content === "loader") {
        return (
          <div className={`messages__item messages__item--${role}`} ref={ref}>
            <Loader />
          </div>
        );
      }

      return (
        <div className={`messages__item messages__item--${role}`} ref={ref}>
          <SafeHTML html={converter.makeHtml(content as string)} />
        </div>
      );
    }

    // Image message (single product)
    if (role === "image") {
      const [imagePath, url, productName, productRating] = (content as string).split("|");
      
      if (imagePath && url && productName && productRating) {
        return (
          <div className={`messages__item messages__item--${role}`} ref={ref}>
            <img className="messages__item--image-img" src={imagePath} alt={productName} />
            <div className="messages__item--image-box">
              <div className="messages__item--image-stars">
                {productName}
              </div>
            </div>
          </div>
        );
      }
    }

    // Image row message (multiple products)
    if (role === "image_row") {
      const images = content as ImageRowContent;
      
      return (
        <div style={{ 
          width: "100%", 
          height: "auto", 
          display: "inline-flex", 
          flexFlow: "row wrap" 
        }}>
          {images.map((image: ImageContent, index: number) => (
            <div key={index} className={`messages__item messages__item--image`} ref={ref}>
              <img 
                className="messages__item--image-img-rowitem" 
                src={image.productUrl} 
                alt={image.productName}
              />
              <div className="messages__item--image-box">
                <div 
                  className="messages__item--image-stars" 
                  style={{
                    maxWidth: isFashionMode() ? "200px" : "none"
                  }}
                >
                  {image.productName}
                </div>
              </div>
            </div>
          ))}
        </div>
      );
    }

    // User uploaded image
    if (role === "user_image" && content) {
      return (
        <div className={`messages__item messages__item--${role}`} ref={ref}>
          <img 
            className="messages__item--image-img" 
            src={content as string} 
            alt="User uploaded image"
            style={{ borderRadius: "20px" }} 
          />
        </div>
      );
    }

    return null;
  }
);

ChatMessage.displayName = "ChatMessage";

export default ChatMessage;
