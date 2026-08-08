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

/**
 * Chat message component for displaying different types of messages
 */

import React from "react";
import Showdown from "showdown";
import SafeHTML from "./SafeHTML";
import Loader from "./Loader";
import MediaAnalysisCard from "./MediaAnalysisCard";
import {
  ChatMessageProps,
  ImageContent,
  ImageRowContent,
  MediaAnalysis,
} from "../../types";
import { isFashionMode } from "../../config/config";
import nvinfo from "../../assets/nvinfo.jpg";

const ChatMessage = React.forwardRef<HTMLDivElement, ChatMessageProps>(
  ({ role, content, productName, selectedProductName, onProductSelect }, ref) => {
    
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
      simpleLineBreaks: true,  // This will convert single line breaks to <br>
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
          <div ref={ref} style={{ display: "inline-flex", alignItems: "flex-start", gap: 8, marginTop: 10 }}>
            <img src={nvinfo} alt="Assistant" style={{ width: 28, height: 28, borderRadius: "50%", objectFit: "cover" }} />
            <div className={`messages__item messages__item--${role}`}>
              <Loader />
            </div>
          </div>
        );
      }

      // Preprocess to convert list markers at the beginning of lines
      let preprocessedContent = (content as string)
        .replace(/^\* /gm, '• ')           // Convert * to bullet
        .replace(/^- /gm, '• ')             // Convert - to bullet
        .replace(/^\d+\. /gm, (match) => { // Keep numbered lists as-is
          return match;
        });
      
      // Then use the Markdown converter to handle all formatting including bold
      const processedContent = converter.makeHtml(preprocessedContent);

      return (
        <div ref={ref} style={{ display: "inline-flex", alignItems: "flex-start", gap: 8, marginTop: 10 }}>
          <img src={nvinfo} alt="Assistant" style={{ width: 28, height: 28, borderRadius: "50%", objectFit: "cover" }} />
          <div className={`messages__item messages__item--${role}`}>
            <SafeHTML html={processedContent} />
          </div>
        </div>
      );
    }

    // Image message (single product)
    if (role === "image") {
      const [imagePath, url, imageProductName, productRating] = (content as string).split("|");
      
      if (imagePath && url && imageProductName && productRating) {
        const product = {
          productName: imageProductName,
          productUrl: imagePath,
        };

        return (
          <div className={`messages__item messages__item--${role}`} ref={ref}>
            <button
              type="button"
              className="product-result-card"
              onClick={() => onProductSelect?.(product)}
            >
              <img className="product-result-card__image" src={imagePath} alt={imageProductName} />
              <span className="product-result-card__name">{imageProductName}</span>
            </button>
          </div>
        );
      }
    }

    // What the vision model saw. In the message stream rather than beside it,
    // so it sits above the results it precedes: it arrives seconds after the
    // upload and the products land a minute later.
    if (role === "media_analysis") {
      return (
        <div className="messages__item messages__item--media-analysis" ref={ref}>
          <MediaAnalysisCard analysis={content as MediaAnalysis} />
        </div>
      );
    }

    // Image row message (multiple products)
    if (role === "image_row") {
      const images = content as ImageRowContent;
      
      return (
        <div className="product-result-grid" ref={ref}>
          {images.map((image: ImageContent, index: number) => (
            <div key={`${image.productName}-${index}`} className="messages__item messages__item--image">
              <button
                type="button"
                className={`product-result-card${
                  image.productName === selectedProductName ? " is-selected" : ""
                }`}
                onClick={() => onProductSelect?.(image)}
              >
                <img
                  className="product-result-card__image"
                  src={image.productUrl}
                  alt={image.productName}
                />
                <span
                  className="product-result-card__name"
                  style={{
                    maxWidth: isFashionMode() ? "200px" : "none",
                  }}
                >
                  {image.productName}
                </span>
                {image.price && (
                  <span className="product-result-card__price">
                    {formatCardPrice(image.price)}
                  </span>
                )}
              </button>
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
            alt="User upload"
            style={{ borderRadius: "20px" }} 
          />
        </div>
      );
    }

    if (role === "user_video" && content) {
      return (
        <div className={`messages__item messages__item--${role}`} ref={ref}>
          <video
            src={content as string}
            controls
            muted
            style={{ width: "180px", maxHeight: "140px", borderRadius: "20px" }}
          />
        </div>
      );
    }

    return null;
  }
);

ChatMessage.displayName = "ChatMessage";

const formatCardPrice = (price: { amount: number; currency?: string }): string => {
  try {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: price.currency || "USD",
    }).format(price.amount);
  } catch {
    return `$${price.amount.toFixed(2)}`;
  }
};

export default ChatMessage;
