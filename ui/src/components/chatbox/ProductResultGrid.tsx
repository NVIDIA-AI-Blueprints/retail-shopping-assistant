/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from "react";
import { ChatMessageProps, ImageContent, ImageRowContent } from "../../types";
import { isFashionMode } from "../../config/config";

// A search can return 36 products; the full set stays one click away here and
// in the left panel's recent results.
export const COLLAPSED_PRODUCT_COUNT = 6;

type ProductResultGridProps = Pick<ChatMessageProps, "selectedProductName" | "onProductSelect"> & {
  images: ImageRowContent;
};

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

const ProductResultGrid = React.forwardRef<HTMLDivElement, ProductResultGridProps>(
  ({ images, selectedProductName, onProductSelect }, ref) => {
    const [expanded, setExpanded] = useState(false);
    const hiddenCount = images.length - COLLAPSED_PRODUCT_COUNT;
    const visible = expanded || hiddenCount <= 0 ? images : images.slice(0, COLLAPSED_PRODUCT_COUNT);

    return (
      <div className="product-result-block" ref={ref}>
        <div className="product-result-grid">
          {visible.map((image: ImageContent, index: number) => (
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
        {hiddenCount > 0 && (
          <button
            type="button"
            className="product-result-toggle"
            aria-expanded={expanded}
            onClick={() => setExpanded((value) => !value)}
          >
            {expanded ? "Show fewer" : `Show all ${images.length}`}
          </button>
        )}
      </div>
    );
  }
);

ProductResultGrid.displayName = "ProductResultGrid";

export default ProductResultGrid;
