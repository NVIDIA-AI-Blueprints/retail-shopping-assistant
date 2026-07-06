// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Product inspection panel for catalog results returned by the assistant.
 */

import React from "react";
import { ProductDetailPanelProps, ProductPrice, ProductSummary } from "../types";
import { getDefaultImage } from "../config/config";

const ProductDetailPanel: React.FC<ProductDetailPanelProps> = ({
  selectedProduct,
  products,
  onProductSelect,
}) => {
  const displayImage = selectedProduct?.productUrl || getDefaultImage();
  const catalogFacts = getCatalogFacts(selectedProduct);

  return (
    <aside className="product-detail-panel" aria-label="Product detail panel">
      <div className="product-detail-panel__media">
        <img src={displayImage} alt={selectedProduct?.productName || "Product preview"} />
      </div>

      <div className="product-detail-panel__body">
        {selectedProduct ? (
          <>
            <div className="product-detail-panel__eyebrow">
              {selectedProduct.category || "Catalog item"}
            </div>
            <h2>{selectedProduct.productName}</h2>

            <div className="product-detail-panel__meta">
              {selectedProduct.price && (
                <span>{formatPrice(selectedProduct.price)}</span>
              )}
              {selectedProduct.availability && (
                <span>{formatAvailability(selectedProduct.availability)}</span>
              )}
              {selectedProduct.brand && <span>{selectedProduct.brand}</span>}
            </div>

            {selectedProduct.description && (
              <p className="product-detail-panel__description">
                {selectedProduct.description}
              </p>
            )}

            {catalogFacts.length > 0 && (
              <div className="product-detail-panel__facts">
                {catalogFacts.map((fact) => (
                  <p key={fact}>{fact}</p>
                ))}
              </div>
            )}
          </>
        ) : (
          <>
            <div className="product-detail-panel__eyebrow">Catalog item</div>
            <h2>Product details</h2>
            <p className="product-detail-panel__description">
              Returned catalog items appear here with available product facts.
            </p>
          </>
        )}
      </div>

      {products.length > 0 && (
        <div className="product-detail-panel__recent" aria-label="Recent catalog results">
          <div className="product-detail-panel__recent-title">Recent results</div>
          <div className="product-detail-panel__recent-list">
            {products.map((product) => (
              <button
                key={product.productId || product.productName}
                type="button"
                className={`product-detail-panel__recent-item${
                  product.productName === selectedProduct?.productName ? " is-selected" : ""
                }`}
                onClick={() => onProductSelect(product)}
              >
                {product.productUrl && (
                  <img src={product.productUrl} alt="" aria-hidden="true" />
                )}
                <span>{product.productName}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </aside>
  );
};

const formatPrice = (price: ProductPrice): string => {
  const currency = price.currency || "USD";
  try {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency,
    }).format(price.amount);
  } catch {
    return `$${price.amount.toFixed(2)}`;
  }
};

const formatAvailability = (availability: string): string => {
  return availability
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
};

const getCatalogFacts = (product: ProductSummary | null): string[] => {
  const text = product?.attributes?.catalog_text;
  if (typeof text !== "string") return [];

  return text
    .split(/\n+/)
    .map((line) => line.trim())
    .filter((line) => line && line !== product?.productName)
    .slice(0, 5);
};

export default ProductDetailPanel;
