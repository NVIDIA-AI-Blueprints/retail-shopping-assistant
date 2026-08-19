// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Product inspection panel for catalog results returned by the assistant.
 */

import React, { useCallback, useRef, useState } from "react";
import { ProductDetailPanelProps, ProductPrice, ProductSummary } from "../types";
import { getDefaultImage } from "../config/config";

/** Neither half of the panel may be driven to nothing by a drag. */
const MIN_DETAIL_HEIGHT = 140;
const MIN_RECENT_HEIGHT = 96;

const ProductDetailPanel: React.FC<ProductDetailPanelProps> = ({
  selectedProduct,
  products,
  onProductSelect,
}) => {
  const displayImage = selectedProduct?.productUrl || getDefaultImage();
  const catalogFacts = getCatalogFacts(selectedProduct);

  // The detail area used to take whatever height its content wanted, so a
  // product with a long description squeezed the list below it to nothing and
  // there was no longer anything to scroll. Its height is a split the shopper
  // sets instead, and neither half can be driven to zero.
  const panelRef = useRef<HTMLElement | null>(null);
  const [detailHeight, setDetailHeight] = useState<number | null>(null);

  const startResize = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    const panel = panelRef.current;
    if (!panel) return;
    event.preventDefault();
    const handle = event.currentTarget;
    const { pointerId } = event;
    handle.setPointerCapture(pointerId);

    const onMove = (move: PointerEvent) => {
      const bounds = panel.getBoundingClientRect();
      const next = move.clientY - bounds.top;
      const largest = bounds.height - MIN_RECENT_HEIGHT;
      setDetailHeight(Math.max(MIN_DETAIL_HEIGHT, Math.min(next, largest)));
    };
    const onUp = () => {
      // Releasing a pointer that has already gone throws, and a cancelled
      // gesture never sends pointerup -- both would leave the handle dragging
      // for the rest of the session.
      try {
        handle.releasePointerCapture(pointerId);
      } catch {
        /* already released */
      }
      handle.removeEventListener("pointermove", onMove);
      handle.removeEventListener("pointerup", onUp);
      handle.removeEventListener("pointercancel", onUp);
    };
    handle.addEventListener("pointermove", onMove);
    handle.addEventListener("pointerup", onUp);
    handle.addEventListener("pointercancel", onUp);
  }, []);

  return (
    <aside
      className="product-detail-panel"
      aria-label="Product detail panel"
      ref={panelRef}
    >
      <div
        className="product-detail-panel__detail"
        style={detailHeight === null ? undefined : { flex: `0 0 ${detailHeight}px` }}
      >
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
      </div>

      {products.length > 0 && (
        <div
          className="product-detail-panel__resizer"
          role="separator"
          aria-orientation="horizontal"
          aria-label="Resize product details"
          onPointerDown={startResize}
        />
      )}

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
