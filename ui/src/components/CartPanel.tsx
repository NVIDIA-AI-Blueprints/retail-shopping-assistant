// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * The cart, as a shopper expects to find it on a shop.
 *
 * Structure follows ShopperPicker, the only overlay precedent in this app:
 * a trigger-anchored dialog, dismissed by outside click or Escape, with focus
 * moved in on open and returned to the trigger on close.
 */

import React, { useCallback, useEffect, useRef, useState } from "react";
import CloseIcon from "@mui/icons-material/Close";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import ShoppingCartIcon from "@mui/icons-material/ShoppingCart";

import { CartLine, CartSnapshot } from "../types";

interface CartPanelProps {
  cart: CartSnapshot | null;
  isLoading: boolean;
  error: string | null;
  /** True while the assistant is mid-turn; it may be changing the cart itself. */
  isAgentBusy: boolean;
  pendingLineId: string | null;
  onOpen: () => void;
  onSetQuantity: (cartLineId: string, quantity: number) => void;
}

const CartPanel: React.FC<CartPanelProps> = ({
  cart,
  isLoading,
  error,
  isAgentBusy,
  pendingLineId,
  onOpen,
  onSetQuantity,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const dialogRef = useRef<HTMLDivElement | null>(null);

  const lines = cart?.lines ?? [];
  const itemCount = lines.reduce((total, line) => total + line.quantity, 0);

  const close = useCallback((restoreFocus: boolean) => {
    setIsOpen(false);
    if (restoreFocus) triggerRef.current?.focus();
  }, []);

  useEffect(() => {
    if (!isOpen) return;

    const onPointerDown = (event: Event) => {
      const target = event.target as Node;
      if (dialogRef.current?.contains(target)) return;
      if (triggerRef.current?.contains(target)) return;
      close(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") close(true);
    };

    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("touchstart", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("touchstart", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [isOpen, close]);

  useEffect(() => {
    if (isOpen) dialogRef.current?.focus();
  }, [isOpen]);

  return (
    <div className="cart-panel">
      <button
        ref={triggerRef}
        type="button"
        className="cart-panel__trigger"
        aria-haspopup="dialog"
        aria-expanded={isOpen}
        aria-label={`Cart, ${itemCount} ${itemCount === 1 ? "item" : "items"}`}
        onClick={() => {
          const next = !isOpen;
          setIsOpen(next);
          if (next) onOpen();
        }}
      >
        <ShoppingCartIcon fontSize="small" />
        {itemCount > 0 && <span className="cart-panel__count">{itemCount}</span>}
      </button>

      {isOpen && (
        <div
          ref={dialogRef}
          className="cart-panel__dialog"
          role="dialog"
          aria-label="Your cart"
          tabIndex={-1}
        >
          <div className="cart-panel__header">
            <div>
              <div className="cart-panel__eyebrow">Your cart</div>
              <h2>
                {itemCount} {itemCount === 1 ? "item" : "items"}
              </h2>
            </div>
            <button
              type="button"
              className="cart-panel__close"
              onClick={() => close(true)}
              aria-label="Close cart"
            >
              <CloseIcon fontSize="small" />
            </button>
          </div>

          <div className="cart-panel__body">
            {error && <p className="cart-panel__error">{error}</p>}
            {!error && isLoading && lines.length === 0 && (
              <p className="cart-panel__empty">Loading…</p>
            )}
            {!error && !isLoading && lines.length === 0 && (
              <p className="cart-panel__empty">
                Nothing here yet. Ask for something and add it.
              </p>
            )}

            {lines.map((line) => (
              <CartLineRow
                key={line.cart_line_id}
                line={line}
                // The assistant may be mutating this same line mid-turn.
                disabled={isAgentBusy || pendingLineId === line.cart_line_id}
                onSetQuantity={onSetQuantity}
              />
            ))}
          </div>

          {lines.length > 0 && (
            <div className="cart-panel__footer">
              <span>Subtotal</span>
              <strong>
                {cart?.subtotal != null ? formatPrice(cart.subtotal) : "—"}
              </strong>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

interface CartLineRowProps {
  line: CartLine;
  disabled: boolean;
  onSetQuantity: (cartLineId: string, quantity: number) => void;
}

const CartLineRow: React.FC<CartLineRowProps> = ({
  line,
  disabled,
  onSetQuantity,
}) => (
  <div className="cart-line">
    <div className="cart-line__detail">
      <span className="cart-line__name">{line.display_name}</span>
      <span className="cart-line__meta">
        {/* Absent for one-size goods, so presence is the test. Without it a
            cart holding two sizes of one dress is unreadable. */}
        {line.size ? `Size ${line.size}` : "One size"}
        {line.unit_price != null && ` · ${formatPrice(line.unit_price)}`}
      </span>
    </div>

    <div className="cart-line__quantity">
      <button
        type="button"
        className="cart-line__step"
        disabled={disabled}
        aria-label={`Decrease quantity of ${line.display_name}`}
        onClick={() => onSetQuantity(line.cart_line_id, line.quantity - 1)}
      >
        −
      </button>
      <span className="cart-line__count" aria-live="polite">
        {line.quantity}
      </span>
      <button
        type="button"
        className="cart-line__step"
        disabled={disabled}
        aria-label={`Increase quantity of ${line.display_name}`}
        onClick={() => onSetQuantity(line.cart_line_id, line.quantity + 1)}
      >
        +
      </button>
    </div>

    <button
      type="button"
      className="cart-line__remove"
      disabled={disabled}
      aria-label={`Remove ${line.display_name}`}
      onClick={() => onSetQuantity(line.cart_line_id, 0)}
    >
      <DeleteOutlineIcon fontSize="small" />
    </button>
  </div>
);

const formatPrice = (amount: number): string => {
  try {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
    }).format(amount);
  } catch {
    return `$${amount.toFixed(2)}`;
  }
};

export default CartPanel;
