// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * The cart's HTTP surface.
 *
 * Every other call in this app is a hand-rolled `fetch` at its call site. These
 * are collected here because the cart adds several and they share the identity
 * handling and the error shape.
 */

import { config } from "../config/config";
import { CartSnapshot } from "../types";

const cartUrl = (path: string, cartId: string): string =>
  `${config.api.baseUrl}${path}?cart_id=${encodeURIComponent(cartId)}`;

export class CartError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "CartError";
    this.status = status;
  }
}

const parse = async (response: Response): Promise<CartSnapshot> => {
  if (!response.ok) {
    let detail = `Cart request failed (${response.status})`;
    try {
      const body = await response.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      // A non-JSON error body is still an error; keep the status message.
    }
    throw new CartError(response.status, detail);
  }
  const body = await response.json();
  return {
    lines: Array.isArray(body?.lines) ? body.lines : [],
    subtotal: typeof body?.subtotal === "number" ? body.subtotal : null,
  };
};

export const fetchCart = async (
  cartId: string,
  signal?: AbortSignal
): Promise<CartSnapshot> =>
  parse(await fetch(cartUrl("/cart", cartId), { signal }));

/**
 * Set one line to an absolute quantity. Zero removes it.
 *
 * The caller supplies the idempotency key and mints a new one per intent.
 * Deriving it from the target quantity looks safer and is not: a replay
 * returns the stored response without mutating, so 2 -> 3 -> 2 would report
 * success and leave the cart at 3.
 */
export const setCartLineQuantity = async (
  cartId: string,
  cartLineId: string,
  quantity: number,
  idempotencyKey: string
): Promise<CartSnapshot> =>
  parse(
    await fetch(cartUrl(`/cart/lines/${encodeURIComponent(cartLineId)}`, cartId), {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quantity, idempotency_key: idempotencyKey }),
    })
  );

export const newIdempotencyKey = (): string => {
  const browserCrypto =
    typeof crypto !== "undefined" ? (crypto as Crypto) : undefined;
  if (browserCrypto?.randomUUID) return browserCrypto.randomUUID();
  return `cart-${Date.now()}-${Math.random().toString(16).slice(2)}`;
};
