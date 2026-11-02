// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Cart state for the panel.
 *
 * Lives beside the other App-level state and is threaded down as props, which
 * is how everything else in this app is wired. No context is introduced for
 * one consumer.
 */

import { useCallback, useRef, useState } from "react";
import { toast } from "react-toastify";

import { CartError, fetchCart, newIdempotencyKey, setCartLineQuantity } from "../api/cart";
import { CartSnapshot } from "../types";
import { getOrCreateUserSession } from "../utils";

export interface UseCart {
  cart: CartSnapshot | null;
  isLoading: boolean;
  error: string | null;
  pendingLineId: string | null;
  refresh: () => void;
  setQuantity: (cartLineId: string, quantity: number) => void;
  reset: () => void;
}

export const useCart = (): UseCart => {
  const [cart, setCart] = useState<CartSnapshot | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pendingLineId, setPendingLineId] = useState<string | null>(null);
  const inFlight = useRef<AbortController | null>(null);

  const refresh = useCallback(() => {
    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;
    setIsLoading(true);
    setError(null);
    fetchCart(getOrCreateUserSession().cartId, controller.signal)
      .then((snapshot) => {
        if (controller.signal.aborted) return;
        setCart(snapshot);
      })
      .catch((err) => {
        if (controller.signal.aborted || err?.name === "AbortError") return;
        setError(err instanceof Error ? err.message : "Could not read the cart");
      })
      .finally(() => {
        if (!controller.signal.aborted) setIsLoading(false);
      });
  }, []);

  const setQuantity = useCallback(
    (cartLineId: string, quantity: number) => {
      if (quantity < 0) return;
      // One key per intent. The control is disabled while this is in flight,
      // so a double-click cannot mint a second key for the same intent, and a
      // later change to the same quantity correctly gets a new one -- reusing
      // a key would replay the old response and leave the cart unchanged.
      const key = newIdempotencyKey();
      setPendingLineId(cartLineId);
      setError(null);
      setCartLineQuantity(
        getOrCreateUserSession().cartId,
        cartLineId,
        quantity,
        key
      )
        .then((snapshot) => setCart(snapshot))
        .catch((err) => {
          const message =
            err instanceof Error ? err.message : "Could not update the cart";
          setError(message);
          toast.error(message);
          if (err instanceof CartError && err.status === 404) {
            // The line is gone -- someone else's change, most likely the
            // assistant's. Re-read rather than leave a phantom row on screen.
            refresh();
          }
        })
        .finally(() => setPendingLineId(null));
    },
    [refresh]
  );

  const reset = useCallback(() => {
    inFlight.current?.abort();
    setCart(null);
    setError(null);
    setPendingLineId(null);
  }, []);

  return { cart, isLoading, error, pendingLineId, refresh, setQuantity, reset };
};
