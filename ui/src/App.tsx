// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Main App component for the Shopping Assistant
 */

import React, { useCallback, useState } from "react";
import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

import Navbar from "./components/Navbar";
import ProductDetailPanel from "./components/ProductDetailPanel";
import Chatbox from "./components/chatbox/chatbox";
import Footer from "./components/Footer";
import { useCart } from "./hooks/useCart";
import { ProductSummary } from "./types";

const App: React.FC = () => {
  const [selectedProduct, setSelectedProduct] = useState<ProductSummary | null>(null);
  const [products, setProducts] = useState<ProductSummary[]>([]);
  const [isChatBusy, setIsChatBusy] = useState(false);
  const cart = useCart();
  const { refresh: cartRefresh, reset: cartReset } = cart;
  const [chatInstance, setChatInstance] = useState(0);

  /**
   * Must be stable. Chatbox holds this in an effect's dependency list, so an
   * inline arrow re-runs that effect on every render -- and since this one
   * re-reads the cart, that is an unbounded fetch loop.
   */
  const handleBusyChange = useCallback(
    (busy: boolean) => {
      setIsChatBusy(busy);
      // A turn that just ended may have added or removed a line.
      if (!busy) cartRefresh();
    },
    [cartRefresh]
  );

  /**
   * Start a fresh session.
   *
   * Remounting runs Chatbox's own reset with identity clearing, which is the
   * existing path -- so a new user, cart and conversation are minted on the
   * next request. Reimplementing that here would be a second definition of
   * what "reset" means.
   */
  const resetSession = useCallback(() => {
    setSelectedProduct(null);
    setProducts([]);
    // A reset mints a new cart_id, so the old snapshot on screen would be a
    // cart the shopper no longer has.
    cartReset();
    setChatInstance((instance) => instance + 1);
  }, [cartReset]);

  return (
    <div className="shopping-app">
      <Navbar
        isResetDisabled={isChatBusy}
        onReset={resetSession}
        cart={{
          cart: cart.cart,
          isLoading: cart.isLoading,
          error: cart.error,
          isAgentBusy: isChatBusy,
          pendingLineId: cart.pendingLineId,
          onOpen: cart.refresh,
          onSetQuantity: cart.setQuantity,
        }}
      />
      <main className="assistant-workspace">
        <ProductDetailPanel
          selectedProduct={selectedProduct}
          products={products}
          onProductSelect={setSelectedProduct}
        />
        <Chatbox
          key={chatInstance}
          selectedProduct={selectedProduct}
          onProductSelect={setSelectedProduct}
          onProductsUpdate={setProducts}
          onBusyChange={handleBusyChange}
        />
      </main>
      <Footer />
      <ToastContainer position="top-right" />
    </div>
  );
};

export default App;
