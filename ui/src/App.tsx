// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Main App component for the Shopping Assistant
 */

import React, { useCallback, useEffect, useState } from "react";
import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

import Navbar from "./components/Navbar";
import ProductDetailPanel from "./components/ProductDetailPanel";
import Chatbox from "./components/chatbox/chatbox";
import Footer from "./components/Footer";
import { useCart } from "./hooks/useCart";
import { ShopperProfilesStatus } from "./components/ShopperPicker";
import { config } from "./config/config";
import { ProductSummary, ShopperProfile } from "./types";
import {
  getSelectedShopperProfileId,
  parseShopperProfiles,
  rotateUserSession,
  setSelectedShopperProfileId,
} from "./utils";

interface ChatMount {
  instance: number;
  preserveIdentityOnMount: boolean;
}

const App: React.FC = () => {
  const [selectedProduct, setSelectedProduct] = useState<ProductSummary | null>(null);
  const [products, setProducts] = useState<ProductSummary[]>([]);
  const [shopperProfiles, setShopperProfiles] = useState<ShopperProfile[]>([]);
  const [shopperProfilesStatus, setShopperProfilesStatus] =
    useState<ShopperProfilesStatus>("loading");
  const [selectedProfileId, setSelectedProfileId] = useState<string | null>(
    getSelectedShopperProfileId
  );
  const [isChatBusy, setIsChatBusy] = useState(false);
  const cart = useCart();
  const { refresh: cartRefresh, reset: cartReset } = cart;
  const [chatMount, setChatMount] = useState<ChatMount>({
    instance: 0,
    preserveIdentityOnMount: false,
  });

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

  const startShopperSession = useCallback((shopperProfileId: string | null) => {
    setSelectedShopperProfileId(shopperProfileId);
    rotateUserSession();
    setSelectedProfileId(shopperProfileId);
    setSelectedProduct(null);
    setProducts([]);
    cartReset();
    setChatMount(({ instance }) => ({
      instance: instance + 1,
      preserveIdentityOnMount: true,
    }));
  }, [cartReset]);

  /**
   * Start a fresh session.
   *
   * Remounting with `preserveIdentityOnMount: false` runs Chatbox's own reset
   * with identity clearing, which is the existing path -- so a new user, cart
   * and conversation are minted on the next request. Reimplementing that here
   * would be a second definition of what "reset" means.
   */
  const resetSession = useCallback(() => {
    setSelectedProduct(null);
    setProducts([]);
    // A reset mints a new cart_id, so the old snapshot on screen would be a
    // cart the shopper no longer has.
    cartReset();
    setChatMount(({ instance }) => ({
      instance: instance + 1,
      preserveIdentityOnMount: false,
    }));
  }, [cartReset]);

  useEffect(() => {
    const abortController = new AbortController();

    const loadShopperProfiles = async () => {
      try {
        const response = await fetch(
          `${config.api.baseUrl}${config.api.endpoints.shopperProfiles}`,
          { signal: abortController.signal }
        );
        if (!response.ok) {
          throw new Error(`Shopper profiles request failed: ${response.status}`);
        }
        const profiles = parseShopperProfiles(await response.json());
        if (abortController.signal.aborted) return;

        setShopperProfiles(profiles);
        setShopperProfilesStatus("ready");

        const storedProfileId = getSelectedShopperProfileId();
        if (
          storedProfileId &&
          !profiles.some(
            (profile) => profile.shopper_profile_id === storedProfileId
          )
        ) {
          startShopperSession(null);
        }
      } catch (error) {
        if (abortController.signal.aborted) return;
        console.warn("Failed to load representative shoppers", error);
        setShopperProfiles([]);
        setShopperProfilesStatus("unavailable");

        if (getSelectedShopperProfileId() !== null) {
          startShopperSession(null);
        }
      }
    };

    loadShopperProfiles();
    return () => abortController.abort();
  }, [startShopperSession]);

  const handleShopperChange = (shopperProfileId: string | null) => {
    if (isChatBusy || shopperProfileId === selectedProfileId) return;

    startShopperSession(shopperProfileId);
  };

  return (
    <div className="shopping-app">
      <Navbar
        shopperProfiles={shopperProfiles}
        shopperProfilesStatus={shopperProfilesStatus}
        selectedShopperProfileId={selectedProfileId}
        isShopperSwitchDisabled={isChatBusy}
        onShopperChange={handleShopperChange}
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
          key={chatMount.instance}
          selectedProduct={selectedProduct}
          selectedShopperProfileId={selectedProfileId}
          onProductSelect={setSelectedProduct}
          onProductsUpdate={setProducts}
          onBusyChange={handleBusyChange}
          preserveIdentityOnMount={chatMount.preserveIdentityOnMount}
        />
      </main>
      <Footer />
      <ToastContainer position="top-right" />
    </div>
  );
};

export default App;
