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
  const [chatMount, setChatMount] = useState<ChatMount>({
    instance: 0,
    preserveIdentityOnMount: false,
  });

  const startShopperSession = useCallback((shopperProfileId: string | null) => {
    setSelectedShopperProfileId(shopperProfileId);
    rotateUserSession();
    setSelectedProfileId(shopperProfileId);
    setSelectedProduct(null);
    setProducts([]);
    setChatMount(({ instance }) => ({
      instance: instance + 1,
      preserveIdentityOnMount: true,
    }));
  }, []);

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
          onBusyChange={setIsChatBusy}
          preserveIdentityOnMount={chatMount.preserveIdentityOnMount}
        />
      </main>
      <Footer />
      <ToastContainer position="top-right" />
    </div>
  );
};

export default App;
