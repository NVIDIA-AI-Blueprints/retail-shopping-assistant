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
  clearSelectedShopperProfileId,
  clearUserSession,
  getSelectedShopperProfileId,
  parseShopperProfiles,
  rotateUserSession,
  setSelectedShopperProfileId,
} from "./utils";

interface ChatMount {
  instance: number;
  preserveIdentityOnMount: boolean;
}

const SHOPPER_PROFILE_RETRY_DELAY_MS = 5000;
const SHOPPER_PROFILE_LOAD_ATTEMPTS = 2;

const App: React.FC = () => {
  const [selectedProduct, setSelectedProduct] = useState<ProductSummary | null>(null);
  const [products, setProducts] = useState<ProductSummary[]>([]);
  const [shopperProfiles, setShopperProfiles] = useState<ShopperProfile[]>([]);
  const [shopperProfilesStatus, setShopperProfilesStatus] =
    useState<ShopperProfilesStatus>("loading");
  const [selectedProfileId, setSelectedProfileId] = useState<
    string | null | undefined
  >(
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

  const clearShopperSelection = useCallback(() => {
    clearSelectedShopperProfileId();
    clearUserSession();
    setSelectedProfileId(undefined);
    setSelectedProduct(null);
    setProducts([]);
  }, []);

  useEffect(() => {
    const abortController = new AbortController();
    let attemptCount = 0;
    let profilesLoaded = false;
    let requestInFlight = false;
    let retryTimer: number | undefined;

    const loadShopperProfiles = async () => {
      if (
        abortController.signal.aborted ||
        profilesLoaded ||
        requestInFlight
      ) {
        return;
      }

      requestInFlight = true;
      attemptCount += 1;
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

        profilesLoaded = true;
        setShopperProfiles(profiles);
        setShopperProfilesStatus("ready");

        const storedProfileId = getSelectedShopperProfileId();
        if (
          typeof storedProfileId === "string" &&
          !profiles.some(
            (profile) => profile.shopper_profile_id === storedProfileId
          )
        ) {
          clearShopperSelection();
        }
      } catch (error) {
        if (abortController.signal.aborted) return;
        console.warn("Failed to load representative shoppers", error);
        setShopperProfiles([]);
        setShopperProfilesStatus("unavailable");
        if (attemptCount < SHOPPER_PROFILE_LOAD_ATTEMPTS) {
          retryTimer = window.setTimeout(() => {
            retryTimer = undefined;
            void loadShopperProfiles();
          }, SHOPPER_PROFILE_RETRY_DELAY_MS);
        }
      } finally {
        requestInFlight = false;
      }
    };

    const retryUnavailableProfiles = () => {
      if (
        abortController.signal.aborted ||
        profilesLoaded ||
        requestInFlight
      ) {
        return;
      }

      if (retryTimer !== undefined) {
        window.clearTimeout(retryTimer);
        retryTimer = undefined;
      }
      attemptCount = 0;
      void loadShopperProfiles();
    };

    window.addEventListener("focus", retryUnavailableProfiles);
    window.addEventListener("online", retryUnavailableProfiles);
    void loadShopperProfiles();

    return () => {
      abortController.abort();
      if (retryTimer !== undefined) {
        window.clearTimeout(retryTimer);
      }
      window.removeEventListener("focus", retryUnavailableProfiles);
      window.removeEventListener("online", retryUnavailableProfiles);
    };
  }, [clearShopperSelection]);

  const handleShopperChange = (shopperProfileId: string | null) => {
    if (isChatBusy || shopperProfileId === selectedProfileId) return;

    startShopperSession(shopperProfileId);
  };

  const shopperSessionReady =
    selectedProfileId === null ||
    (typeof selectedProfileId === "string" &&
      shopperProfilesStatus === "ready" &&
      shopperProfiles.some(
        (profile) => profile.shopper_profile_id === selectedProfileId
      ));

  return (
    <div className="shopping-app">
      <Navbar
        shopperProfiles={shopperProfiles}
        shopperProfilesStatus={shopperProfilesStatus}
        selectedShopperProfileId={selectedProfileId}
        isShopperSwitchDisabled={isChatBusy}
        onShopperChange={handleShopperChange}
      />
      {shopperSessionReady ? (
        <main className="assistant-workspace">
          <ProductDetailPanel
            selectedProduct={selectedProduct}
            products={products}
            onProductSelect={setSelectedProduct}
          />
          <Chatbox
            key={chatMount.instance}
            selectedProduct={selectedProduct}
            selectedShopperProfileId={selectedProfileId ?? null}
            onProductSelect={setSelectedProduct}
            onProductsUpdate={setProducts}
            onBusyChange={setIsChatBusy}
            preserveIdentityOnMount={chatMount.preserveIdentityOnMount}
          />
        </main>
      ) : (
        <main className="shopper-session-gate">
          <section aria-labelledby="shopper-session-heading">
            <p>New shopping session</p>
            <h1 id="shopper-session-heading">Choose how you’d like to shop</h1>
            <span>
              Select Guest mode or one of the five representative shoppers
              from the <strong>Shop as</strong> menu above.
            </span>
          </section>
        </main>
      )}
      <Footer />
      <ToastContainer position="top-right" />
    </div>
  );
};

export default App;
