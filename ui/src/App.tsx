// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Main App component for the Shopping Assistant
 */

import React, { useState } from "react";
import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

import Navbar from "./components/Navbar";
import ProductDetailPanel from "./components/ProductDetailPanel";
import Chatbox from "./components/chatbox/chatbox";
import Footer from "./components/Footer";
import { ProductSummary } from "./types";

const App: React.FC = () => {
  const [selectedProduct, setSelectedProduct] = useState<ProductSummary | null>(null);
  const [products, setProducts] = useState<ProductSummary[]>([]);

  return (
    <div className="shopping-app">
      <Navbar />
      <main className="assistant-workspace">
        <ProductDetailPanel
          selectedProduct={selectedProduct}
          products={products}
          onProductSelect={setSelectedProduct}
        />
        <Chatbox
          selectedProduct={selectedProduct}
          onProductSelect={setSelectedProduct}
          onProductsUpdate={setProducts}
        />
      </main>
      <Footer />
      <ToastContainer position="top-right" />
    </div>
  );
};

export default App;
