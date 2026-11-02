// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Main App component for the Shopping Assistant
 */

import React, { useState } from "react";
import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

import Navbar from "./components/Navbar";
import Apparel from "./components/Apparel";
import Chatbox from "./components/chatbox/chatbox";
import Footer from "./components/Footer";

const App: React.FC = () => {
  const [newRenderImage, setNewRenderImage] = useState<string>("");

  return (
    <div className="bg-[#FFFFFF] flex flex-col h-screen w-screen">
      <Navbar />
      <Apparel newRenderImage={newRenderImage} />
      <Chatbox setNewRenderImage={setNewRenderImage} />
      <Footer />
      <ToastContainer position="top-right" />
    </div>
  );
};

export default App;
