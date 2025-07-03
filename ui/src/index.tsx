/**
 * Main entry point for the Shopping Assistant UI
 */

import React from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import "./chatbox.css";
import App from "./App";

const root = ReactDOM.createRoot(document.getElementById("root") as HTMLElement);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
