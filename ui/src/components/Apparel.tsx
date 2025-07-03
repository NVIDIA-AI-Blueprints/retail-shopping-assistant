/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Apparel/Product display component
 */

import React from "react";
import { ApparelProps } from "../types";
import { getDefaultImage } from "../config/config";

const Apparel: React.FC<ApparelProps> = ({ newRenderImage }) => {
  const displayImage = newRenderImage || getDefaultImage();

  return (
    <div
      style={{ width: "40vw" }}
      className="flex overflow-hidden items-center justify-center h-[85vh] flex-grow-1 object-contain"
    >
      <img
        src={displayImage}
        alt="Product display"
        className="product-image"
      />
    </div>
  );
};

export default Apparel;
