/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * Navigation bar component
 */

import React from "react";
import MenuIcon from "@mui/icons-material/Menu";
import { config, isFashionMode } from "../config/config";

const Navbar: React.FC = () => {
  const categories = config.ui.categories;

  const getCategoryLink = (categoryKey: keyof typeof categories): string => {
    // Remove all mode switching logic
    return "#";
  };

  const isCategoryActive = (categoryKey: keyof typeof categories): boolean => {
    // Only fashion is active
    return categoryKey === 'fashion';
  };

  return (
    <div>
      {/* Main navigation bar */}
      <div className="bg-[#FFFFFF] h-[48px] px-3 py-2 lg:px-5 text-white flex justify-between items-center">
        {/* Left side - Menu and Brand */}
        <div className="flex items-center shrink-0">
          <MenuIcon sx={{ color: "#5E5E5E" }} fontSize="small" />
          <p className="text-[22px] ml-[20px] font-bold text-[#202020]">
            Avanzare
          </p>
        </div>
        
        {/* Right side - Welcome message */}
        <div className="flex items-center gap-x-2">
          <div className="flex items-center gap-2 p-3 rounded-full">
            <p className="text-[14px] text-[#202020]">Welcome!</p>
          </div>
        </div>
      </div>

      {/* Categories bar */}
      <div className="bg-[#F2F2F2] mt-[1px] h-[57px] text-white px-3 py-2 lg:px-8 flex items-center gap-8">
        {/* Beauty and Wellness */}
        <div className="flex items-center hover:underline">
          <p className="text-[15px] text-[#666] font-medium hover:underline">
            {categories.beauty}
          </p>
        </div>

        {/* Fashion - Always Active */}
        <div className="flex items-center">
          <p className="text-[15px] font-medium text-[#000] underline">
            {categories.fashion}
          </p>
        </div>

        {/* Remove Home Goods section entirely */}
        {/* 
<div 
  className="flex items-center" 
  style={{ 
    textDecoration: isCategoryActive('homeGoods') ? 'underline' : 'none',
    pointerEvents: isCategoryActive('homeGoods') ? 'none' : 'auto'
  }}
>
  <a 
    className="text-[15px] font-medium hover:underline" 
    style={{ 
      color: isCategoryActive('homeGoods') ? "#000" : "#666" 
    }}
  >
    {categories.homeGoods}
  </a>
</div>
*/}

        {/* Grocery */}
        <div className="flex items-center hover:underline">
          <p className="text-[15px] text-[#666] font-medium hover:underline">
            {categories.grocery}
          </p>
        </div>

        {/* Office */}
        <div className="flex items-center hover:underline">
          <p className="text-[15px] text-[#666] font-medium hover:underline">
            {categories.office}
          </p>
        </div>

        {/* Lifestyle */}
        <div className="flex items-center hover:underline">
          <p className="text-[15px] text-[#666] font-medium hover:underline">
            {categories.lifestyle}
          </p>
        </div>

        {/* Last Call */}
        <div className="flex items-center hover:underline">
          <p className="text-[15px] text-[#666] font-medium hover:underline">
            {categories.lastCall}
          </p>
        </div>
      </div>
    </div>
  );
};

export default Navbar;
