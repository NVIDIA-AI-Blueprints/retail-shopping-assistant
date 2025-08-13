// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Loading spinner component
 */

import React from "react";
import styled from "@emotion/styled";
import { keyframes } from "@emotion/react";

const SpinAnimation = keyframes`
  100% { transform: rotate(360deg); }
`;

const Spinner = styled.div`
  animation: ${SpinAnimation} 1.2s linear infinite;
`;

const Loader: React.FC = () => {
  return (
    <Spinner>
      <svg 
        xmlns="http://www.w3.org/2000/svg" 
        width="25" 
        height="25" 
        viewBox="0 0 106 106" 
        fill="none"
        aria-label="Loading spinner"
      >
        <path 
          d="M3.46289 53C3.46289 80.6142 25.6414 103 52.9999 103C80.3585 103 102.537 80.6142 102.537 53C102.537 25.3858 80.3585 3 52.9999 3" 
          stroke="#59A700" 
          strokeWidth="6" 
        />
      </svg>
    </Spinner>
  );
};

export default Loader;