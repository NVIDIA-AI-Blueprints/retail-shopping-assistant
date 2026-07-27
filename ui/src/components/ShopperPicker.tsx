/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import React from "react";

import { ShopperProfile } from "../types";

export type ShopperProfilesStatus = "loading" | "ready" | "unavailable";

interface ShopperPickerProps {
  profiles: ShopperProfile[];
  profilesStatus: ShopperProfilesStatus;
  selectedShopperProfileId: string | null | undefined;
  disabled: boolean;
  onChange: (shopperProfileId: string | null) => void;
}

const GUEST_OPTION_VALUE = "__guest__";

const ShopperPicker: React.FC<ShopperPickerProps> = ({
  profiles,
  profilesStatus,
  selectedShopperProfileId,
  disabled,
  onChange,
}) => {
  const selectedProfileIsLoading =
    typeof selectedShopperProfileId === "string" &&
    !profiles.some(
      (profile) =>
        profile.shopper_profile_id === selectedShopperProfileId
    ) &&
    profilesStatus === "loading";
  const selectedValue =
    selectedShopperProfileId === undefined
      ? ""
      : selectedShopperProfileId ?? GUEST_OPTION_VALUE;

  const handleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const value = event.target.value;
    if (!value) return;
    onChange(value === GUEST_OPTION_VALUE ? null : value);
  };

  return (
    <label className="shopper-picker">
      <span className="shopper-picker__label">Shop as</span>
      <select
        className="shopper-picker__select"
        aria-label="Shopper profile"
        value={selectedValue}
        disabled={disabled}
        onChange={handleChange}
        title={
          disabled
            ? "Wait for the current response to finish."
            : "Choose Guest mode or a representative shopper."
        }
      >
        <option value="" disabled>
          Choose shopper
        </option>
        <option value={GUEST_OPTION_VALUE}>Guest mode</option>
        {selectedProfileIsLoading && (
          <option value={selectedShopperProfileId}>
            Loading selected shopper…
          </option>
        )}
        {profiles.map((profile) => (
          <option
            key={profile.shopper_profile_id}
            value={profile.shopper_profile_id}
          >
            {profile.display_name}
          </option>
        ))}
        {profilesStatus === "loading" && !selectedProfileIsLoading && (
          <option disabled>Loading profiles…</option>
        )}
        {profilesStatus === "unavailable" && (
          <option disabled>Profiles unavailable</option>
        )}
      </select>
    </label>
  );
};

export default ShopperPicker;
