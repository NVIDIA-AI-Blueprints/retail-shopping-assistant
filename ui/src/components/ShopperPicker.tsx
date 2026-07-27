/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { ShopperProfile } from "../types";

export type ShopperProfilesStatus = "loading" | "ready" | "unavailable";

interface ShopperPickerProps {
  profiles: ShopperProfile[];
  profilesStatus: ShopperProfilesStatus;
  selectedShopperProfileId: string | null;
  disabled: boolean;
  onChange: (shopperProfileId: string | null) => void;
}

const PICKER_DIALOG_ID = "shopper-picker-dialog";

const ShopperPicker: React.FC<ShopperPickerProps> = ({
  profiles,
  profilesStatus,
  selectedShopperProfileId,
  disabled,
  onChange,
}) => {
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const dialogRef = useRef<HTMLElement>(null);
  const [isOpen, setIsOpen] = useState(false);
  const [previewedProfileId, setPreviewedProfileId] = useState<string | null>(
    selectedShopperProfileId
  );

  const selectedProfile = useMemo(
    () =>
      profiles.find(
        (profile) => profile.shopper_profile_id === selectedShopperProfileId
      ) ?? null,
    [profiles, selectedShopperProfileId]
  );
  const previewedProfile = useMemo(
    () =>
      profiles.find(
        (profile) => profile.shopper_profile_id === previewedProfileId
      ) ?? null,
    [profiles, previewedProfileId]
  );
  const isTriggerDisabled = disabled || profilesStatus === "loading";

  const closePicker = useCallback((restoreTriggerFocus: boolean) => {
    setIsOpen(false);
    if (restoreTriggerFocus) {
      triggerRef.current?.focus();
    }
  }, []);

  useEffect(() => {
    if (!isOpen) {
      setPreviewedProfileId(selectedShopperProfileId);
    }
  }, [isOpen, selectedShopperProfileId]);

  useEffect(() => {
    if (!isOpen) return;

    const previewedOption =
      dialogRef.current?.querySelector<HTMLButtonElement>(
        ".shopper-picker__option.is-previewed"
      );
    const firstDialogControl =
      dialogRef.current?.querySelector<HTMLButtonElement>("button");
    (previewedOption ?? firstDialogControl)?.focus();
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) return;

    const handlePointerDown = (event: MouseEvent | TouchEvent) => {
      if (
        rootRef.current &&
        event.target instanceof Node &&
        !rootRef.current.contains(event.target)
      ) {
        closePicker(false);
      }
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closePicker(true);
      }
    };

    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("touchstart", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("touchstart", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [closePicker, isOpen]);

  const togglePicker = () => {
    if (isTriggerDisabled) return;
    if (isOpen) {
      closePicker(false);
      return;
    }
    setPreviewedProfileId(selectedShopperProfileId);
    setIsOpen(true);
  };

  const selectPreviewedShopper = () => {
    if (isTriggerDisabled) return;
    if (previewedProfileId !== selectedShopperProfileId) {
      onChange(previewedProfileId);
    }
    closePicker(true);
  };

  const selectedName =
    profilesStatus === "loading" && selectedShopperProfileId
      ? "Loading…"
      : selectedProfile?.display_name ?? "Guest";
  const previewIsSelected =
    previewedProfileId === selectedShopperProfileId;

  return (
    <div className="shopper-picker" ref={rootRef}>
      <button
        ref={triggerRef}
        type="button"
        className="shopper-picker__trigger"
        aria-controls={PICKER_DIALOG_ID}
        aria-expanded={isOpen}
        aria-haspopup="dialog"
        disabled={isTriggerDisabled}
        onClick={togglePicker}
        title={
          disabled
            ? "Wait for the current response to finish."
            : profilesStatus === "loading"
              ? "Representative shoppers are loading."
              : undefined
        }
      >
        <span className="shopper-picker__trigger-label">Shop as</span>
        <span className="shopper-picker__trigger-name">{selectedName}</span>
        <span aria-hidden="true" className="shopper-picker__chevron">
          ▾
        </span>
      </button>

      {isOpen && (
        <section
          ref={dialogRef}
          id={PICKER_DIALOG_ID}
          className="shopper-picker__dialog"
          role="dialog"
          aria-label="Choose a representative shopper"
        >
          <div className="shopper-picker__dialog-header">
            <div>
              <p className="shopper-picker__eyebrow">Representative shoppers</p>
              <h2>Choose who is shopping</h2>
            </div>
            <button
              type="button"
              className="shopper-picker__close"
              aria-label="Close shopper picker"
              onClick={() => closePicker(true)}
            >
              ×
            </button>
          </div>

          <div className="shopper-picker__body">
            <div
              className="shopper-picker__options"
              aria-label="Available shoppers"
            >
              <button
                type="button"
                className={`shopper-picker__option ${
                  previewedProfileId === null ? "is-previewed" : ""
                }`}
                aria-pressed={selectedShopperProfileId === null}
                onClick={() => setPreviewedProfileId(null)}
                onFocus={() => setPreviewedProfileId(null)}
                onMouseEnter={() => setPreviewedProfileId(null)}
              >
                <span>Guest</span>
                <small>No saved shopper details</small>
              </button>

              {profiles.map((profile) => (
                <button
                  key={profile.shopper_profile_id}
                  type="button"
                  className={`shopper-picker__option ${
                    previewedProfileId === profile.shopper_profile_id
                      ? "is-previewed"
                      : ""
                  }`}
                  aria-pressed={
                    selectedShopperProfileId === profile.shopper_profile_id
                  }
                  onClick={() =>
                    setPreviewedProfileId(profile.shopper_profile_id)
                  }
                  onFocus={() =>
                    setPreviewedProfileId(profile.shopper_profile_id)
                  }
                  onMouseEnter={() =>
                    setPreviewedProfileId(profile.shopper_profile_id)
                  }
                >
                  <span>{profile.display_name}</span>
                  <small>{profile.shopper_type}</small>
                </button>
              ))}

              {profilesStatus === "loading" && (
                <p className="shopper-picker__status" role="status">
                  Loading shoppers…
                </p>
              )}
              {profilesStatus === "unavailable" && (
                <p className="shopper-picker__status" role="status">
                  Shopper profiles are unavailable. Guest remains available.
                </p>
              )}
            </div>

            <div
              className="shopper-picker__details"
              aria-live="polite"
              aria-atomic="true"
            >
              {previewedProfile ? (
                <>
                  <p className="shopper-picker__eyebrow">Shopper details</p>
                  <h3>{previewedProfile.display_name}</h3>
                  <dl>
                    <div>
                      <dt>Shopper type</dt>
                      <dd>
                        <code>{previewedProfile.shopper_type}</code>
                      </dd>
                    </div>
                    <div>
                      <dt>Behavior</dt>
                      <dd>{previewedProfile.behavior}</dd>
                    </div>
                    <div>
                      <dt>ZIP code</dt>
                      <dd>{previewedProfile.zipcode}</dd>
                    </div>
                  </dl>
                </>
              ) : (
                <>
                  <p className="shopper-picker__eyebrow">Shopper details</p>
                  <h3>Guest</h3>
                  <p className="shopper-picker__guest-copy">
                    Continue without a representative shopper profile.
                  </p>
                </>
              )}

              <button
                type="button"
                className="shopper-picker__confirm"
                disabled={isTriggerDisabled}
                onClick={selectPreviewedShopper}
              >
                {previewIsSelected
                  ? "Done"
                  : previewedProfile
                    ? `Shop as ${previewedProfile.display_name}`
                    : "Continue as Guest"}
              </button>
            </div>
          </div>
        </section>
      )}
    </div>
  );
};

export default ShopperPicker;
