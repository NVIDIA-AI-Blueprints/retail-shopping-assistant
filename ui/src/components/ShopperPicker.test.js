// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import { createRoot } from "react-dom/client";

import ShopperPicker from "./ShopperPicker";

global.IS_REACT_ACT_ENVIRONMENT = true;

const profiles = [
  {
    shopper_profile_id: "shopper_morgan",
    display_name: "Morgan",
    shopper_type: "skeptical_researcher",
    behavior:
      "Probes for material, care burden, and repeated-wear practicality before choosing.",
    zipcode: "60601",
  },
  {
    shopper_profile_id: "shopper_alex",
    display_name: "Alex",
    shopper_type: "occasion_driven_explorer",
    behavior: "Starts from an occasion and asks for a complete look.",
    zipcode: "98101",
  },
  {
    shopper_profile_id: "shopper_casey",
    display_name: "Casey",
    shopper_type: "strict_budget_style_mixer",
    behavior: "Balances a complete look with an explicit budget.",
    zipcode: "85004",
  },
  {
    shopper_profile_id: "shopper_jordan",
    display_name: "Jordan",
    shopper_type: "impatient_decisive",
    behavior: "Uses concise references and changes cart decisions.",
    zipcode: "10001",
  },
  {
    shopper_profile_id: "shopper_riley",
    display_name: "Riley",
    shopper_type: "iterative_refiner",
    behavior: "Refines one part of an existing outfit direction.",
    zipcode: "33130",
  },
];

const selectValue = (select, value) => {
  React.act(() => {
    select.value = value;
    select.dispatchEvent(new Event("change", { bubbles: true }));
  });
};

describe("ShopperPicker", () => {
  let container;
  let root;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(() => {
    React.act(() => {
      root.unmount();
    });
    container.remove();
  });

  test("requires an explicit Guest or representative-shopper selection", () => {
    const onChange = jest.fn();
    // This test uses React createRoot directly, so rendering must be inside act.
    // eslint-disable-next-line testing-library/no-unnecessary-act
    React.act(() => {
      root.render(
        <ShopperPicker
          profiles={profiles}
          profilesStatus="ready"
          selectedShopperProfileId={undefined}
          disabled={false}
          onChange={onChange}
        />
      );
    });

    const select = container.querySelector('[aria-label="Shopper profile"]');
    expect(select.tagName).toBe("SELECT");
    expect(select.id).toBe("shopper-profile");
    expect(select.name).toBe("shopper-profile");
    expect(select.value).toBe("");
    expect(Array.from(select.options).map((option) => option.text)).toEqual([
      "Choose shopper",
      "Guest mode",
      "Morgan",
      "Alex",
      "Casey",
      "Jordan",
      "Riley",
    ]);

    selectValue(select, "__guest__");
    expect(onChange).toHaveBeenLastCalledWith(null);

    selectValue(select, "shopper_morgan");
    expect(onChange).toHaveBeenLastCalledWith("shopper_morgan");
  });

  test("keeps Guest available while profiles load or are unavailable", () => {
    const onChange = jest.fn();
    // eslint-disable-next-line testing-library/no-unnecessary-act
    React.act(() => {
      root.render(
        <ShopperPicker
          profiles={[]}
          profilesStatus="loading"
          selectedShopperProfileId={undefined}
          disabled={false}
          onChange={onChange}
        />
      );
    });

    let select = container.querySelector('[aria-label="Shopper profile"]');
    expect(select.disabled).toBe(false);
    expect(select.textContent).toContain("Guest mode");
    expect(select.textContent).toContain("Loading profiles…");
    selectValue(select, "__guest__");
    expect(onChange).toHaveBeenCalledWith(null);

    // eslint-disable-next-line testing-library/no-unnecessary-act
    React.act(() => {
      root.render(
        <ShopperPicker
          profiles={[]}
          profilesStatus="unavailable"
          selectedShopperProfileId={undefined}
          disabled={false}
          onChange={onChange}
        />
      );
    });

    select = container.querySelector('[aria-label="Shopper profile"]');
    expect(select.disabled).toBe(false);
    expect(select.textContent).toContain("Profiles unavailable");
    selectValue(select, "__guest__");
    expect(onChange).toHaveBeenLastCalledWith(null);

    // eslint-disable-next-line testing-library/no-unnecessary-act
    React.act(() => {
      root.render(
        <ShopperPicker
          profiles={profiles}
          profilesStatus="ready"
          selectedShopperProfileId={null}
          disabled
          onChange={onChange}
        />
      );
    });
    expect(
      container.querySelector('[aria-label="Shopper profile"]').disabled
    ).toBe(true);
  });
});
