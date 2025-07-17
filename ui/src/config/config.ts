/**
 * Configuration file for the Shopping Assistant UI
 */

export interface AppConfig {
  api: {
    baseUrl: string;
    port: number;
    endpoints: {
      query: string;
      stream: string;
      health: string;
    };
  };
  ui: {
    defaultImages: {
      fashion: string;
      furniture: string;
    };
    categories: {
      beauty: string;
      fashion: string;
      homeGoods: string;
      grocery: string;
      office: string;
      lifestyle: string;
      lastCall: string;
    };
  };
  features: {
    guardrails: {
      enabled: boolean;
      defaultState: boolean;
    };
    imageUpload: {
      enabled: boolean;
      maxSize: number; // in MB
      allowedTypes: string[];
    };
  };
}

// Get configuration based on environment
const getConfig = (): AppConfig => {
  const isDevelopment = process.env.NODE_ENV === 'development';
  const port = window.location.port;
  
  // Determine backend port based on current port
  let backendPort = 8009; // default
  if (port === '3000' || port === '3001') {
    backendPort = 8009;
  } else if (port === '13000' || port === '13001') {
    backendPort = 8009;
  }

  return {
    api: {
      baseUrl: `${window.location.protocol}//${window.location.hostname}:${backendPort}`,
      port: backendPort,
      endpoints: {
        query: '/query',
        stream: '/query/stream',
        health: '/health',
      },
    },
    ui: {
      defaultImages: {
        fashion: "/images/hans-isaacson-ehYEq99-psc-unsplash-lg.jpg",
        furniture: "https://images.unsplash.com/photo-1567016376408-0226e4d0c1ea?q=80&w=1887&auto=format&f[…]3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
      },
      categories: {
        beauty: "BEAUTY AND WELLNESS",
        fashion: "FASHION",
        homeGoods: "HOME GOODS",
        grocery: "GROCERY",
        office: "OFFICE",
        lifestyle: "LIFESTYLE",
        lastCall: "LAST CALL!"
      }
    },
    features: {
      guardrails: {
        enabled: true,
        defaultState: true,
      },
      imageUpload: {
        enabled: true,
        maxSize: 10, // 10MB
        allowedTypes: ['image/jpeg', 'image/png', 'image/gif', 'image/webp'],
      },
    },
  };
};

export const config = getConfig();

// Helper functions
export const getApiUrl = (endpoint: keyof AppConfig['api']['endpoints']): string => {
  return `${config.api.baseUrl}${config.api.endpoints[endpoint]}`;
};

export const isFashionMode = (): boolean => {
  const port = window.location.port;
  return port === '3000' || port === '3001';
};

export const getDefaultImage = (): string => {
  return isFashionMode() ? config.ui.defaultImages.fashion : config.ui.defaultImages.furniture;
}; 