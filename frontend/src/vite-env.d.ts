/// <reference types="vite/client" />

/** Product branding injected at build time from <project>/brand.json (see vite.config.ts). */
declare const __BRAND__: { name: string; tagline: string; logo_letter: string };
