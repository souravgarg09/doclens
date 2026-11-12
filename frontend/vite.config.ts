import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

// The product name lives in one place: <project>/brand.json.
type Brand = { name: string; tagline: string; logo_letter: string };
const brandFile = fileURLToPath(new URL("../brand.json", import.meta.url));
const brand: Brand = { name: "DocLens", tagline: "", logo_letter: "D", ...JSON.parse(readFileSync(brandFile, "utf-8")) };

// Replaces %APP_NAME% in index.html so the browser tab title follows brand.json too.
const brandHtml: Plugin = {
  name: "brand-html",
  transformIndexHtml: (html) => html.replaceAll("%APP_NAME%", brand.name),
};

export default defineConfig({
  plugins: [react(), brandHtml],
  define: { __BRAND__: JSON.stringify(brand) },
  server: {
    port: 5173,
    proxy: { "/api": { target: process.env.VITE_API_URL ?? "http://127.0.0.1:8000", changeOrigin: true } },
  },
});
