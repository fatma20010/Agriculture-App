import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 8099,
    proxy: {
      // Ensure the most specific rules are first
      '/api/images': {
        target: 'http://localhost:5002',
        changeOrigin: true,
        secure: false,
        ws: false,
        // No rewrite: let the path pass through as-is
      },
      '/images': {
        target: 'http://localhost:5002',
        changeOrigin: true,
        secure: false,
        ws: false,
        rewrite: (path) => path.replace(/^\/images\/OutputImages\//, '/images/'),
      },
      '/grass-images': {
        target: 'http://localhost:5002',
        changeOrigin: true,
        secure: false,
        ws: false,
        rewrite: (path) => path.replace(/^\/grass-images\/OutputImages\//, '/grass-images/'),
      },
      '/grass-api': {
        target: 'http://localhost:5002',
        changeOrigin: true,
        rewrite: (path) => {
          if (path.startsWith('/grass-api/images/')) {
            return path.replace('/grass-api/images/', '/grass-images/');
          } else if (path === '/grass-api/generate_report') {
            return '/generate_grass_report';
          }
          return path.replace(/^\/grass-api/, '');
        },
        secure: false,
        ws: true,
        configure: (proxy, _options) => {
          proxy.on('error', (err, _req, _res) => {
            console.log('proxy error', err);
          });
          proxy.on('proxyReq', (proxyReq, req, _res) => {
            console.log('Sending Grass Request to the Target:', req.method, req.url);
          });
          proxy.on('proxyRes', (proxyRes, req, _res) => {
            console.log('Received Grass Response from the Target:', proxyRes.statusCode, req.url);
          });
        },
      },
      '/api': {
        target: 'http://localhost:5002',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
        secure: false,
        ws: true,
        configure: (proxy, _options) => {
          proxy.on('error', (err, _req, _res) => {
            console.log('proxy error', err);
          });
          proxy.on('proxyReq', (proxyReq, req, _res) => {
            console.log('Sending Request to the Target:', req.method, req.url);
          });
          proxy.on('proxyRes', (proxyRes, req, _res) => {
            console.log('Received Response from the Target:', proxyRes.statusCode, req.url);
          });
        },
      },
    },
  },
  plugins: [
    react(),
    mode === 'development' &&
    componentTagger(),
  ].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
