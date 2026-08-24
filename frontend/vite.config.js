import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // in dev, `npm run dev` runs on :5173 and the API on :8000 (see
    // api/main.py) -- proxy /api so the app can always fetch relative
    // paths, whether it's running under Vite's dev server or served by
    // FastAPI itself from the built dist/.
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
