import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { OverlayWindow } from './components/OverlayWindow';
import './styles/app.css';
import './styles/overlay.css';

const isOverlay = new URLSearchParams(window.location.search).has('overlay');

createRoot(document.getElementById('root') as HTMLElement).render(
  <StrictMode>
    {isOverlay ? <OverlayWindow /> : <App />}
  </StrictMode>,
);
