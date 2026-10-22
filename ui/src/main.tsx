import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { connect } from "./host";
import "@fontsource/barlow/latin-400.css";
import "@fontsource/barlow/latin-500.css";
import "@fontsource/barlow/latin-600.css";
import "@fontsource/barlow-condensed/latin-500.css";
import "@fontsource/barlow-condensed/latin-600.css";
import "./style.css";

// The Tauri window and the browser page both provide `#ui-root` over the Bevy surface.
connect().then((host) => {
  createRoot(document.getElementById("ui-root")!).render(
    <StrictMode>
      <App host={host} />
    </StrictMode>,
  );
});
