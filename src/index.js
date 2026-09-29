// Cloudflare Worker in front of the Dash app.
// The Worker sends every request to one container that runs gunicorn.
import { Container, getContainer } from "@cloudflare/containers";

export class DashApp extends Container {
  defaultPort = 8080;
  // Stop the container after 30 idle minutes. Uploaded data is lost then.
  sleepAfter = "30m";
}

export default {
  async fetch(request, env) {
    // One named instance, so every user hits the same in-memory datasets.
    return getContainer(env.DASH_APP).fetch(request);
  },
};
