import { VideoPage } from "./view";

// Rendered in the browser from the local API's data: nothing to prerender, and navigation may wait
// for that data.
export const instant = false;

export default function Page() {
  return <VideoPage />;
}
