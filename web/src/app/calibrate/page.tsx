import { CalibratePage } from "./view";

// Works on images uploaded in the browser: nothing to prerender.
export const instant = false;

export default function Page() {
  return <CalibratePage />;
}
