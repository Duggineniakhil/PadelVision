/** Court geometry in metres, matching pipeline/src/padelvision/court/geometry.py.
 * Origin at the court centre, x across (-5..5), y along (-10..10), net at y = 0,
 * the camera (near) side at negative y. */
export const COURT_W = 10;
export const COURT_L = 20;
export const SERVICE = 6.95;

export type Seg = [[number, number], [number, number]];

const HW = COURT_W / 2;
const HL = COURT_L / 2;

export const COURT_LINES: Seg[] = [
  [[-HW, -HL], [HW, -HL]],
  [[-HW, HL], [HW, HL]],
  [[-HW, -HL], [-HW, HL]],
  [[HW, -HL], [HW, HL]],
  [[-HW, -SERVICE], [HW, -SERVICE]],
  [[-HW, SERVICE], [HW, SERVICE]],
  [[0, -SERVICE], [0, SERVICE]],
];
export const NET: Seg = [[-HW, 0], [HW, 0]];

export const TEAM_COLOR = { near: "#16a34a", far: "#ea580c" } as const;
export const PLAYER_TEAM: Record<number, "near" | "far"> = { 1: "near", 2: "near", 3: "far", 4: "far" };

// --- calibration constraints (pipeline/src/padelvision/court/geometry.py NAMED_LINES, COURT_KEYPOINTS)
export type CalLine =
  | "near_service_line"
  | "far_service_line"
  | "center_line"
  | "left_side"
  | "right_side"
  | "near_baseline"
  | "far_baseline"
  | "net_base";

/** Lines in the order the calibration page lists them; `mode` = how a click snaps by default. */
export const CAL_LINES: { name: CalLine; label: string; hint: string; seg: Seg; mode: "paint" | "edge" }[] = [
  { name: "near_service_line", label: "Near service line", hint: "Painted line across the near half", seg: [[-HW, -SERVICE], [HW, -SERVICE]], mode: "paint" },
  { name: "far_service_line", label: "Far service line", hint: "Painted line across the far half", seg: [[-HW, SERVICE], [HW, SERVICE]], mode: "paint" },
  { name: "center_line", label: "Centre line", hint: "Between the service lines, either side of the net", seg: [[0, -SERVICE], [0, SERVICE]], mode: "paint" },
  { name: "left_side", label: "Left side", hint: "Where the floor meets the left wall or fence", seg: [[-HW, -HL], [-HW, HL]], mode: "edge" },
  { name: "right_side", label: "Right side", hint: "Where the floor meets the right wall or fence", seg: [[HW, -HL], [HW, HL]], mode: "edge" },
  { name: "far_baseline", label: "Far baseline", hint: "Where the floor meets the back wall", seg: [[-HW, HL], [HW, HL]], mode: "edge" },
  { name: "net_base", label: "Net", hint: "Where the net meets the floor (not its top band)", seg: [[-HW, 0], [HW, 0]], mode: "edge" },
  { name: "near_baseline", label: "Near baseline", hint: "Usually out of view for a low camera", seg: [[-HW, -HL], [HW, -HL]], mode: "edge" },
];

export type CalPoint =
  | "near_left" | "near_right" | "far_left" | "far_right" | "net_left" | "net_right"
  | "near_service_left" | "near_service_center" | "near_service_right"
  | "far_service_left" | "far_service_center" | "far_service_right";

export const CAL_POINTS: { name: CalPoint; label: string; xy: [number, number] }[] = [
  { name: "far_left", label: "Far left corner", xy: [-HW, HL] },
  { name: "far_right", label: "Far right corner", xy: [HW, HL] },
  { name: "far_service_left", label: "Far service line, left end", xy: [-HW, SERVICE] },
  { name: "far_service_center", label: "Far service line, centre (T)", xy: [0, SERVICE] },
  { name: "far_service_right", label: "Far service line, right end", xy: [HW, SERVICE] },
  { name: "net_left", label: "Net, left end (floor)", xy: [-HW, 0] },
  { name: "net_right", label: "Net, right end (floor)", xy: [HW, 0] },
  { name: "near_service_left", label: "Near service line, left end", xy: [-HW, -SERVICE] },
  { name: "near_service_center", label: "Near service line, centre (T)", xy: [0, -SERVICE] },
  { name: "near_service_right", label: "Near service line, right end", xy: [HW, -SERVICE] },
  { name: "near_left", label: "Near left corner", xy: [-HW, -HL] },
  { name: "near_right", label: "Near right corner", xy: [HW, -HL] },
];
