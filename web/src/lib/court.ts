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
