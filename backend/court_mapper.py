import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CourtMapper:
    def __init__(self, court_landmarks, mini_court):
        """
        court_landmarks: Dictionary of YOLO-detected padel court landmarks {class_name: [(x, y), ...]}
        mini_court: Instance of MiniCourt (has the 2D coordinate space defined)
        """
        self.mini_court = mini_court
        self.homography_matrix = None
        self._from_keypoints = False

        src_points = None
        for class_name, points in court_landmarks.items():
            if len(points) >= 4:
                corners = self._find_four_corners(points)
                src_points = self._sort_corners(corners)
                logger.info("Court corners from class '%s': %d keypoints", class_name, len(points))
                break

        if src_points is None:
            all_points = []
            for points in court_landmarks.values():
                all_points.extend(points)
            if len(all_points) >= 4:
                corners = self._find_four_corners(all_points)
                src_points = self._sort_corners(corners)
                logger.info("Court corners from all classes combined: %d keypoints", len(all_points))

        if src_points is not None:
            xs = [p[0] for p in src_points]
            ys = [p[1] for p in src_points]
            width, height = max(xs) - min(xs), max(ys) - min(ys)
            if width < 100 or height < 50:
                logger.warning("Court keypoints bounding box too small (%dx%d). Rejecting model output.", width, height)
                src_points = None

        if src_points is not None:
            dst_points = self._get_mini_court_corners(mini_court)
            logger.info("Homography src corners: %s", src_points)
            logger.info("Homography dst corners: %s", dst_points)
            self.homography_matrix, _ = cv2.findHomography(
                np.array(src_points, dtype=np.float32),
                np.array(dst_points, dtype=np.float32),
            )
            self._from_keypoints = True
        else:
            logger.warning(
                "Court keypoint detection failed — got %s. "
                "Will estimate from player positions.",
                {k: len(v) for k, v in court_landmarks.items()} if court_landmarks else "empty",
            )

    @property
    def is_valid(self):
        return self.homography_matrix is not None and self._from_keypoints

    def estimate_from_player_positions(self, player_detections):
        """
        Estimate court boundaries from the bounding box of all player foot
        positions observed across all frames. Used as a fallback when the
        keypoint model produces no usable output.
        """
        all_feet = []
        for frame_dict in player_detections:
            for bbox in frame_dict.values():
                foot_x = (bbox[0] + bbox[2]) / 2
                foot_y = bbox[3]
                all_feet.append((foot_x, foot_y))

        if len(all_feet) < 4:
            logger.warning("Too few player positions (%d) to estimate court bounds", len(all_feet))
            self.homography_matrix = np.eye(3)
            return

        xs = [p[0] for p in all_feet]
        ys = [p[1] for p in all_feet]

        margin_x = 0.08 * (max(xs) - min(xs))
        margin_y = 0.08 * (max(ys) - min(ys))

        tl = (min(xs) - margin_x, min(ys) - margin_y)
        tr = (max(xs) + margin_x, min(ys) - margin_y)
        br = (max(xs) + margin_x, max(ys) + margin_y)
        bl = (min(xs) - margin_x, max(ys) + margin_y)

        src = np.array([tl, tr, br, bl], dtype=np.float32)
        dst = np.array(self._get_mini_court_corners(self.mini_court), dtype=np.float32)

        logger.info("Estimated court bounds from %d foot positions", len(all_feet))
        logger.info("  src corners: TL=%s TR=%s BR=%s BL=%s", tl, tr, br, bl)

        self.homography_matrix, _ = cv2.findHomography(src, dst)

    def _find_four_corners(self, points):
        """Extract the 4 outermost corners from a set of keypoints."""
        if len(points) == 4:
            return list(points)

        pts = np.array(points, dtype=np.float32)
        sums = pts[:, 0] + pts[:, 1]
        diffs = pts[:, 0] - pts[:, 1]

        indices = set()
        indices.add(int(np.argmin(sums)))
        indices.add(int(np.argmax(diffs)))
        indices.add(int(np.argmax(sums)))
        indices.add(int(np.argmin(diffs)))

        corners = [points[i] for i in sorted(indices)]
        if len(corners) < 4:
            return list(points[:4])
        return corners[:4]

    def _sort_corners(self, points):
        """
        Sort 4 points into Top-Left, Top-Right, Bottom-Right, Bottom-Left
        assuming a standard broadcast camera view.
        """
        sorted_by_y = sorted(points, key=lambda p: p[1])
        top_points = sorted_by_y[:2]
        bottom_points = sorted_by_y[2:]

        tl, tr = sorted(top_points, key=lambda p: p[0])
        bl, br = sorted(bottom_points, key=lambda p: p[0])

        return [tl, tr, br, bl]

    def _get_mini_court_corners(self, mini_court):
        tl = (mini_court.court_start_x, mini_court.court_start_y)
        tr = (mini_court.court_end_x, mini_court.court_start_y)
        br = (mini_court.court_end_x, mini_court.court_end_y)
        bl = (mini_court.court_start_x, mini_court.court_end_y)
        return [tl, tr, br, bl]

    def get_mini_court_coordinates(self, object_position):
        if self.homography_matrix is None:
            return object_position

        pt = np.array([[[object_position[0], object_position[1]]]], dtype=np.float32)
        dst_pt = cv2.perspectiveTransform(pt, self.homography_matrix)

        x = dst_pt[0][0][0]
        y = dst_pt[0][0][1]

        # Clamp to the mini-court rectangle so downstream normalization
        # (which divides by court_w / court_h) always produces values in [0, 1].
        mc = self.mini_court
        x = float(np.clip(x, mc.court_start_x, mc.court_end_x))
        y = float(np.clip(y, mc.court_start_y, mc.court_end_y))

        return (int(x), int(y))

