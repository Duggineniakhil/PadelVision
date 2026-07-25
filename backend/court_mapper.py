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
        self.homography_matrix = None

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

        if src_points is None:
            logger.warning(
                "Court keypoint detection failed — got %s. Using identity fallback.",
                {k: len(v) for k, v in court_landmarks.items()} if court_landmarks else "empty",
            )
            self.homography_matrix = np.eye(3)
            return

        dst_points = self._get_mini_court_corners(mini_court)

        logger.info("Homography src corners: %s", src_points)
        logger.info("Homography dst corners: %s", dst_points)

        self.homography_matrix, _ = cv2.findHomography(
            np.array(src_points, dtype=np.float32),
            np.array(dst_points, dtype=np.float32)
        )

    def _find_four_corners(self, points):
        """Extract the 4 outermost corners from a set of keypoints."""
        if len(points) == 4:
            return list(points)

        pts = np.array(points, dtype=np.float32)
        sums = pts[:, 0] + pts[:, 1]
        diffs = pts[:, 0] - pts[:, 1]

        indices = set()
        indices.add(int(np.argmin(sums)))   # TL: min(x+y)
        indices.add(int(np.argmax(diffs)))  # TR: max(x-y)
        indices.add(int(np.argmax(sums)))   # BR: max(x+y)
        indices.add(int(np.argmin(diffs)))  # BL: min(x-y)

        corners = [points[i] for i in sorted(indices)]
        if len(corners) < 4:
            return list(points[:4])
        return corners[:4]

    def _sort_corners(self, points):
        """
        Sort 4 points into Top-Left, Top-Right, Bottom-Right, Bottom-Left
        assuming a standard broadcast camera view.
        """
        # Sort by y-coordinate
        sorted_by_y = sorted(points, key=lambda p: p[1])
        top_points = sorted_by_y[:2]    # furthest from camera
        bottom_points = sorted_by_y[2:] # closest to camera
        
        # Sort top points by x-coordinate
        tl, tr = sorted(top_points, key=lambda p: p[0])
        # Sort bottom points by x-coordinate
        bl, br = sorted(bottom_points, key=lambda p: p[0])
        
        return [tl, tr, br, bl]

    def _get_mini_court_corners(self, mini_court):
        """
        Get the 4 corners of the 2D mini court.
        """
        # Top-Left, Top-Right, Bottom-Right, Bottom-Left
        tl = (mini_court.court_start_x, mini_court.court_start_y)
        tr = (mini_court.court_end_x, mini_court.court_start_y)
        br = (mini_court.court_end_x, mini_court.court_end_y)
        bl = (mini_court.court_start_x, mini_court.court_end_y)
        return [tl, tr, br, bl]

    def get_mini_court_coordinates(self, object_position):
        """
        Convert a single (x, y) video coordinate to (x, y) mini-court coordinate.
        """
        if self.homography_matrix is None:
            return object_position
            
        # Convert to homogenous coordinates
        pt = np.array([[[object_position[0], object_position[1]]]], dtype=np.float32)
        dst_pt = cv2.perspectiveTransform(pt, self.homography_matrix)
        
        return (int(dst_pt[0][0][0]), int(dst_pt[0][0][1]))
