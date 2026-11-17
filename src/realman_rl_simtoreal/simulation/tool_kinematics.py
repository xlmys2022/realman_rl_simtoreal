"""Rigid tool-frame selection over the existing, unchanged FK/IK solver."""
import numpy as np

from .kinematics import Kinematics


class ToolKinematics(Kinematics):
    """Reuse the arm solver with an explicitly selected, rigidly attached TCP.

    The original flange-frame Kinematics class and its legacy callers remain
    unchanged. Only tool-frame initialization and accessory FK live here.
    """

    def __init__(self, model, tcp_site):
        super().__init__(model)
        try:
            site = model.site(tcp_site).id
        except KeyError as exc:
            raise ValueError("Missing TCP site: " + tcp_site) from exc
        body = int(model.site_bodyid[site])
        while body and body != model.body("link_6").id:
            if model.body_jntnum[body]:
                raise ValueError("TCP must be rigidly attached to link_6")
            body = int(model.body_parentid[body])
        if not body:
            raise ValueError("TCP must belong to the arm")
        self.site_id = site
        self.home_position, self.home_rotation = self.fk(self.home_q)
        body = int(model.site_bodyid[site])
        self.reach_bound = float(np.linalg.norm(model.site_pos[site]))
        while body:
            self.reach_bound += float(np.linalg.norm(model.body_pos[body]))
            body = int(model.body_parentid[body])

    def body_pose(self, q, body_name):
        """Read an accessory frame from private FK data without dynamics writes."""
        self._forward(self._angles(q))
        body = self.model.body(body_name).id
        return self._data.xpos[body].copy(), self._data.xmat[body].reshape(3, 3).copy()
