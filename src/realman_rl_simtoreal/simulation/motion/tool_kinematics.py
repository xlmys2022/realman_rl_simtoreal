"""为抓取 TCP 及腕部附件提供显式工具坐标系，复用法兰 FK/IK 的数值实现。"""
import numpy as np

from .kinematics import Kinematics


class ToolKinematics(Kinematics):
    """按 tcp_site 选择工具端点，复用 Kinematics 的 fk、jacobian 和 solve_ik。

    tcp_site 必须与 link_6 刚性相连。home_position 和 home_rotation 随工具点更新，
    关节角仍使用 rad，所有工具位姿均以世界系表示。
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
        """输入关节角（rad）和刚体名，返回其世界系位置（m）和姿态矩阵。"""
        self._forward(self._angles(q))
        body = self.model.body(body_name).id
        return self._data.xpos[body].copy(), self._data.xmat[body].reshape(3, 3).copy()
