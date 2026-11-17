"""Orange cube localization from aligned RGB-D and calibrated camera geometry.

Inputs are sensor arrays and calibration only. No simulator object IDs, poses,
contacts, velocities, reset coordinates or segmentation labels are accepted.
The current object model is a known, upright 20-mm cube on a horizontal table.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RgbdFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    camera_to_base: np.ndarray
    timestamp: float
    distortion: tuple = ()

    def __post_init__(self):
        h,w=self.depth_m.shape
        if self.rgb.shape!=(h,w,3) or self.rgb.dtype!=np.uint8:
            raise ValueError('Aligned RGB uint8 and metric optical-Z depth are required')
        if self.intrinsics.shape!=(3,3) or self.camera_to_base.shape!=(4,4):
            raise ValueError('Calibrated 3x3 intrinsics and 4x4 camera-to-base transform required')
        if not np.isfinite(self.intrinsics).all() or not np.isfinite(self.camera_to_base).all():
            raise ValueError('Finite camera calibration required')
        if min(self.intrinsics[0,0],self.intrinsics[1,1])<=0 or not np.isfinite(self.timestamp):
            raise ValueError('Positive focal lengths and finite capture time required')
        rotation=self.camera_to_base[:3,:3]
        if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(rotation),1):
            raise ValueError('Camera extrinsics must be a rigid rotation')
        if not np.allclose(self.camera_to_base[3],[0,0,0,1]):raise ValueError('Invalid homogeneous transform')


@dataclass(frozen=True)
class CubeMeasurement:
    position_m: np.ndarray
    normal: np.ndarray
    confidence: float
    pixels: int
    centroid_px: tuple
    timestamp: float


def orange_mask(rgb,low=(3,175,50),high=(25,255,255)):
    import cv2
    if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8:raise ValueError('Expected RGB uint8 image')
    lower=np.asarray(low);upper=np.asarray(high)
    if (lower.shape!=(3,) or upper.shape!=(3,) or not np.isfinite([lower,upper]).all()
            or np.any(lower<0) or np.any(upper>[179,255,255]) or np.any(lower>upper)
            or np.any(lower!=lower.astype(int)) or np.any(upper!=upper.astype(int))):raise ValueError('Invalid HSV thresholds')
    lower=lower.astype(np.uint8);upper=upper.astype(np.uint8)
    return cv2.inRange(cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV),lower,upper)


def backproject(frame,pixels_xy):
    """Pixel (u,v) and aligned optical-Z depth -> metric robot-base points."""
    import cv2
    pixels=np.asarray(pixels_xy,dtype=float).reshape(-1,2)
    uv=np.rint(pixels).astype(int);h,w=frame.depth_m.shape
    if np.any(uv<0) or np.any(uv[:,0]>=w) or np.any(uv[:,1]>=h):raise ValueError('Pixels outside aligned images')
    depth=frame.depth_m[uv[:,1],uv[:,0]].astype(float)
    rays=cv2.undistortPoints(pixels.reshape(-1,1,2),frame.intrinsics,np.asarray(frame.distortion) if frame.distortion else None).reshape(-1,2)
    camera=np.c_[rays*depth[:,None],depth]
    return camera@frame.camera_to_base[:3,:3].T+frame.camera_to_base[:3,3]


class OrangeCubeDetector:
    def __init__(self,*,edge_m=.02,hsv_low=(3,175,50),hsv_high=(25,255,255),minimum_pixels=8):
        if not .005<=edge_m<=.1 or minimum_pixels<4:raise ValueError('Invalid cube or component size')
        self.edge_m=edge_m;self.low=tuple(hsv_low);self.high=tuple(hsv_high);self.minimum_pixels=minimum_pixels
        self.mask=None

    def detect(self,frame):
        import cv2
        mask=orange_mask(frame.rgb,self.low,self.high)
        n,labels,stats,centroids=cv2.connectedComponentsWithStats(mask)
        self.mask=np.zeros_like(mask)
        if n<=1:return None
        which=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
        selected=labels==which;self.mask[selected]=255
        v,u=np.nonzero(selected)
        if len(u)<self.minimum_pixels:return None
        points=backproject(frame,np.c_[u,v]);depth=frame.depth_m[v,u]
        valid=np.isfinite(points).all(axis=1)&np.isfinite(depth)&(depth>.02)&(depth<3.)
        points=points[valid]
        if len(points)<self.minimum_pixels:return None
        # Reject mixed edge pixels/background. Both the surface and this cutoff
        # come from the measured point cloud, never the table/object true pose.
        median_depth=float(np.median(depth[valid]))
        points=points[np.abs(depth[valid]-median_depth)<self.edge_m*2]
        if len(points)<self.minimum_pixels:return None
        top=float(np.percentile(points[:,2],90))
        points=points[points[:,2]>top-self.edge_m*.95]
        if len(points)<self.minimum_pixels:return None
        low,high=np.percentile(points[:,:2],[1,99],axis=0)
        if np.any(high-low>self.edge_m*1.5):return None
        # RGB centroid is biased by the fingers. Recover the center from the
        # near visible cube faces and the explicitly configured physical size.
        camera=frame.camera_to_base[:3,3]
        side=np.sign(camera[:2]-np.median(points[:,:2],axis=0))
        xy=np.where(side>=0,high-self.edge_m/2,low+self.edge_m/2)
        position=np.r_[xy,top-self.edge_m/2]
        surface=points[points[:,2]>top-.0015]
        normal=np.array([0.,0.,1.]);plane_confidence=.5
        if len(surface)>=6:
            _,singular,vectors=np.linalg.svd(surface-surface.mean(axis=0),full_matrices=False)
            if singular[1]>1e-5:
                fitted=vectors[-1]
                if fitted[2]<0:fitted=-fitted
                if fitted[2]>.85:normal=fitted;plane_confidence=1.
        confidence=float(min(1.,len(points)/40)*(.7+.3*plane_confidence))
        return CubeMeasurement(position,normal,confidence,len(points),tuple(centroids[which]),frame.timestamp)


class VisualTracker:
    """Finite-difference velocity and bounded stale camera estimates."""
    def __init__(self,detector,*,timeout=.25):
        self.detector=detector;self.timeout=timeout;self.reset()

    def reset(self):
        self.measurement=None;self.velocity=np.zeros(3);self.last_capture=None;self.visible=False

    def update(self,frame):
        found=self.detector.detect(frame);self.last_capture=frame.timestamp;self.visible=found is not None
        if found is not None:
            if self.measurement is not None:
                dt=found.timestamp-self.measurement.timestamp
                if dt<=0:raise ValueError('Camera timestamps must increase')
                raw=(found.position_m-self.measurement.position_m)/dt
                self.velocity=.25*raw+.75*self.velocity
            self.measurement=found
        return found

    def state(self,now):
        if self.measurement is None:raise ValueError('Vision target not detected at observation pose')
        m=self.measurement;age=max(0.,float(now-m.timestamp))
        # Stale state is flagged. It is not attached to the TCP or replaced by
        # a simulator pose. The environment stops after the bounded timeout.
        return dict(position_m=m.position_m.copy(),velocity_m_s=self.velocity.copy(),normal=m.normal.copy(),
                    confidence=m.confidence if self.visible else 0.,visible=self.visible,age=age,
                    expired=age>self.timeout,pixels=m.pixels,centroid_px=m.centroid_px)
