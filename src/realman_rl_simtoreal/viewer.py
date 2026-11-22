"""旧 Viewer 导入入口，供已有外部演示调用。"""
from .apps.visualization.viewer import configure_backend, passive_viewer

__all__ = ['configure_backend', 'passive_viewer']
