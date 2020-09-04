

from .base import register_view

from survos2.frontend.components.base import QCSWidget

@register_view(name='load_workspace', title='Load Workspace')
class WorkspaceLoader(QCSWidget):
    pass