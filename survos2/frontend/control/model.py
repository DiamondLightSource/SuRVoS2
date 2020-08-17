

from .singleton import Singleton


@Singleton
class DataModel(object):

    def __init__(self):
        self.server_uri = None
        self.current_session = 'default'
        self.current_workspace = 'test_s17'
        self.current_workspace_shape = (99, 256, 256)

    def dataset_uri(self, dataset, group=None):
        session = self.current_session
        workspace = self.current_workspace

        if group:
            params = session, workspace, group, dataset
            return 'survos://{}@{}:{}/{}'.format(*params)
                
        return 'survos://{}@{}:{}'.format(session, workspace, dataset)

    def dataset_name(self, dataset_uri):
        return dataset_uri.split(':')[-1]

    
    #@property
    #def current_workspace(self):
    #    self.current_workspace