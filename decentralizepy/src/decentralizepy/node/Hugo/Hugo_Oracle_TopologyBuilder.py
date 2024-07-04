from decentralizepy.node.PeerSamplerDynamic import PeerSamplerDynamic


class Hugo_Oracle_TopologyBuilder(PeerSamplerDynamic):
    """
    This class defines the topology builder that responds to neighbor requests from the clients.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
