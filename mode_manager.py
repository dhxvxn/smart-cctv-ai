class ModeManager:
    def __init__(self):
        self.mode = "full"  # default

    def set_mode(self, mode):
        if mode in ["full", "query"]:
            self.mode = mode

    def is_query_mode(self):
        return self.mode == "query"