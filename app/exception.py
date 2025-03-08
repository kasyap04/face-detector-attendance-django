class AppException(Exception):
    def __init__(self, code, msg):
        self.msg = msg
        self.code = code
        super().__init__(code, msg)

    def __str__(self):
        return f"\n[ERROR]  AppException(code={self.code} error={self.msg})\n"