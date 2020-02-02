class TEST:
    def __init__(self, s):
        self._str = s

    def __str__(self):
        return self._str

if __name__ == "__main__":
    import tests
    a = tests.TEST('hello, world')
    print(a)

