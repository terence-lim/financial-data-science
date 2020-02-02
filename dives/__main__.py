import dives
if __name__ == '__main__':
    import importlib
    importlib.reload(dives)
    importlib.reload(dives.util)
    importlib.reload(dives.custom)
    print('dives/__main__: reload dives')
