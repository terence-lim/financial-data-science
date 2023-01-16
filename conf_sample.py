from typing import Dict, Mapping, Any
from pathlib import Path

# pip install pyqt5
#import matplotlib
#matplotlib.use('Qt5Agg')

CRSP_DATE: int = 20220331
VERBOSE: int = 1

credentials: Mapping[str, Mapping[str, Any]] = {
    'sql': {'user': '...',
            'password': '...',
            'database': '...',
            'host': '...',
            'port': 3306
    },
    'user': {'user': '...',
             'password': '...',
             'database': '...',
             'host': '...',
             'port': 3306,
    },
    'redis': {'host': '...',
              'port': 6379,
              'charset': "utf-8",
              'decode_responses': False,

    },
    'mongodb': {'host': '...',
                'port': 27017,
    },
    'googleapi': {'id': '...',
                  'secret':  '...'},
    'bea': {'userid': '...'},
    'fred': {'api_key': "..."}
}

paths: Mapping[str, Any] = {
    'yahoo': Path('...'),
    'images': Path('...'),
    'taq': Path('...'),
    'scratch': Path('...'),
    'downloads': Path('...'),
    '10X': Path('...'),
}

paths: Mapping[str, Any] = {
    'images': '...',
    'taq': '...',
    'scratch': '...',
    'downloads': '...',
    '10X': '...',
}

