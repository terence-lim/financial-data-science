from typing import Dict, Mapping, Any
from pathlib import Path

credentials: Mapping[str, Mapping[str, Any]] = {
    'sql': {'user': '',
            'password': '',
            'database': '',
            'host': 'localhost',
            'port': 3306
    },
    'user': {'user': '',
             'password': '',
             'database': '',
             'host': 'localhost',
             'port': 3306,
    },
    'redis': {'host': 'localhost',
              'port': 6379,
              'charset': "utf-8",
              'decode_responses': False,

    },
    'mongodb': {'host': 'localhost',
                'port': 27017,
    },
    'googleapi': {'id': '',
                  'secret':  ''},
    'bea': {'userid': ''},
    'fred': {'api_key': ""}
}

paths: Mapping[str, Any] = {
    'images': Path(''),
    'taq': Path(''),
    'scratch': Path(''),
    'downloads': Path(''),
    '10X': Path(''),
}

