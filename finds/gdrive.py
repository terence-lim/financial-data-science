"""Convenience class methods to use google drive apis

- google REST api's

Author: Terence Lim
License: MIT
"""
import pickle
import os.path
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import time

to_url = "https://drive.google.com/uc?export=download&id=".format

# If modifying scopes, delete the file token.pkl.
_SCOPES = ['https://www.googleapis.com/auth/drive']
_FOLDER = 'application/vnd.google-apps.folder'   # mimeType of folders
_HOME = 'root'   # fileId of root folder

class GDRIVE:
    """Base class provides basic interface to essential google drive api's

    Attributes
    ----------
    to_url : str formatter
        prepend to shared google drive file_id to construct url for download
        'https://drive.google.com/uc?export=download&id='

    Examples
    --------
    # init() prompts for googleapi authorization, which is stored in a tokenfile
    #   in current folder so that not prompted in subsequent executions
    g = GDRIVE()

    _ = g.ls()                   # display list of folders and files (and types)

    # NOTE: in all methods, set silent=True to turn off raise exceptions
    #   and return None on error (letting caller handle errors gracefully)
    items = g.ls(silent=True)  # returns files+folders and their fields

    g.cd()                       # returns remote current working directory
    g.mkdir('/new_folder')       # create new folder with absolute path name
    g.cd('new_folder')           # change remote cwd
    g.mkdir('another_folder')    # create new folder in remote cwd
    g.put('local.txt', 'another_folder/newfile.txt')   # upload, relative path
    g.put('local.txt', '/new_folder')    # upload, absolute path, infer filename
    g.put('local.txt')                   # upload to remote cwd, infer filename
    g.get('/new_folder/local.txt', 'localfile.txt')   # upload, absolute path
    g.get('another_folder/newfile.txt')  # upload, relative path, infer filename
    g.rm('/new_folder/local.txt')        # remove, absolute path
    g.rm('another_folder/newfile.txt')   # remove, relative path
    g.rmdir('another_folder')            # remove folder from remote cwd
    g.rmdir('/new_folder')               # remove folder with absolute path name

    Notes
    -----
    See https://developers.google.com/drive/api/v3/quickstart/python
    to turn on the Drive API.  In resulting dialog click DOWNLOAD CLIENT 
    CONFIGURATION and save the file credentials.json to your working directory.

    Initially, the class will attempt to open browser to prompt for
    authorization (information is stored in tokenfile, so subsequent executions
    will not prompt).  If this fails, copy the URL from the console and manually
    open it in your browser.

    Requirements
    ------------
    pip install --upgrade \
        google-api-python-client google-auth-httplib2 google-auth-oauthlib
    """

    to_url = "https://drive.google.com/uc?export=download&id={}".format
    
    def __init__(self, tokenfile='token.pkl'):
        """Prompt for authorization, then save for subsequent executions

        Parameters
        ----------
        tokenfile : str, default is 'token.pkl'
            locally stores authorization information for subsequent executions
        """
        creds = None
        # The file token.pkl stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the 
        # first time.  See https://developers.google.com/drive/api/v3/
        if os.path.exists(tokenfile):
            with open(tokenfile, 'rb') as token:
                creds = pickle.load(token)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', _SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open(tokenfile, 'wb') as token:
                pickle.dump(creds, token)
        self.service = build('drive', 'v3', credentials=creds)
        self.pwd = '/'   # initial remote current working directory

    def _fullpath(self, path):
        """Helper method to clean and complete path str"""
        if path == '.':
            path = self.pwd
        if path == '..':
            path = os.path.abspath(os.path.join(self.pwd, path))
        while len(path)> 1 and path.endswith('/'):     # no trailing '/'
            path = path[:-1]
        if not path.startswith('/'):  # prepend cwd if not absolute path name
            path = os.path.join(self.pwd, path)
        while path.startswith('//'):  # remove repeated '//'
            path = path[1:]
        return os.path.abspath(path) if len(path) else '/'

    def fetchall(self, q=f"'{_HOME}' in parents",
                 filters={'ownedByMe': True, 'trashed':False},
                 fields=['id','name','parents','mimeType'], silent=True):
        """Helper method to fetch list of all files

        Parameters
        ----------
        q : str, default "'root' in parents" (select files in top-level folder)
            Search query string with which to call service.files().list()
        filters : list of dict, default is {'ownedByMe': True, 'trashed':False}
            Files metadata properties to filter on, see
            https://developers.google.com/drive/api/v3/reference/files
        fields : list of str, default is ['id','name','parents','mimeType']
            Specific fields of files to return, see
            https://developers.google.com/drive/api/v3/fields-parameter

        Returns
        -------
        result : list of dicts
            dicts of specified fields for list of files that pass filter

        Notes
        -----
        See https://developers.google.com/drive/api/v3/search-files
        """
        results = []
        page_token = None
        fields = ",".join(set(fields).union(set(filters.keys())))
        while True:  # max pageSize 1000, so loop until done (page_token None)
            r = self.service.files().list(
                q=q, pageSize=1000, pageToken=page_token, 
                spaces='drive',   # not photos nor appDataFolder
                fields=f'nextPageToken, files({fields})').execute(num_retries=9)
            items = r.get('files', [])
            results += items
            page_token = r.get('nextPageToken')
            if page_token is None:
                break
        return [r for r in results if all(r[k]==v for k,v in filters.items())]

    def fetchone(self, path, field=None):
        """Fetch metadata for one file given its path name

        Parameters
        ----------
        path : str
            remote path name
        field : str in {'id', 'name', 'mimeType', 'parents'}, default is None
            specific metadata field to return.  None to return all fields

        Returns
        -------
        match : value, or dict of values, or None
            value or dict of metadata values of matched file. None if not found
        """
        def parts(p):
            """helper method to split path into parts (why not in os.path???)"""
            p, t = os.path.split(p)
            return ((parts(p) if p != '/' else ['/']) + ([t] if t else []))

        path = self._fullpath(path)
        match = {'id': _HOME, 'name': '/', 'mimeType': _FOLDER, 'parents': []}
        for p in parts(path)[1:]:    # sequentially try to match path name parts
            items = self.fetchall(f"'{match['id']}' in parents",
                                  fields=match.keys())
            matches = [item for item in items if item['name'] == p]
            if matches:
                match = matches[0]
            else:      # this part of path name has no match, so exit the method
                return None
        return match if field is None else match[field]

    def ls(self, path='.', silent=False):
        """Return list of files in remote folder"""
        path = self._fullpath(path)
        item = self.fetchone(path)
        if item is None:
            if not silent:    # raise Exception if verbose
                raise Exception(f"Cannot ls {path}")
            return None    # else silently return None
        elif item['mimeType'] != _FOLDER:
            files = [item]
            if not silent:
                print(f"{path}  {item['mimeType']}")
        else:
            files = self.fetchall(f"'{item['id']}' in parents")
            if not silent:
                maxlen, ndirs, nfiles = 0, 0, 0
                for item in files:
                    if item['mimeType'] == _FOLDER:
                        print(os.path.join(path, item['name']) + '/')
                        ndirs += 1
                    else:
                        maxlen = max(maxlen, len(item['name']))
                        nfiles += 1
                for item in files:
                    if item['mimeType'] != _FOLDER:
                        print("{:{width}}  {}".format(
                            item['name'], item['mimeType'], width=maxlen))
                print(f"  '{path}' has {ndirs} folders, {nfiles} files")
        return files

    def dir(self, path='.', silent=False):
        """Return list of files in remote folder"""
        return self.ls(path=path, silent=silent)
        
    def cd(self, path=None, silent=False):
        """Change remote working directory"""
        if path is not None:   # return cwd if no args
            path = self._fullpath(path)
            if self.fetchone(path, field='mimeType') != _FOLDER:
                if not silent:    # raise Exception if verbose
                    raise Exception('Cannot cd ' + path)
                return None    # else silently return
            self.pwd = path
        return self.pwd
        
    def mkdir(self, path, silent=False):
        """Create remote folder"""
        path = self._fullpath(path)
        fileId = self.fetchone(path, field='id')    # check if already exists
        folder, base = os.path.split(path)  # try to create new in parent folder
        folderId = self.fetchone(folder, field='id')
        if folderId is None or fileId is not None:
            if not silent:
                raise Exception(f"Cannot mkdir {path}")
            return None
        body = {'name': base, "parents": [folderId], 'mimeType': _FOLDER}
        self.service.files().create(body=body).execute()

    def md(self, path, silent=False):
        """Create remote folder"""
        return self.mkdir(path=path, silent=silent)

    def rmdir(self, path, silent=False):
        """Remove remote folder and all its contents, bypassing trash bin"""
        path = self._fullpath(path)
        fileId = self.fetchone(path, field='id')
        if fileId is None:
            if not silent:        # raise Exception if verbose
                raise Exception(f"Cannot remove {path}")
            return None        # else silently return None
        self.service.files().delete(fileId=fileId).execute()
        if not silent:
            print(f"{path} '{fileId}' deleted")
        if path == self.pwd:   # if deleted cwd, then cd to parent
            self.pwd = os.path.dirname(path)

    def rm(self, path, silent=False):
        """Remove remote file, bypassing trash bin"""
        if self.fetchone(path, field='mimeType') == _FOLDER:
            if not silent:  # raise Exception, else silently return None
                raise Exception(f"Use rmdir to rm {path} and all its contents")
        return self.rmdir(path)

    def delete(self, path, silent=False):
        """Remove remote file, bypassing trash bin"""
        return self.rm(path=path, silent=silent)

    def get(self, path, filename='', silent=False):
        """Download a remote file to local filename

        Parameters
        ----------
        path : str
            relative or absolute path name of remote file
        filename : str, default is ''
            Output filename.  If blank: infer basename, save in curr local dir
        silent : bool, default False
            if True, print file statistics if successful, raise exception if not

        Returns
        -------
        total_file_size : int
            or None if unsuccessful and silent

        Notes
        -----
        see https://developers.google.com/drive/api/v3/manage-downloads
        """
        path = self._fullpath(path)
        fileId = self.fetchone(path, field='id')
        tic = time.time()
        media = self.service.files().get_media(fileId=fileId)
        if not filename:
            filename = os.path.basename(path)
        with open(filename,'wb') as f:
            downloader = MediaIoBaseDownload(f, media)
            done = False
            while done is False:
                try:
                    status, done = downloader.next_chunk()
                except Exception as e:
                    if not silent:   # raise Exception if verbose
                        raise Exception(str(e))
                    return None   # else silently return
        if not silent:
            print("Retrieved to {}, len={},  in {:.0f} secs".format(
                filename, status.total_size, time.time()-tic))
        return status.total_size

    def put(self, filename, path='', silent=False):
        """Upload a local filename to remote folder file

        Parameters
        ----------
        filename : str
            Input local filename
        path : str, default is ''
            Absolute or relative output filename or folder (basename inferred
            from local filename).  If blank: upload to current remote folder
        silent : bool, default False
            if False, print file statistics if successful

        Returns
        -------
        total_file_size : int
            or None if unsuccessful and silent flag is set to True

        Notes
        -----
        see https://developers.google.com/drive/api/v3/manage-uploads
        """
        if not path:     # if path not specified, put in current remote folder
            path = self.pwd
        path = self._fullpath(path)
        if self.fetchone(path, field='mimeType') == _FOLDER:
            base = os.path.basename(filename)  # infer base from input filename
            folder = path                      #   if output path is folder
        else:
            folder, base = os.path.split(path)
        media_body = MediaFileUpload(filename, resumable=True)

        fileId = self.fetchone(os.path.join(folder, base), field='id')
        tic = time.time()
        if fileId:     # fileId already exists, so use update method
            self.service.files().update(
                fileId=fileId, media_body=media_body).execute()
        else:
            folderId = self.fetchone(folder, field='id')  # try to get folder ID
            if folderId is None:
                if not silent:       # raise Exception if verbose
                    raise Exception('Cannot put ' + path)
                return None       # else silently return
            self.service.files().create(     # method creates new file in folder
                body={'name': base, "parents": [folderId]},
                media_body=media_body).execute()
        if not silent:
            print("{} to {}, len={}, in {:.0f} secs".format(
                "Updated" if fileId else "Saved", 
                os.path.join(folder, base), media_body.size(), time.time()-tic))
        return media_body.size()

    def chmod(self, path, action='list', role='reader', email=None,
              silent=False):
        """List, delete or create sharing permissions for a remote file

        Parameters
        ----------
        path : str
            name of remote file to change permissions of
        action : str in {'list' (default), 'create', 'delete'}
            permissions action to execute
        role : str in {'reader' (default), 'writer', 'owner', 'commenter'}
            role to permit

        Returns
        -------
        result: int or str
            number of permissions listed or deleted, or shareable link created
        """
        path = self._fullpath(path)
        fileId = self.fetchone(path, field='id')
        if fileId is None:
            if not silent:
                raise Exception(f"Cannot chmod {path}")
            return None
        items = self.service.permissions().list(
            fileId=fileId).execute().get('permissions', [])
        if action[0].lower() == 'c':       # action is create
            roles = ['reader', 'writer', 'owner', 'commenter']
            role = {r[0].lower(): r for r in roles}.get(role[0].lower())
            if role is None:
                if not silent:
                    raise Exception(f'chmod invalid role not in {roles}')
                return None
            body = {'role': role, 
                    'type': 'anyone' if email is None else "user",
                    'value': email,
                    'emailAddress': email}
            item = self.service.permissions().create(
                fileId=fileId, body=body).execute()
            return self.service.files().get(
                fileId=fileId, fields='webContentLink').execute()
        elif action[0].lower() == 'd':     # action is delete
            for item in items:
                if item['role'] != 'owner':
                    self.service.permissions().delete(
                        fileId=fileId, permissionId=item['id']).execute()
                    if not silent:
                        print(item)
        else:      # action is list
            if not silent:
                for item in items:
                    print(item)
                print(f"{len(items)} permissions exist in {path}")
        return items

if __name__ == '__main__':
    g = GDRIVE()
    silent=False
    _ = g.ls(silent=silent)

