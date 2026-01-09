import os
import shutil
from time import sleep
from datetime import datetime
from pathlib import Path
from loguru import logger
from PyQt6.QtCore import QFileInfo

class FileInfo:
    def __init__(self, path: Path):
        self.path = path
        self.name = path.name
        self.size = path.stat().st_size
        self.modified_time = datetime.fromtimestamp(path.stat().st_mtime)
        self.created_time = datetime.fromtimestamp(path.stat().st_ctime)

def __repr__(self):
        return (f"FileInfo(path='{self.path}', name='{self.name}', created='{self.created_time}', "
                f"modified='{self.modified_time}', size={self.size})")

class DiskManager():
    def __init__(self, mw):
        self.mw = mw

    def list_files(self, directory: str) -> tuple[list[FileInfo], int]:
        total_size = 0
        file_infos = []
        for dirpath, dirnames, filenames in os.walk(directory):
            for f in filenames:
                try:
                    file_info = FileInfo(Path(os.path.join(dirpath, f)))
                    total_size += file_info.size
                    file_infos.append(file_info)
                except Exception as ex:
                    print(f"file listing exception: {ex}")
                    pass

        file_infos.sort(key=lambda x: x.created_time)
        return file_infos, total_size

    def removeAssociatedPictureFiles(self, filename):
        try:
            info = QFileInfo(filename)
            finish = info.lastModified()
            dir = info.absoluteDir().dirName()
            pic_dir = os.path.join(self.mw.settingsPanel.storage.dirPictures.txtDirectory.text(), dir)
            files = os.listdir(pic_dir)
            for file in files:
                pic_info = QFileInfo(os.path.join(pic_dir, file))
                stem = Path(file).stem
                if len(stem) == 14 and stem.isnumeric():
                    target = pic_info.birthTime()
                    #if (target >= start and target <= finish):
                    # just wipe all the older pictures
                    if target <= finish:
                        os.remove(os.path.join(pic_dir, file))
        except Exception as ex:
            logger.error(f'Exception occurred during removal of associated picture files: {ex}')

    def manageDirectory(self, d):
        print("manage directory")

        files, size = self.list_files(d)
        max_size = self.mw.settingsPanel.storage.spnDiskLimit.value() * 1_000_000_000

        files_to_be_deleted = []
        total = 0
        diff = size - max_size
        if diff > 0:
            for file in files:
                total += file.size
                files_to_be_deleted.append(file)
                if total > diff:
                    break

        for file in files_to_be_deleted:
            try:
                self.removeAssociatedPictureFiles(str(file.path))
                os.remove(file.path)
            except Exception as ex:
                logger.error(f'File delete error: {ex}')
                pass
        
        self.mw.settingsPanel.storage.signals.updateDiskUsage.emit()
        print("DONE MANAGING DIRECTORY")
