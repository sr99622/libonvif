import os
import shutil
from time import sleep
from pathlib import Path
from loguru import logger
from PyQt6.QtCore import QFileInfo

class DiskManager():
    def __init__(self, mw):
        self.mw = mw
        self.thread_lock = False

    def lock(self):
        # the lock protects the players during size calculations *maybe not necessary
        while self.thread_lock:
            sleep(0.001)
        self.thread_lock = True

    def unlock(self):
        self.thread_lock = False

    def estimateFileSize(self, uri):
        # duration is in seconds, cameras report bitrate in kbps (usually), result in bytes
        result = 0
        bitrate = 0
        profile = self.mw.cameraPanel.getProfile(uri)
        if profile:
            audio_bitrate = min(profile.audio_bitrate(), 128)
            video_bitrate = min(profile.bitrate(), 16384)
            bitrate = video_bitrate + audio_bitrate
        result = (bitrate * 1000 / 8) * self.mw.STD_FILE_DURATION
        return result

    def getCommittedSize(self):
        committed = 0
        for player in self.mw.pm.players:
            if player.isRecording():
                committed += self.estimateFileSize(player.uri) - player.pipeBytesWritten()
        return committed

    def getDirectorySize(self, d):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(d):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    try:
                        total_size += os.path.getsize(fp)
                    except FileNotFoundError:
                        pass

        dir_size = "{:.2f}".format(total_size / 1000000000)
        self.mw.settingsPanel.storage.grpDiskUsage.setTitle(f'Disk Usage (currently {dir_size} GB)')
        return total_size
    
    def getOldestFile(self, d):
        oldest_file = None
        oldest_time = None
        for dirpath, dirnames, filenames in os.walk(d):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):

                    stem = Path(fp).stem
                    if len(stem) == 14 and stem.isnumeric():
                        try:
                            if oldest_file is None:
                                oldest_file = fp
                                oldest_time = os.path.getmtime(fp)
                            else:
                                file_time = os.path.getmtime(fp)
                                if file_time < oldest_time:
                                    oldest_file = fp
                                    oldest_time = file_time
                        except FileNotFoundError:
                            pass
        return oldest_file
    
    def removeAssociatedPictureFiles(self, filename):
        try:
            info = QFileInfo(filename)
            alarm_buffer_size = self.mw.settingsPanel.alarm.spnBufferSize.value()
            start = info.birthTime().addSecs(-alarm_buffer_size)
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

    def getMaximumDirectorySize(self, d, uri):
        estimated_file_size = self.estimateFileSize(uri)
        space_committed = self.getCommittedSize()
        allowed_space = min(self.mw.settingsPanel.storage.spnDiskLimit.value() * 1000000000, shutil.disk_usage(d)[2])
        return allowed_space - (space_committed + estimated_file_size)
    
    def manageDirectory(self, d, uri):
        self.lock()
        try:
            while self.getDirectorySize(d) > self.getMaximumDirectorySize(d, uri):
                if oldest_file := self.getOldestFile(d):
                    self.removeAssociatedPictureFiles(oldest_file)
                    os.remove(oldest_file)
                    #logger.debug(f'File has been deleted by auto process: {oldest_file}')
                else:
                    logger.debug("Unable to find the oldest file for deletion during disk management")
                    break
        except Exception as ex:
            logger.error(f'Directory Manager exception: {ex}')
        self.unlock()