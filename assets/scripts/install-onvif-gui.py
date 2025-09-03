#********************************************************************
# libonvif/assets/scripts/install-onvif-gui.py
#
# Copyright (c) 2025  Stephen Rhodes
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#*********************************************************************/

import os
import sys
import shutil
import platform
import argparse

ERROR = 1
SUCCESS = 0

INSTALL_SOURCE = "pypi"


#INSTALL_SOURCE = "smb"
SMB_SERVER = "//10.1.1.143/share"
SMB_DOMAIN = "WORKGROUP"
SMB_USERNAME = "username"
SMB_PASSWORD = "password"
SMB_MOUNT = "smb_share"
SMB_FOLDER = "home/safe/install/3.0.10/linux"

# credential = f"\"domain={domain},username={user},password={passwd}\""
# f'sudo mount -t cifs -o {credential} {host} {target}'

class Install():
    def __init__(self):
        self.status = SUCCESS
        self.venv = None

    def run_command(self, command, env=None, exit_on_error=False, silent=True):
        if env is None:
            env = ""
            env_dict = os.environ.copy()
            for key in env_dict:
                env += f'{key}={env_dict[key]}\n'

        stdout_r, stdout_w = os.pipe()
        stderr_r, stderr_w = os.pipe()
        pid = os.fork()

        if pid == 0:
            os.close(stdout_r)  
            os.close(stderr_r)
            os.dup2(stdout_w, 1)
            os.dup2(stderr_w, 2)
            os.close(stdout_w)
            os.close(stderr_w)
            os.execlp("bash", "bash", "-c", command, env)
        else:
            os.close(stdout_w)
            os.close(stderr_w)
            with os.fdopen(stdout_r) as stdout_pipe, os.fdopen(stderr_r) as stderr_pipe:
                output = stdout_pipe.read().strip()
                error_output = stderr_pipe.read().strip()
            pid, self.status = os.waitpid(pid, 0)

            if os.WIFEXITED(self.status) and os.WEXITSTATUS(self.status) != 0:
                if not silent:
                    print(f"Command failed: {command}", file=sys.stderr)
                    print(f"Error message: {error_output}", file=sys.stderr)
                if exit_on_error:
                    sys.exit(os.WEXITSTATUS(self.status))
            return output + error_output

    def install_package(self, pkg_mgr, name):
        print(f"installing {name} using package manager {pkg_mgr}")

        if pkg_mgr == "apt":
            format = "-Wf'${db:Status-abbrev}'"
            self.run_command(f"dpkg-query {format} {name} | grep -q '^i'")
            if self.status:
                print(f"Existing package was not found, installing {name}")
                print(self.run_command(f'sudo apt install -y {name}'))
                if self.status:
                    print(f"An error occurred installing the package: {name}, error code: {self.status}")
                    return ERROR
            else:
                print("Found existing package")
        
        if pkg_mgr == "dnf":
            print(self.run_command(f'sudo dnf install -y {name}'))
            if self.status:
                print(f"An error occurred installing the package: {name}, error code: {self.status}")
                return ERROR
        
        if pkg_mgr == "pacman":
            print("The installation program does not install packages on systems that use pacman")

        if pkg_mgr == "zypper":
            print(f"Installing {name} using zypper")
            self.run_command(f'sudo zypper install -y {name}')
            if self.status:
                print(f"An error occurred installing the package: {name}, error code: {self.status}")
            return ERROR


    def check_python(self, pkg_mgr):
        major, minor, micro = platform.python_version_tuple()
        print(f"The Python version for this installation is {major}.{minor}.{micro}")       
        supported = True 
        if int(major) < 3:
            print("Unsupported Python version, the minimum is 3.10")
            supported = False
        if int(minor) < 10:
            print("Unsupported Python version, the minimum is 3.10")
            supported = False

        if not supported:
            kernel_supported = False
            kernel_version = platform.release()
            print(f"kernel version {kernel_version}")
            if len(kernel_version) > 1:
                try:
                    kernel_major = int(kernel_version.split(".")[0])
                    kernel_minor = int(kernel_version.split(".")[1])
                    if kernel_major * 100 + kernel_minor >= 504:
                        kernel_supported = True
                except Exception as ex:
                    kernel_supported = False

            if not kernel_supported:
                print(f"Unsupported kernel {kernel_version}, the minimum required kernel version is 5.4")
                return ERROR
            
            print("Attempting to find a compatible python version")
            if valid_existing_python_version := self.check_for_alt_python_versions():
                print(f"\nThere is an existing supported version {valid_existing_python_version} on this machine.\nOnvif GUI will be installed using this version, please wait ...")
                print(self.run_command(f"{valid_existing_python_version} install-onvif-gui.py"))
            else:
                if response := input("Would you like to safely install an alternate Python version (3.12) for running Onvif GUI N/y? "):
                    if response.lower() == "y":
                        if self.build_python(pkg_mgr):
                            print("\n\nPython3.12 installation SUCCESSFUL. Press the enter key to continue ...")
                        else:
                            print("\n\n An error occurred attempting to build Python 3.12")
                        print(self.run_command('sudo rm -rf Python-3.12.9*'))
                        print(self.run_command("python3.12 install-onvif-gui.py"))
                    else:
                        print("Onvif GUI installation has been cancelled")
            return ERROR

    def build_python(self, pkg_mgr):
        
        if pkg_mgr == "apt":
            packages = ["curl", "libssl-dev", "zlib1g-dev", "libbz2-dev", "build-essential",
                        "libreadline-dev", "libsqlite3-dev", "wget", "llvm", 
                        "libncurses5-dev", "libncursesw5-dev", "xz-utils", "tk-dev", 
                        "libffi-dev", "liblzma-dev", "python-openssl", "git", "pkg-config"]
        
        elif pkg_mgr == "dnf":
            packages = ["gcc", "openssl-devel", "bzip2-devel", "libffi-devel", "wget", "tar", 
                        "make", "zlib-devel", "sqlite-devel", "xz-devel", "ncurses-devel", 
                        "readline-devel", "tk-devel", "libuuid-devel"]

        elif pkg_mgr == "zypper":
            packages = ["gcc", "make", "libssl-devel", "zlib-devel", "libbz2-devel",
                        "libffi-devel", "libreadline-devel", "libsqlite3-devel",
                        "tk-devel", "wget", "curl"]

        else:
            print("Unsupported package manager for Python compile")
            return False

        for package in packages:
            print(f"package: {package}")
            self.install_package(pkg_mgr, package)

        print(f"{self.run_command('wget https://www.python.org/ftp/python/3.12.9/Python-3.12.9.tgz')}")
        if self.status:
            print("An error occurred downloading the Python source code")
            return False
        
        print("Python 3.12 will be compiled from source. This will take some time ...")
        command = "tar -xf Python-3.12.9.tgz && cd Python-3.12.9 && ./configure --enable-optimizations && make -j$(nproc) && sudo make altinstall"
        output = self.run_command(command)
        if self.status:
            print(f'An error occurred during installation for Python3.12\v{output}')
            return False
        
        return True

    def check_for_alt_python_versions(self):
        self.run_command('python3.10 --version', silent=True)
        if not self.status:
            return "python3.10"
        self.run_command('python3.11 --version', silent=True)
        if not self.status:
            return "python3.11"
        self.run_command('python3.12 --version', silent=True)
        if not self.status:
            return "python3.12"
        self.run_command('python3.13 --version', silent=True)
        if not self.status:
            return "python3.13"
        return None

    def create_venv(self, venv, pkg_mgr):
        print(f"Creating virtual environment at {venv}")
        major, minor, _ = platform.python_version_tuple()
        print(self.run_command(f'python{major}.{minor} -m venv {venv}'))
        if self.status:
            print("Missing venv module, will attempt to install")
            pkg_name = f'python{major}.{minor}-venv'
            if self.install_package(pkg_mgr, pkg_name):
                return ERROR
            print(self.run_command(f'python{major}.{minor} -m venv {venv}'))
            if self.status:
                return ERROR
        self.venv = venv

    def check_display_protocol(self, pkg_mgr):
        # This function is allowed to fail without returning ERROR
        print("Checking display protocol")
        
        if protocol := os.environ.get("XDG_SESSION_TYPE"):

            if protocol.casefold() == "wayland".casefold():
                print("Display protocol is wayland, no further configuration required")
                return

            if protocol.casefold() == "x11".casefold():
                if pkg_mgr == "apt":
                    self.install_package(pkg_mgr, "libxcb-cursor-dev")
                if pkg_mgr == "dnf":
                    ...
                if pkg_mgr == "pacman":
                    ...
                if pkg_mgr == "zypper":
                    self.install_package(pkg_mgr, "libxcb-cursor0")
                    self.install_package(pkg_mgr, "libgthread-2_0-0")

            else:
                print(f"Unsupported display protocol: {protocol}")
        else:
            print(f"Unable to determine display protocol")
        
    def find_package_manager(self):
        self.run_command("apt list --installed")
        if not self.status:
            print("The package manager for this platform is apt")
            print(self.run_command('sudo apt update'))
            return "apt"
        self.run_command('dnf list')
        if not self.status:
            print("The package manager for this platform is dnf")
            #print(self.run_command('sudo dnf update'))
            return "dnf"
        self.run_command("pacman -Q")
        if not self.status:
            print("The package manager for this platform is pacman")
            #print(self.run_command('sudo pacman -Syy'))
            return "pacman"
        self.run_command("zypper refresh")
        if not self.status:
            print("The package manager for this platform is zypper")
        return "zypper"
        
        print("Unable to determine the package manager for this platform")
        return None
    
    def mount_smb(self, host, domain, user, passwd, target):
        try:
            os.makedirs(target, exist_ok=True)
        except Exception as ex:
            print(f"Error creating mount point: {ex}")
            return ERROR

        credential = f"\"domain={domain},username={user},password={passwd}\""

        print(self.run_command(f'sudo mount -t cifs -o {credential} {host} {target}'))
        if self.status:
            print(f"An error occurred while attempting to mount {host}")
            return ERROR
        
    def umount_smb(self, target):
        print(self.run_command(f'sudo umount {target}'))
        if self.status:
            print(f"An error occured during umount from {target}")
            return ERROR
        try:
            os.rmdir(target)
        except Exception as ex:
            print(f'An error occured while removing mount point {target}')
            return ERROR
        
    def install_onvif_gui(self, venv):
        print(self.run_command(f'source {venv}/bin/activate && pip install --upgrade pip && pip install onvif-gui'))
        if self.status:
            print("ERROR: Unable to properly install onvif-gui")
        return self.status
    
    def install_onvif_gui_local(self, venv, pkg_mgr):
        pkg_name = "cifs-utils"
        if self.install_package(pkg_mgr, pkg_name):
            return ERROR

        if self.mount_smb(SMB_SERVER, SMB_DOMAIN, SMB_USERNAME, SMB_PASSWORD, SMB_MOUNT):
            return ERROR
        
        major, minor, micro = platform.python_version_tuple()
        VER = f'{major}{minor}'
        dir = f'{SMB_MOUNT}/{SMB_FOLDER}'

        commands = [f"source {venv}/bin/activate && pip install --disable-pip-version-check {dir}/avio*{VER}*whl",
                    f"source {venv}/bin/activate && pip install --disable-pip-version-check {dir}/libonvif*{VER}*whl",
                    f"source {venv}/bin/activate && pip install --disable-pip-version-check {dir}/kankakee*{VER}*whl",
                    f"source {venv}/bin/activate && pip install --disable-pip-version-check {dir}/onvif*whl"]
        
        for command in commands:
            print(self.run_command(command))
            if self.status:
                print("An error occuring during pip install onvif-gui")
                return ERROR
            
        if self.umount_smb(SMB_MOUNT):
            return ERROR
        
    def check_for_nvidia(self):
        print(f"venv: {self.venv}")
        self.run_command('nvidia-smi')
        if self.status:
            print("Did not find NVIDIA drivers, installing CPU only pytorch")
            print(self.run_command(f'source {self.venv}/bin/activate && pip install --disable-pip-version-check torch torchvision --index-url https://download.pytorch.org/whl/cpu'))
        else:
            print("Found NVIDIA drivers, installing GPU version of pytorch, this may take a few minutes")
            print(self.run_command(f'source {self.venv}/bin/activate && pip install --disable-pip-version-check torch torchvision'))
        print(self.run_command(f'source {self.venv}/bin/activate && pip install --disable-pip-version-check openvino'))

    def rcwec(self, command, show=True):
        # run command with error check
        output = self.run_command(command)
        if show:
            print(output)
        if self.status:
            print(f'\n\n *** An error occurred running the command {command}\n\n')
        return self.status
    
    def check_for_intel_gpu(self, pkg_mgr):
        output = self.run_command("lspci -k | grep -EA3 'VGA|3D|Display'")
        if not "Intel" in output:
            return
        print(output)
        response = input("\nIn order to run YOLO on Intel iGPU, some drivers need to be installed. Would you like to install compute drivers for Intel iGPU? N/y ")
        if not response:
            return
        if response.lower() != "y":
            return

        if pkg_mgr == "apt":

            print(self.run_command("sudo apt install curl"))

            if self.rcwec("curl -OL https://github.com/intel/intel-graphics-compiler/releases/download/v2.10.8/intel-igc-opencl-2_2.10.8+18926_amd64.deb", False): return
            print("Downloaded intel-igc-opencl-2_2.10.8+18926_amd64.deb")
            if self.rcwec("curl -OL https://github.com/intel/compute-runtime/releases/download/25.13.33276.16/intel-level-zero-gpu-dbgsym_1.6.33276.16_amd64.ddeb", False): return
            print("Downloaded intel-level-zero-gpu-dbgsym_1.6.33276.16_amd64.ddeb")
            if self.rcwec("curl -OL https://github.com/intel/compute-runtime/releases/download/25.13.33276.16/intel-opencl-icd-dbgsym_25.13.33276.16_amd64.ddeb", False): return
            print("Downloaded intel-opencl-icd-dbgsym_25.13.33276.16_amd64.ddeb")
            if self.rcwec("curl -OL https://github.com/intel/intel-graphics-compiler/releases/download/v2.10.8/intel-igc-core-2_2.10.8+18926_amd64.deb", False): return
            print("Downloaded intel-igc-core-2_2.10.8+18926_amd64.deb")
            if self.rcwec("curl -OL https://github.com/intel/compute-runtime/releases/download/25.13.33276.16/intel-level-zero-gpu_1.6.33276.16_amd64.deb", False): return
            print("Downloaded intel-level-zero-gpu_1.6.33276.16_amd64.deb")
            if self.rcwec("curl -OL https://github.com/intel/compute-runtime/releases/download/25.13.33276.16/intel-opencl-icd_25.13.33276.16_amd64.deb", False): return
            print("Downloaded intel-opencl-icd_25.13.33276.16_amd64.deb")
            if self.rcwec("curl -OL https://github.com/intel/compute-runtime/releases/download/25.13.33276.16/libigdgmm12_22.7.0_amd64.deb", False): return
            print("Downloaded libigdgmm12_22.7.0_amd64.deb")

            if self.rcwec("sudo dpkg -i *deb"): return

            if self.rcwec("rm intel-igc-core-2_2.10.8+18926_amd64.deb"): return
            if self.rcwec("rm intel-igc-opencl-2_2.10.8+18926_amd64.deb"): return
            if self.rcwec("rm intel-level-zero-gpu-dbgsym_1.6.33276.16_amd64.ddeb"): return
            if self.rcwec("rm intel-level-zero-gpu_1.6.33276.16_amd64.deb"): return
            if self.rcwec("rm intel-opencl-icd-dbgsym_25.13.33276.16_amd64.ddeb"): return
            if self.rcwec("rm intel-opencl-icd_25.13.33276.16_amd64.deb"): return
            if self.rcwec("rm libigdgmm12_22.7.0_amd64.deb"): return

        if pkg_mgr == "dnf":

            print(self.run_command("sudo dnf install -y intel-compute-runtime"))

        if pkg_mgr == "pacman":
            output = self.run_command("pacman -Q | grep intel-compute-runtime")
            if output:
                print("Intel compute runtime is already installed on this system")
            else:
                print(self.run_command("sudo pacman -S --noconfirm intel-compute-runtime"))

    def check_for_intel_npu(self, pkg_mgr):
        if not pkg_mgr == "apt":
            return

        print("Checking for Intel NPU ...")
        output = self.run_command("lscpu | grep 'Model name:'")
        print(f"CPU: {output.split(':')[1].strip()}")
        if not "Intel(R) Core(TM) Ultra" in output:
            print("Did not find Intel NPU")
            return

        response = input("Would you like to install optional drivers for Intel NPU? N/y ")
        if not response:
            return
        if response.lower() != "y":
            return

        if self.rcwec("wget https://github.com/oneapi-src/level-zero/releases/download/v1.20.2/level-zero_1.20.2+u22.04_amd64.deb"): return
        if self.rcwec("wget https://github.com/intel/linux-npu-driver/releases/download/v1.16.0/intel-level-zero-npu_1.16.0.20250328-14132024782_ubuntu24.04_amd64.deb"): return
        if self.rcwec("wget https://github.com/intel/linux-npu-driver/releases/download/v1.16.0/intel-fw-npu_1.16.0.20250328-14132024782_ubuntu24.04_amd64.deb"): return
        if self.rcwec("wget https://github.com/intel/linux-npu-driver/releases/download/v1.16.0/intel-driver-compiler-npu_1.16.0.20250328-14132024782_ubuntu24.04_amd64.deb"): return

        if self.rcwec("sudo apt install -y libtbb12"): return
        if self.rcwec("sudo dpkg -i *.deb"): return

        if self.rcwec("sudo chown root:render /dev/accel/accel0"): return
        if self.rcwec("sudo chmod g+rw /dev/accel/accel0"): return
        if self.rcwec("sudo usermod -a -G render $USER"): return
        if self.rcwec("sudo bash -c \"echo 'SUBSYSTEM==\"accel\", KERNEL==\"accel*\", GROUP=\"render\", MODE=\"0660\"' > /etc/udev/rules.d/10-intel-vpu.rules\""): return
        if self.rcwec("sudo udevadm control --reload-rules"): return
        if self.rcwec("sudo udevadm trigger --subsystem-match=accel"): return

        self.rcwec("rm level-zero_1.20.2+u22.04_amd64.deb")
        self.rcwec("rm intel-level-zero-npu_1.16.0.20250328-14132024782_ubuntu24.04_amd64.deb")
        self.rcwec("rm intel-fw-npu_1.16.0.20250328-14132024782_ubuntu24.04_amd64.deb")
        self.rcwec("rm intel-driver-compiler-npu_1.16.0.20250328-14132024782_ubuntu24.04_amd64.deb")

        if response := input("The system needs to reboot to enable the NPU drivers, would you like to reboot now? N/y  "):
            if response.lower() == "y":
                self.run_command("sudo reboot now")

    def get_host_dist(self):
        result = None
        try:
            with open("/etc/os-release", 'r') as file:
                for line in file:
                    line = line.strip()
                    key, val = line.split("=")
                    if key == "PRETTY_NAME":
                        result = val.strip('"')
                        return result
        except Exception as ex:
            print(f"File read error /etc/os-release: {ex}")
        return result
    
    def collect_user_sudo(self):
        self.run_command('sudo ls -l')

    def install_icon(self):
        print("\n")
        print(self.run_command(f'sudo {self.venv}/bin/onvif-gui --icon'))
        print("\n")

    def uninstall(self):
        try:
            home = os.environ.get('HOME')
            targets = [f"{home}/.config/onvif-gui", f"{home}/.cache/onvif-gui"]
            icon_filename = "/usr/share/applications/onvif-gui.desktop"
            if not os.path.exists(icon_filename):
                raise Exception(f"Did not find installed onvif gui icon at {icon_filename}")

            with open(icon_filename, 'r') as file:
                for line in file:
                    tokens = line.split("=")
                    if tokens[0] == "Exec":
                        vars = tokens[1].split()
                        install_directory = vars[0][:vars[0].find('/bin/onvif-gui')]
                        targets.append(install_directory)
            
            for target in targets:
                if os.path.exists(target):
                    shutil.rmtree(target)

            output = self.run_command(f'sudo rm {icon_filename}')
            if self.status:
                raise Exception(output)

            print("Onvif GUI has been removed from the system")

        except Exception as ex:
            print(f"An error occurred during un-installation: {ex}")

if __name__ == "__main__":

    if not(home := os.environ.get('HOME')):
        sys.exit(1)

    install = Install()

    parser = argparse.ArgumentParser(description="Install Onvif GUI using an automated script")
    parser.add_argument("-u", "--uninstall", action="store_true", help="Uninstall")

    args = parser.parse_args()

    if args.uninstall:
        install.uninstall()
        sys.exit(0)

    install.collect_user_sudo()

    print(f"Operating System of host: {install.get_host_dist()}")
    
    if not(pkg_mgr := install.find_package_manager()):
        sys.exit(1)

    if install.check_python(pkg_mgr):
        sys.exit(1)

    venv = f"{os.environ.get('HOME')}/.local/share/onvif-gui-env"
    if install.create_venv(venv, pkg_mgr):
        sys.exit(1)
    
    if INSTALL_SOURCE == "pypi":
        if install.install_onvif_gui(venv):
            sys.exit(1)
    elif INSTALL_SOURCE == "smb":
        if install.install_onvif_gui_local(venv, pkg_mgr):
            sys.exit(1)

    install.check_display_protocol(pkg_mgr)
    install.check_for_nvidia()
    install.check_for_intel_gpu(pkg_mgr)

    install.install_icon()

    install.check_for_intel_npu(pkg_mgr)
