import os
from PIL import Image
from pickle import TRUE
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
# 9 catagory + other
# https://linasm.sourceforge.net/docs/syscalls/index.php
sys_file = ['close', 'creat', 'open', 'openat', 'name_to_handle_at', 'open_by_handle_at', 'memfd_create', 'mknod', 'mknodat', 'rename', 'renameat', 'renameat', 'truncate', 'ftruncate', 'fallocate', 'mkdir', 'mkdirat', 'rmdir', 'getcwd', 'chdir', 'fchdir', 'chroot', 'getdents', 'getdents64', 'lookup_dcookie', 'link', 'linkat', 'symlink', 'symlinkat', 'unlink', 'unlinkat', 'readlink', 'readlinkat', 'umask', 'stat', 'lstat', 'fstat', 'fstatat', 'chmod', 'fchmod', 'fchmodat', 'chown', 'lchown', 'fchown', 'fchownat', 'utime', 'utimes', 'futimesat', 'utimensat', 'access', 'faccessat', 'setxattr', 'lsetxattr', 'fsetxattr', 'getxattr', 'lgetxattr', 'fgetxattr', 'listxattr', 'llistxattr', 'flistxattr', 'removexattr', 'lremovexattr', 'fremovexattr', 'ioctl', 'fcntl', 'dup', 'dup2', 'dup3', 'flock', 'read', 'readv', 'pread', 'preadv', 'write',
            'writev', 'pwrite', 'pwritev', 'lseek', 'sendfile', 'fdatasync', 'fsync', 'msync', 'sync_file_range', 'sync', 'syncfs', 'io_setup', 'io_destroy', 'io_submit', 'io_cancel', 'io_getevents', 'select', 'pselect6', 'poll', 'ppoll', 'epoll_create', 'epoll_create1', 'epoll_ctl', 'epoll_wait', 'epoll_pwait', 'inotify_init', 'inotify_init1', 'inotify_add_watch', 'inotify_rm_watch', 'fanotify_init', 'fanotify_mark', 'fadvise64', 'readahead', 'getrandom']
sys_network = ['socket', 'socketpair', 'setsockopt', 'getsockopt', 'getsockname', 'getpeername', 'bind', 'listen', 'accept', 'accept4',
               'connect', 'shutdown', 'recvfrom', 'recvmsg', 'recvmmsg', 'sendto', 'sendmsg', 'sendmmsg', 'sethostname', 'setdomainname', 'bpf']
sys_time = ['time', 'settimeofday', 'gettimeofday', 'clock_settime', 'clock_gettime', 'clock_getres', 'clock_adjtime', 'clock_nanosleep', 'timer_create', 'timer_delete',
            'timer_settime', 'timer_gettime', 'timer_getoverrun', 'alarm', 'setitimer', 'getitimer', 'timerfd_create', 'timerfd_settime', 'timerfd_gettime', 'adjtimex', 'nanosleep', 'times']
sys_process = ['clone', 'fork', 'vfork', 'execve', 'execveat', 'EXIT', 'exit_group', 'wait4', 'waitid', 'getpid', 'getppid', 'gettid', 'setsid', 'getsid', 'setpgid', 'getpgid', 'getpgrp', 'setuid', 'getuid', 'setgid', 'getgid', 'setresuid', 'getresuid', 'setresgid', 'getresgid', 'setreuid',
               'setregid', 'setfsuid', 'setfsgid', 'geteuid', 'getegid', 'setgroups', 'getgroups', 'setns', 'setrlimit', 'getrlimit', 'prlimit', 'getrusage', 'sched_setattr', 'sched_getattr', 'sched_setscheduler', 'sched_getscheduler', 'sched_setparam', 'sched_getparam', 'sched_setaffinity',
               'sched_getaffinity', 'sched_get_priority_max', 'sched_get_priority_min', 'sched_rr_get_interval', 'sched_yield', 'setpriority', 'getpriority', 'ioprio_set', 'ioprio_get', 'brk', 'mmap', 'munmap', 'mremap', 'mprotect', 'madvise', 'mlock', 'mlock2', 'mlockall', 'munlock', 'munlockall', 'mincore', 'membarrier', 'modify_ldt', 'capset', 'capget', 'set_thread_area', 'get_thread_area', 'set_tid_address', 'arch_prctl', 'uselib', 'prctl', 'seccomp', 'ptrace', 'process_vm_readv', 'process_vm_writev', 'kcmp', 'unshare']
sys_signal = ['kill', 'tkill', 'tgkill', 'pause', 'rt_sigaction', 'rt_sigprocmask', 'rt_sigpending', 'rt_sigqueueinfo', 'rt_tgsigqueueinfo',
              'rt_sigtimedwait', 'rt_sigsuspend', 'rt_sigreturn', 'sigaltstack', 'signalfd', 'signalfd4', 'eventfd', 'eventfd2', 'restart_syscall']
sys_ipc = ['pipe', 'pipe2', 'tee', 'splice', 'vmsplice', 'shmget', 'shmctl', 'shmat', 'shmdt', 'semget', 'semctl', 'semop', 'semtimedop', 'futex', 'set_robust_list',
           'get_robust_list', 'msgget', 'msgctl', 'msgsnd', 'msgrcv', 'mq_open', 'mq_unlink', 'mq_getsetattr', 'mq_timedsend', 'mq_timedreceive', 'mq_notify']
sys_numa = ['getcpu', 'set_mempolicy', 'get_mempolicy',
            'mbind', 'move_pages', 'migrate_pages']
sys_key = ['add_key', 'request_key', 'keyctl']
sys_module = [['create_module', 'init_module', 'finit_module', 'delete_module', 'query_module', 'get_kernel_syms', 'acct', 'quotactl', 'pivot_root', 'swapon', 'swapoff', 'mount', 'umount2',
               'nfsservctl', 'ustat', 'statfs', 'fstatfs', 'sysfs', '_sysctl', 'syslog', 'ioperm', 'iopl', 'personality', 'vhangup', 'reboot', 'kexec_load', 'kexec_file_load', 'perf_event_open', 'uname', 'sysinfo']]
catagories = [sys_file, sys_network, sys_time, sys_process,
              sys_signal, sys_ipc, sys_numa, sys_key, sys_module]
color_map = [
    ['#FFFFFF', '#FFC1E0', '#FFAAD5', '#FF95CA', '#FF79BC ', '#FF60AF',
        '#FF359A', '#FF0080 ', '#F00078 ', '#D9006C', '#BF0060'],
    ['#FFFFFF', '#FFBFFF', '#FFA6FF', '#FF8EFF', '#FF77FF', '#FF44FF',
        '#FF00FF', '#E800E8', '#D200D2', '#AE00AE', '#930093'],
    ['#FFFFFF', '#FFDAC8', '#FFCBB3', '#FFBD9D', '#FFAD86', '#FF9D6F',
        '#FF8F59', '#FF8040', '#FF5809', '#F75000', '#D94600'],
    ['#FFFFFF', '#D3FF93', '#CCFF80', '#B7FF4A', '#A8FF24', '#9AFF02',
        '#8CEA00', '#82D900', '#73BF00', '#64A600', '#548C00'],
    ['#FFFFFF', '#CAFFFF', '#BBFFFF', '#A6FFFF', '#4DFFFF', '#00FFFF',
        '#00E3E3', '#00CACA', '#00AEAE', '#009393', '#005757'],
    ['#FFFFFF', '#C1FFE4', '#ADFEDC', '#96FED1', '#4EFEB3', '#1AFD9C',
        '#02F78E', '#02DF82', '#01B468', '#019858', '#01814A'],
    ['#FFFFFF', '#D6D6AD', '#CDCD9A', '#C2C287', '#B9B973', '#AFAF61',
        '#A5A552', '#949449', '#808040', '#707038', '#616130'],
    ['#FFFFFF', '#DCB5FF', '#D3A4FF', '#CA8EFF', '#BE77FF', '#B15BFF',
        '#9F35FF', '#921AFF', '#8600FF', '#6E00FF', '#5B00AE'],
    ['#FFFFFF', '#FFFF6F', '#FFFF37', '#F9F900', '#E1E100', '#C4C400',
        '#A6A600', '#8C8C00', '#737300', '#5B5B00', '#5B5B00'],
    ['#FFFFFF', '#FFE66F', '#FFE153', '#FFDC35', '#FFD306', '#EAC100',
        '#D9B300', '#C6A300', '#AE8F00', '#977C00', '#796400'],
]

#hex to rgb
def to_rgb(v):
    return np.array([np.int(v[1:3], 16), np.int(v[3:5], 16), np.int(v[5:7], 16)])
# https://stackoverflow.com/questions/66231122/convert-image-saved-in-hexadecimal-in-a-np-array-to-import-it-in-opencv


def categorize(counters):
    image_col = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for key in counters:
        flag = False
        for i, c in enumerate(catagories):
            if key in c:
                image_col[i] += counters[key]
                flag = True
                break
        if flag == False:
            image_col[9] += counters[key]
    return image_col


interval = [1, 3, 7, 12, 18, 25, 33, 42, 100, 200]

# occurrence to color
def match_color(occur):
    rgb_col = []
    for i, num in enumerate(occur):
        for j, val in enumerate(interval):
            if num <= val:
                rgb_col.append(to_rgb(color_map[i][j]))
                break
        if num > 200:
            rgb_col.append(to_rgb(color_map[i][10]))
    return rgb_col

# read label
df = pd.read_csv(
    './dataset.csv', low_memory=False)
df = df[['filename', 'label']]
labelcsv = df[(df['label'] == 'Mirai') | (df['label'] == 'Bashlite') | (df['label'] == 'Unknown')|(df['label'] == 'Tsunami')|(df['label'] == 'Dofloo')|(df['label'] == 'Xorddos')|(df['label'] == 'Hajime')]
label = []
traindata = []
path = './csv_strace_log_linuxmal'
folders = os.listdir(path)
flag = 0
index = 0
pad = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
save_folder = './img_data/all/'
if not os.path.isdir(save_folder):
    os.makedirs(save_folder)
for folder in folders:
    files = os.listdir(os.path.join(path, folder))
    for file in files:
        filename = file.replace('.csv', '')
        match = labelcsv[labelcsv['filename'] == filename]
        if(not match.empty):
            df = pd.read_csv(os.path.join(path, folder, file),
                             engine='python', on_bad_lines='skip')
            start = df.iloc[0]['TIMESTAMP']
            end = df.iloc[-1]['TIMESTAMP']
            unit = (end-start)/16
            time = start+unit
            text = []
            if len(df) > 100:
                label.append(match.iat[0, 1])
                image = []
                count = 0
                for i in df.index:
                    if df['TIMESTAMP'][i] >= time:
                        traindata.append(text)
                        counters = Counter(text)
                        occur = categorize(counters)
                        image_row = match_color(occur)
                        image.append(image_row)
                        time = time+unit
                        count = count+1
                        if count == 16:
                            break
                    systemcall = df['SYSCALL'][i]
                    text.append(systemcall)
                for p in range(16-count):
                    image.append(match_color(pad))
                array = np.array(image, dtype=np.uint8)
                new_image = Image.fromarray(array.transpose(1,0,2))
                save_path = save_folder+filename+'.jpg'
                new_image.save(save_path)
