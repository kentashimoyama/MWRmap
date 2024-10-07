from colorama import Fore, Style


def errInfo(info: str):
	return Fore.MAGENTA + info + Style.RESET_ALL


def errPrint(info: str):
	print(errInfo(info))


def sysInfo(info: str):
	return Fore.CYAN + info + Style.RESET_ALL


def sysPrint(info: str):
	print(errInfo(info))


def colInfo(info: str, cn: str):
	"""
	:param info:
	:param cn: terminal basic colors
	:return:
	"""
	style = {
		"BLACK": Fore.BLACK,
		"RED": Fore.RED,
		"GREEN": Fore.GREEN,
		"YELLOW": Fore.YELLOW,
		"BLUE": Fore.BLUE,
		"MAGENTA": Fore.MAGENTA,
		"CYAN": Fore.CYAN,
		"WHITE": Fore.WHITE,
	}
	return style[cn.upper()] + info + Style.RESET_ALL


def colPrint(info: str, cn: str):
	print(colInfo(info, cn))


def err_exit(info: str, err_code: int = 1):
	errPrint(info)
	exit(err_code)


def printLog(head: str, info: str):
	from MMSProbe.utils.Common.debug_manager import sys_log_path
	import time
	fa = open(sys_log_path, "a")
	t_str = time.strftime("%m-%d %H:%M:%S", time.localtime())
	fa.write(t_str + " >> " + head + ": " + info + "\n")
	fa.close()
	print(Fore.YELLOW + t_str + " >> " + Fore.MAGENTA + head + ": " + Style.RESET_ALL + info)