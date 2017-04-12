import sys
from colors import bcolors

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)


    if count==total:
        sys.stdout.write('[%s] %s%s %s\r' % (bar, bcolors.OKGREEN+str(percents), '%'+bcolors.ENDC,suffix))
        sys.stdout.flush()
        print ''
    else:
        sys.stdout.write('[%s] %s%s %s\r' % (bar, percents, '%', suffix))
        sys.stdout.flush()
