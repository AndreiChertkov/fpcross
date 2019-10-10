import time
import platform

from . import config

def ij():
    t = time.strftime('%l:%M%p %Z on %b %d, %Y')
    print('Start | %s | python %-8s |'%(t, platform.python_version()))
    print('-'*55)

    from IPython.core.display import HTML
    return HTML('<style>%s</style>'%config['css'])
