import time

from . import config

def ij():
    t = time.strftime('%l:%M%p %Z on %b %d, %Y')
    print('Start | %s |'%t)
    print('-'*37)

    from IPython.core.display import HTML
    return HTML('<style>%s</style>'%config['css'])
