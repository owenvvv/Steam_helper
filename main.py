import json

from flask import Flask, request,render_template
from geventwebsocket.handler import WebSocketHandler
from gevent.pywsgi import WSGIServer
import intention


helper_session = []


app = Flask(__name__)


@app.route('/msg')
def msg():
    global helper_session
	# 
    user_socker = request.environ.get('wsgi.websocket')
    # 
    while 1:
    	# 
        msg = user_socker.receive()
        result={}
        result['message']=msg
        print(msg)

        r_text, new_session = intention.response(result, helper_session)
        # If only one sentence return, change it into a list.
        r_text_return=[]
        if not isinstance(r_text, list):
            r_text_return.append(r_text)
        else:
            r_text_return=r_text

        helper_session.extend(new_session)
		# Packed in a dict
        res = {"msg" : r_text_return}
        # Sent to client
        user_socker.send(json.dumps(res))
        
if __name__ == '__main__':	
    http_server = WSGIServer(('127.0.0.1', 5000), app, handler_class=WebSocketHandler)
    # Start Listening:
    http_server.serve_forever()
    