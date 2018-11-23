import sys
from sanic import Sanic
from sanic.response import json, text
from sanic.config import Config
from server import Server
import server

def main():
  app = Sanic(__name__)
  Config.KEEP_ALIVE = False

  server = Server()

  @app.route('/')
  def test(request):
      return text(server.server_running())

  @app.route('/tfrecord', methods=["POST"])
  def vggish(request):
      return(server.get_tfrecord())

  @app.route('/inference', methods=["POST"])
  def post_json(request):
      return json(server.get_inference())

  app.run(host= '0.0.0.0', port=80)
  print('exiting...')
  sys.exit(0)

if __name__ == '__main__':
  main()
