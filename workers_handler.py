from flask import Flask, jsonify


app = Flask('app')

channel = 0
repo = 0

@app.route('/')
def hello_world():
    global channel, repo
    return f"""<h1>Colab workers ready to run! </h1>
    <h1>repo number: {repo}</h1>
    <h1>channel number: {channel}</h1>"""

@app.route('/channel', methods=['GET'])
def channel_n():
  global channel
  try:
    with open("channels.txt", "r") as f:
        url = f.readline(channel)
        channel += 1
    return jsonify({'url': url}), 200
  except:
    return jsonify({'error': 'channels end'}), 404


@app.route('/repo', methods=['GET'])
def repo_n():
    global repo
    repo += 1
    return jsonify({'peers_data': repo-1}), 200


if __name__ == "__main__":
  app.run(host='0.0.0.0', port=8080)
