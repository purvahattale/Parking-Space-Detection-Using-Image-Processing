from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)
app.config["CACHE_TYPE"] = "null"


@app.route('/', methods=['GET', 'POST'])
def camera():
    if request.method == 'POST':
        if request.form['submit_button'] == 'Slot Details':
            return redirect(url_for('slots'))
    return render_template('camera.html')


def get_frame():
    camera = cv2.VideoCapture(0)  # this makes a web cam object
    while True:
        ret, img = camera.read()

        imgencode = cv2.imencode('.jpg', img)[1]
        stringData = imgencode.tostring()
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n' + stringData + b'\r\n')


@app.route('/video_stream')
def video_stream():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/slots')
def slots():
    return render_template('slot.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)
