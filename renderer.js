// This file is required by the index.html file and will
// be executed in the renderer process for that window.
// All of the Node.js APIs are available in this process.

var keras = require('keras-js');
var model = new keras.Model({
    filepaths: {
        model: 'model/model.json',
        weights: 'model/model_weights.buf',
        metadata: 'model/model_metadata.json'
    },
    gpu: true
});

var contextSmall = document.getElementById('smallSizeCanvas').getContext('2d');
var contextFull = document.getElementById('fullSizeCanvas').getContext('2d');
contextFull.lineWidth = 21;
contextFull.lineJoin = 'round';
contextFull.lineCap = 'round';
contextFull.strokeStyle = '#000000';

var x = 0;
var y = 0;

function refresh() {
	contextFull.clearRect(0,0,contextFull.canvas.width, contextFull.canvas.height);
    contextSmall.clearRect(0,0,contextSmall.canvas.width, contextSmall.canvas.height);
}

function startDraw() {
	contextFull.beginPath();
	contextFull.moveTo(x,y);
	contextFull.canvas.addEventListener('mousemove', paint);
}

function stopDraw() {
	contextFull.canvas.removeEventListener('mousemove', paint);
	contextSmall.drawImage(contextFull.canvas,0,0,28,28);
	predictNumber();
}

function draw(e) {
	x = e.pageX - contextFull.canvas.offsetLeft;
	y = e.pageY - contextFull.canvas.offsetTop;
}

function paint() {
	contextFull.lineTo(x,y);
	contextFull.stroke();
}

async function predictNumber() {
    var inputData = {
        'input': new Float32Array(28*28)
    };
    var imageData = Float32Array.from(contextSmall.getImageData(0,0,28,28).data);
    for (let i = 0, len = imageData.length; i < len; i += 4) {
        inputData.input[i / 4] = imageData[i + 3] / 255
    }
    var outputData = await model.predict(inputData);
    console.log(outputData.output)
}

document.getElementById('refresh-button').addEventListener('click', refresh);
document.getElementById('fullSizeCanvas').addEventListener('mousedown', startDraw);
document.getElementById('fullSizeCanvas').addEventListener('mousemove', draw);
document.getElementById('fullSizeCanvas').addEventListener('mouseup', stopDraw);