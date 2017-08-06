// This file is required by the index.html file and will
// be executed in the renderer process for that window.
// All of the Node.js APIs are available in this process.

var keras = require('keras-js');
var model = new keras.Model({
    filepaths: {
        model: 'models/simple_model/model.json',
        weights: 'models/simple_model/model_weights.buf',
        metadata: 'models/simple_model/model_metadata.json'
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

function switchModel() {
    if(document.getElementById("simple").checked) {
        model = new keras.Model({
            filepaths: {
                model: 'models/simple_model/model.json',
                weights: 'models/simple_model/model_weights.buf',
                metadata: 'models/simple_model/model_metadata.json'
            },
            gpu: true
        });
    }
    if(document.getElementById("simple2").checked) {
        model = new keras.Model({
            filepaths: {
                model: 'models/simple2_model/model.json',
                weights: 'models/simple2_model/model_weights.buf',
                metadata: 'models/simple2_model/model_metadata.json'
            },
            gpu: true
        });
    }
    if(document.getElementById("pretrained").checked) {
        model = new keras.Model({
            filepaths: {
                model: 'https://transcranial.github.io/keras-js-demos-data/mnist_cnn/mnist_cnn.json',
                weights: 'https://transcranial.github.io/keras-js-demos-data/mnist_cnn/mnist_cnn_weights.buf',
                metadata: 'https://transcranial.github.io/keras-js-demos-data/mnist_cnn/mnist_cnn_metadata.json'
            },
            gpu: true
        });
    }
}

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

function findMaxIndex(data) {
    var index = -1;
    var value = -Infinity;
    for(i = 0; i < data.length; i++) {
        if(value < data[i]) {
            index = i;
            value = data[i];
        }
    }
    return index;
}

async function predictNumber() {
    var inputData = {
        'input': new Float32Array(28*28)
    };
    var imageData = Float32Array.from(contextSmall.getImageData(0,0,28,28).data);
    for (i = 0, len = imageData.length; i < len; i += 4) {
        inputData.input[i / 4] = imageData[i + 3] / 255
    }
    var outputData = await model.predict(inputData); 
    var predicted = findMaxIndex(outputData.output);
    document.getElementById('predicted').innerHTML = predicted;
}

document.getElementById('refresh-button').addEventListener('click', refresh);
document.getElementById('fullSizeCanvas').addEventListener('mousedown', startDraw);
document.getElementById('fullSizeCanvas').addEventListener('mousemove', draw);
document.getElementById('fullSizeCanvas').addEventListener('mouseup', stopDraw);
document.getElementById('simple').addEventListener('change', switchModel);
document.getElementById('simple2').addEventListener('change', switchModel);
document.getElementById('pretrained').addEventListener('change', switchModel);