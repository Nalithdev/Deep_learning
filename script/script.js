data_name = {
    "0": "Avion", "1": "Voiture", "2": "Oiseau", "3": "Chat",
    "4": "Cerf", "5": "Chien", "6": "Grenouille", "7": "Cheval",
    "8": "Bateau", "9": "Camion"
};

const imageElement = document.getElementById("image");

let image = null;

document.getElementById("imageInput").addEventListener("change", function (event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.readAsDataURL(file);

    reader.onload = function (e) {
        imageElement.src = e.target.result;
    };
});
// Predict button
document.getElementById("predictButton").addEventListener("click", async () => {
    // Resize the canvas drawing to 28x28 and normalize
    const resizedImage = getResizedImage(image, 32, 32);
    const normalizedImage = normalizeImage(resizedImage);

    // Load the ONNX model and make a prediction
    try {
        const modelUrl = "model.onnx"; // Path to your ONNX model
        const session = await ort.InferenceSession.create(modelUrl);

        console.log(normalizedImage);
        const inputTensor = new ort.Tensor(
            "float32",
            normalizedImage,
            [1, 3, 32, 32]
        );
        console.log(inputTensor);
        const feeds = { input: inputTensor }; // Replace "input" with the actual input name
        const results = await session.run(feeds);

        const output = results[Object.keys(results)[0]].data;
        const predictedDigit = output.indexOf(Math.max(...output));
        console.log(predictedDigit);
        let name = data_name[predictedDigit.toString()];
        document.getElementById(
            "output"
        ).textContent = `Prediction: ${name}`;
    } catch (error) {
        console.error("Error during inference:", error);
        document.getElementById("output").textContent =
            "Error during prediction. Check the console for details.";
    }
});

// Helper functions
function getResizedImage(canvas, width, height) {
    const offScreenCanvas = document.createElement("canvas");
    offScreenCanvas.width = width;
    offScreenCanvas.height = height;
    const offScreenCtx = offScreenCanvas.getContext("2d");

    return offScreenCtx.getImageData(0, 0, width, height);
}

function normalizeImage(imageData) {
    const { data, width, height } = imageData;
    const normalized = new Float32Array(3 * width * height);

    // Moyenne et écart-type pour normalisation (ex: ImageNet)
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    let index = 0;
    for (let i = 0; i < data.length; i += 4) {
        // R, G, B canaux
        const r = data[i] / 255.0;
        const g = data[i + 1] / 255.0;
        const b = data[i + 2] / 255.0;

        // Normalisation par canal
        normalized[index] = (r - mean[0]) / std[0]; // Red
        normalized[index + width * height] = (g - mean[1]) / std[1]; // Green
        normalized[index + 2 * width * height] = (b - mean[2]) / std[2]; // Blue

        index++;
    }

    return normalized;
}

