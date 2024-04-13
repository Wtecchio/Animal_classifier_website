console.log("Script loaded");

document.getElementById('animalForm').addEventListener('submit', function(event) {
    event.preventDefault();
    console.log("Form submission triggered");

    const fileInput = document.getElementById('imageUpload');
    const file = fileInput.files[0];
    const predictionResult = document.getElementById('predictionResult');
    const uploadedImage = document.getElementById('uploadedImage');

    if (!file) {
        console.log("No file selected");
        alert('Please select a file to upload.');
        return;
    }
    console.log("File selected:", file.name);

    // Display the image
    uploadedImage.style.display = 'block';
    uploadedImage.src = URL.createObjectURL(file);

    const formData = new FormData();
    formData.append('imageUpload', file);

    predictionResult.textContent = 'Processing...';
    console.log("Sending request to server");

    fetch('http://localhost:3000/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())  // Parse JSON response
    .then(data => {
        console.log("Prediction received:", data.prediction);
        predictionResult.textContent = data.prediction;  // Display just the prediction string
        // If you also want to display the image based on the imageUrl
        // uploadedImage.src = data.imageUrl;
    })
    .catch(error => {
        console.error('Error during fetch:', error);
        predictionResult.textContent = 'Failed to process the image. Please try again.';
    });
});