console.log("Script loaded");

// Define the API base URL dynamically based on the window location
const API_BASE_URL = window.location.hostname === 'localhost' ? 'http://localhost:8181' : 'https://www.animalclassificationucf.com';


document.getElementById('animalForm').addEventListener('submit', function(event) {
    event.preventDefault();
    console.log("Form submission triggered");

    const fileInput = document.getElementById('imageUpload');
    const file = fileInput.files[0]; // Get the file from the input
    const predictionResult = document.getElementById('predictionResult');
    const uploadedImage = document.getElementById('uploadedImage');
    const loading = document.getElementById('loading');

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

    fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
    })

    .then(response => response.json())  // Parse JSON response
    .then(data => {
        console.log("Prediction received:", data.prediction);
        predictionResult.textContent = data.prediction;  // Display just the prediction string
        uploadedImage.src = data.imageUrl;  // Update the image source if you're returning it from the server
        loading.style.display = 'none'; // Hide the loading indicator
    })
    .catch(error => {
        console.error('Error during fetch:', error);
        predictionResult.textContent = 'Failed to process the image. Please try again.';
        loading.style.display = 'none'; // Hide the loading indicator
    });
});