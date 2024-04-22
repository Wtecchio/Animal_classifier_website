# Animal Classification Web App

Hey there! Welcome to my Animal Classification Web App. It's a pretty cool project that lets you upload pictures of animals and uses machine learning to predict what kind of animal it is.

## Live Deployment

I've deployed this project live for you to check out without any hassle! You can access it here: www.animalclassificationucf.com

## Features

* Easy-to-use web interface for uploading animal images
* Predicts the animal species in real-time using a pre-trained model
* Secure file upload and processing (don't worry, your images are safe with me!)
* Logs visitor actions and interactions (just to keep track of things)
* Looks great on any device (responsive design for the win!)

## Tech Stack

* Bun.js for the server-side magic
* Python for running the machine learning model
* HTML, CSS, and JavaScript for the client-side awesomeness
* PyTorch for loading and using the pre-trained model
* Multer for handling file uploads
* Morgan and Winston for logging
* And a bunch of other dependencies (check out the package.json file)

## How to Get Started

1. Clone this repo:
```bash
git clone https://github.com/your-username/animal-classification.git
```

2. Install the dependencies:
```bash
cd animal-classification
bun install
```

3. Set up the environment variables:
	* Create a .env file in the root directory.
	* Add the necessary environment variables in the .env file (like SESSION_SECRET)

4. Start the server:
```bash
bun app.js
```

5.Open up the web app in your browser:
```bash
http://localhost:3000
```

## How to Use It

1. Choose an animal image file to upload.
2. Hit the "Predict" button and let the magic happen.
3. Wait for the server to do its thing and give you the predicted animal species.
4. Check out the prediction result and the processed image on the web page.

## Model Training

The animal classification model used in this project is pre-trained and stored in the model.pth file. If you want to train your own model, here's what you can do:

1. Get a dataset of animal images with labels for each species.
2. Use a deep learning framework like PyTorch to define and train a classification model.
3. Save the trained model as model.pth in the project directory.
4. Update the predict.py script to load and use your awesome new model.

## Contributing

If you want to contribute to this project, that's awesome! Feel free to open an issue or submit a pull request if you have any ideas or find any bugs.

## Contact

If you have any questions or just want to chat, feel free to reach out to me at Wtecchio1@gmail.com

Happy classifying!
