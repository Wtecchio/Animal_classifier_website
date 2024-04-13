const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const multer = require('multer');

const app = express();
const PORT = process.env.PORT || 3000;
const uploadDirectory = path.join(__dirname, 'public', 'uploads');

// Ensure upload directory exists
fs.mkdirSync(uploadDirectory, { recursive: true });

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, uploadDirectory);
    },
    filename: (req, file, cb) => {
        const uniqueSuffix = `${Date.now()}-${Math.round(Math.random() * 1E9)}`;
        cb(null, `${file.fieldname}-${uniqueSuffix}`);
    }
});

const upload = multer({ storage: storage, limits: { fileSize: 10 * 1024 * 1024 } });

app.use(express.urlencoded({ extended: true }));
app.use(express.static('public'));

app.post('/predict', upload.single('imageUpload'), (req, res) => {
    console.log("Received prediction request");
    if (!req.file) {
        console.log("No file uploaded");
        return res.status(400).send('No file uploaded.');
    }

    console.log("Uploaded file:", req.file.filename);
    const filePath = path.join(uploadDirectory, req.file.filename);
    const relativeImagePath = `/uploads/${req.file.filename}`;

    const pythonScriptPath = path.join(__dirname, 'public', 'predict.py');
    const modelPath = path.join(__dirname, 'public', 'model.pth');

    fs.rename(req.file.path, filePath, (err) => {
        if (err) {
            console.error('Error moving file:', err);
            return res.status(500).send('Error moving file.');
        }

        console.log("File moved to:", filePath);
        const pythonProcess = spawn('python', [pythonScriptPath, filePath, modelPath]);

        pythonProcess.stdout.on('data', (data) => {
            const prediction = data.toString().trim();
            console.log("Python script output:", prediction);
            res.json({ prediction, imageUrl: relativeImagePath });  // Send both prediction and image URL
        });

        pythonProcess.stderr.on('data', (data) => {
            console.error("Python script error:", data.toString());
        });

        pythonProcess.on('close', (code) => {
            console.log(`Python script exited with code ${code}`);
            if (code !== 0) {
                res.status(500).send('Error processing the image.');
            }
        });
    });
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});