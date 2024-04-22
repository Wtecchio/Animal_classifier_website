const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const multer = require('multer');
const sanitize = require('sanitize-filename');
const helmet = require('helmet');
const morgan = require('morgan');
const rateLimit = require('express-rate-limit');
const http = require('http');
require('dotenv').config(); // Make sure this line is at the top of your main file


const app = express();
const session = require('express-session');

const logFilePath = './visitor-actions-log.json';

// In-memory log storage
let logs = {};

app.use(session({
  secret: process.env.SESSION_SECRET,  // Ensure you have this in your environment variables
  resave: false,
  saveUninitialized: true,
  cookie: { secure: true } // Recommended if you're serving your site over HTTPS
}));

// Log new sessions
app.use((req, res, next) => {
  if (!req.session.isNew) {
    return next(); // If not a new session, continue without logging
  }

  const now = new Date().toISOString();
  const logEntry = {
    timestamp: now,
    sessionId: req.sessionID,
    userAgent: req.headers['user-agent'],
    action: "Session Start",
    ip: req.ip // Optional, log the IP if you need it
  };

  // Append the log to your log file
  appendLog(req.sessionID, logEntry);

  next();
});

function appendLog(sessionId, logEntry) {
  const logFilePath = './visitor-actions-log.json';
  const data = JSON.stringify({ sessionId, logEntry }) + '\n';
  require('fs').appendFile(logFilePath, data, (err) => {
    if (err) {
      console.error('Failed to append to log:', err);
    }
  });
}


function writeLogsToFile() {
  fs.writeFile(logFilePath, JSON.stringify(logs, null, 2), err => {
      if (err) {
          console.error('Failed to write logs:', err);
      } else {
          console.log("Logs written successfully");
      }
  });
}

// Function to append a single log entry
app.use((req, res, next) => {
  const sessionId = req.sessionID;
  if (!logs[sessionId]) {
      logs[sessionId] = {
          sessionId: sessionId,
          userAgent: req.headers['user-agent'],  // capturing the user-agent at the session level
          interactions: []
      };
  }
  next();
});


// Security enhancements
app.use(helmet());
app.use(morgan('combined'));


const cors = require('cors');
app.use(cors({
    origin: ['https://www.animalclassificationucf.com']
}));

const uploadDirectory = path.join(__dirname, 'public', 'uploads');
fs.mkdirSync(uploadDirectory, { recursive: true });

const limiter = rateLimit({
    windowMs: 15 * 60 * 1000,
    max: 100
});
app.use(limiter);

const pythonScriptPath = path.join(__dirname, 'public', 'predict.py');

// Set up storage options for multer
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
      // Define where to store the files
      cb(null, uploadDirectory);
  },
  filename: (req, file, cb) => {
      // Create a unique filename for the file
      const uniqueSuffix = `${Date.now()}-${Math.round(Math.random() * 1E9)}`;
      const fileExtension = path.extname(file.originalname); // Extract file extension
      const safeFileName = sanitize(file.originalname); // Sanitize the filename to prevent any security issues
      cb(null, `${safeFileName}-${uniqueSuffix}${fileExtension}`); // Construct the full filename and pass it to the callback
  }
});

// Configure multer with the storage options, file size limit, and file filter for images
const upload = multer({
  storage: storage,
  limits: { fileSize: 10 * 1024 * 1024 }, // Set file size limit to 10 MB
  fileFilter: (req, file, cb) => {
      // Set the file filter to only accept images
      if (file.mimetype.startsWith('image/')) {
          cb(null, true);
      } else {
          // If the file is not an image, pass an error to the callback
          cb(new Error('Not an image! Please upload only images.'), false);
      }
  }
});





app.post('/predict', upload.single('imageUpload'), (req, res) => {
  const now = new Date().toISOString();
  const sessionId = req.sessionID;
  const logEntryBase = {
      timestamp: now,
      route: '/predict',
      filename: req.file ? req.file.filename : 'No file uploaded'
  };

  // Ensure there's a log object for the session
  if (!logs[sessionId]) {
      logs[sessionId] = {
          sessionId: sessionId,
          interactions: []
      };
  }

  // Append log entry for attempting the upload
  if (!req.file) {
      const errorLog = { ...logEntryBase, error: 'No file uploaded' };
      logs[sessionId].interactions.push(errorLog);
      writeLogsToFile(); // Update the log file with the new entry
      return res.status(400).send('No file uploaded');
  }

  // Append log entry for file existence check
  if (!fs.existsSync(pythonScriptPath)) {
      const errorLog = { ...logEntryBase, error: "Python script file does not exist" };
      logs[sessionId].interactions.push(errorLog);
      writeLogsToFile(); // Update the log file with the new entry
      return res.status(500).send("Server configuration error");
  }

  const filePath = path.join(uploadDirectory, req.file.filename);
  const pythonProcess = spawn('python', [pythonScriptPath, filePath]);
  let pythonData = "";

  pythonProcess.stdout.on('data', data => {
      pythonData += data.toString();
  });

  pythonProcess.stderr.on('data', data => {
      const errorLog = { ...logEntryBase, error: `Python script execution error: ${data.toString()}` };
      logs[sessionId].interactions.push(errorLog);
  });

  pythonProcess.on('close', code => {
      if (code !== 0) {
          const errorLog = { ...logEntryBase, error: `Python script exited with code ${code}` };
          logs[sessionId].interactions.push(errorLog);
          writeLogsToFile(); // Update the log file with the new entry
          return res.status(500).send('Error processing the image with Python script');
      }

      const successLog = { ...logEntryBase, result: pythonData.trim() };
      logs[sessionId].interactions.push(successLog);
      writeLogsToFile(); // Update the log file with the new entry

      res.status(200).json({
          prediction: pythonData.trim(),
          imageUrl: `/uploads/${req.file.filename}`
      });
  });
});


//console.log(path.join(__dirname,'/key.pem'))

// Hostname checking middleware
app.use((req, res, next) => {
    if (req.hostname !== 'animalclassificationucf.com' && req.hostname !== 'www.animalclassificationucf.com') {
        res.status(404).send('Not found');
    } else {
        next();
    }
});

app.use(express.json()); // For parsing application/json
app.use(express.urlencoded({ extended: true })); // For parsing application/x-www-form-urlencoded
app.use(express.static('public'));

//app.use(express.urlencoded({ extended: true })); // For parsing application/x-www-form-urlencoded

// Define your routes here
app.get('/', (req, res) => {
    res.send('Welcome to Animal Classification UCF!');
});

// Other routes and logic...

// HTTPS server setup
http.createServer(app).listen(80, '0.0.0.0', () => {  // Listen on all interfaces
    console.log(`Server running on http://www.animalclassificationucf.com:80`);
    // Consider logging this to your log file or a system log as well
    const startupLog = {
        timestamp: new Date().toISOString(),
        message: `Server started on http://www.animalclassificationucf.com:80`
    };
    fs.appendFile(logFilePath, JSON.stringify(startupLog) + '\n', err => {
        if (err) console.error('Failed to log server startup:', err);
    });
}); 