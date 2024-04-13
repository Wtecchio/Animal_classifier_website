const express = require('express');
const app = express();
const PORT = process.env.PORT || 3000; // Use the PORT environment variable if available, otherwise default to 3000

// Define a route for the root URL
app.get('/', (req, res) => {
    res.send('Hello, world! This is your Node.js server.');
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});