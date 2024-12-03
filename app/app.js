const express = require("express");
const bodyParser = require("body-parser");
const { spawn } = require("child_process");
const path = require("path");

const app = express();
const port = 3000;

// Middleware to parse JSON request bodies
app.use(bodyParser.json());

// Initialize the Python process at server startup
const pythonPath = path.join(__dirname, "../predict.py");
const pythonProcess = spawn("python3", [pythonPath], {
    cwd: path.join(__dirname, ".."),
});

// Ensure the Python process starts successfully
pythonProcess.stderr.on("data", (data) => {
    console.error(`Python error: ${data}`);
});

pythonProcess.on("close", (code) => {
    console.error(`Python process exited with code ${code}`);
});

// Route for predictions
app.post("/predict", (req, res) => {
    const text = req.body.text;

    if (!text) {
        return res.status(400).json({ error: "No text provided" });
    }

    // Send JSON input to Python process
    const input = JSON.stringify({ text });
    pythonProcess.stdin.write(`${input}\n`);

    let output = "";

    // Collect output from Python process
    pythonProcess.stdout.once("data", (data) => {
        output += data.toString().trim();

        try {
            const result = JSON.parse(output);
            res.json(result);
        } catch (error) {
            console.error(`Error parsing JSON: ${output}`);
            res.status(500).json({ error: "Invalid JSON from Python" });
        }
    });
});

app.get("/",function(req,res){
    res.sendFile(path.join(__dirname, "../html/index.html"));
})

// Serve static files
app.use(express.static(path.join(__dirname, "html")));

// Start the Node.js server
app.listen(port, () => {
    console.log(`Node.js server running on http://localhost:${port}`);
});
