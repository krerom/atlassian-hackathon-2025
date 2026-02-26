const express = require('express');
const cors = require('cors');
const https = require('https');
const fs = require('fs');
const mysql = require('mysql2/promise');
const { spawn } = require('child_process');
require('dotenv').config();
const csv = require('csv-parser');

const SSL_KEY = "";
const SSL_CERT = "";

const app = express();
const port = 443;

const dummySprintsPath = "/root/sprintsense/synth_data/dummy_sprint_features.csv";

app.use(cors({
    origin: "*",
    methods: ['GET', 'POST', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
    credentials: true
}));

const db = mysql.createPool({
    connectionLimit: 10,
    host: process.env.DB_HOST,
    user: process.env.DB_USER,
    password: process.env.DB_PASS,
    database: process.env.DB_NAME,
    queueLimit: 0,
    waitForConnections: true
});

app.use(express.json());

app.post('/query/db', async (req, res) => {
    const { query_string, values_array } = req.body;
    console.log("querying db: ", query_string, values_array);

    try {
        const [rows] = await db.query(
            query_string,
            values_array
        );

        if (rows.length > 0) {
            return res.status(200).json({ success: true, data: rows });
        } else{
            return res.status(404).json({success: false, data: null});
        }

    } catch (error) {
        console.log(error);
        return res.status(500).json({ success: false, data: [] });
    }
});

app.post('/train/adapter', async (req, res) => {
    const { auth_token, userDensePath, featureRows } = req.body;

    const csvPath = `/root/sprintsense/lstm/users/dense_${auth_token}.csv`;

    const csvContent = [
        Object.keys(featureRows[0]).join(","),
        ...featureRows.map(row => Object.values(row).join(","))
    ].join("\n");

    fs.writeFileSync(csvPath, csvContent);

    try {
        await new Promise((resolve, reject) => {
            const py = spawn("/usr/bin/python3", [
                "/root/sprintsense/lstm/trainUserDenseLayers.py",
                "--csv",
                csvPath,
                "--output",
                userDensePath
            ]);

            py.stdout.on("data", data => console.log(`[PYTHON]: ${data.toString()}`));
            py.stderr.on("data", data => console.error(`[PYTHON ERROR]: ${data.toString()}`));

            py.on('error', (err) => {
                console.error(`[PYTHON PROCESS ERROR]: ${err}`);
                reject(new Error(`Failed to start Python process: ${err.message}`));
            });

            py.on("close", code => {
                if (code === 0) resolve();
                else reject(new Error("Python training failed"));
            });
        });
    } catch (error) {
        console.log(error);
        return res.status(500).json({ success: false, message: error.message });
    }
    return res.status(200).json({ success: true });
});

app.post('/generate/lstm/simulation', async (req, res) => {
    const { auth_token, csvContent, adapterPath } = req.body;

    if (!fs.existsSync(adapterPath)) {
        return res.status(404).json({ success: false, message: "Please train an adapter first." });
    }

    const csvPath = `/root/sprintsense/lstm/users/${auth_token}.csv`;
    fs.writeFileSync(csvPath, csvContent);

    const userNextSprintPath = `/root/sprintsense/lstm/users/${auth_token}_sim.csv`;


    let predictionJson = '';
    try {
        await new Promise((resolve, reject) => {
            const py = spawn("/usr/bin/python3", [
                "/root/sprintsense/lstm/combinedModelInference.py",
                "--csv",
                csvPath,
                "--output",
                userNextSprintPath,
                "--adapter",
                adapterPath
            ]);

            py.stdout.on("data", data => {
                console.log(`[PYTHON]: ${data.toString()}`);
                predictionJson += data.toString();
            });

            py.stderr.on("data", data => {
                console.error(`[PYTHON STDERR CRASH]: ${data.toString()}`);
            });

            py.on('error', (err) => {
                console.error(`[PYTHON PROCESS ERROR]: ${err}`);
                reject(new Error(`Failed to start Python process: ${err.message}`));
            });

            py.on("close", code => {
                if (code === 0) {
                    resolve();
                }
                else reject(new Error("Python LSTM prediction failed"));
            });
        });

        let predictionData;
        try {
            predictionData = JSON.parse(predictionJson); // predictionData shall be returned by endpoint
            console.log(predictionData);
        } catch (e) {
            console.error("Failed to parse Python JSON output: ", predictionJson);
            throw new Error("Prediction script returned invalid data.");
        }
        return res.status(200).json({ success: true, predictionData });
    } catch (error) {
        return res.status(200).json({ success: false, message: error });
    }
});

app.post('/generate/lstm/prediction', async (req, res) => {
    const { auth_token, csvContent, adapterPath } = req.body;

    if (!fs.existsSync(adapterPath)) {
        return res.status(404).json({ success: false, message: "Please train an adapter first." });
    }

    const csvPath = `/root/sprintsense/lstm/users/${auth_token}.csv`;
    const userNextSprintPath = `/root/sprintsense/lstm/users/${auth_token}.csv`;

    fs.writeFileSync(csvPath, csvContent);

    let predictionJson = '';
    try {

        await new Promise((resolve, reject) => {
            const py = spawn("/usr/bin/python3", [
                "/root/sprintsense/lstm/combinedModelInference.py",
                "--csv",
                csvPath,
                "--output",
                userNextSprintPath,
                "--adapter",
                adapterPath
            ]);

            py.stdout.on("data", data => {
                console.log(`[PYTHON]: ${data.toString()}`);
                predictionJson += data.toString();
            });

            py.stderr.on("data", data => {
                console.error(`[PYTHON STDERR CRASH]: ${data.toString()}`);
            });

            py.on('error', (err) => {
                console.error(`[PYTHON PROCESS ERROR]: ${err}`);
                reject(new Error(`Failed to start Python process: ${err.message}`));
            });

            py.on("close", code => {
                if (code === 0) {
                    resolve();
                }
                else reject(new Error("Python LSTM prediction failed"));
            });
        });

        let predictionData;
        try {
            predictionData = JSON.parse(predictionJson);
            return res.status(200).json({ success: true, predictionData });
        } catch (e) {
            console.error("Failed to parse Python JSON output: ", predictionJson);
            return res.status(500).json({ success: false, message: e });
        }
    } catch (error) {
        return res.status(500).json({ success: false, message: error });
    }
});

app.get('/sprints/dummy', async (req, res) => {
    const results = [];

    if (!fs.existsSync(dummySprintsPath)) {
        return res.status(404).send({ error: 'CSV file not found.' });
    }

    try {
        fs.createReadStream(dummySprintsPath)
            .pipe(csv())
            .on('data', (data) => results.push(data))
            .on('end', () => {
                console.log(`Successfully parsed ${results.length} rows.`);
                return res.json(results);
            })
            .on('error', (err) => {
                console.error("CSV Parsing Error:", err);
                return res.status(500).send({ error: 'Error processing CSV file.' });
            });

    } catch (error) {
        console.error("File System Error:", error);
        return res.status(500).send({ error: 'Internal server error.' });
    }
});

const options = {
    key: fs.readFileSync(SSL_KEY),
    cert: fs.readFileSync(SSL_CERT),
};

https.createServer(options, app).listen(port, () => {
    console.log(`HTTPS Server is running on Port: ${port}`);
});
