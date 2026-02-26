import React, {useState, useEffect} from "react";
import styles from "./LLMprovider.module.css";
import { invoke } from "@forge/bridge";

function LLMprovider({userData}){
    const [provider, setProvider] = useState("openai");
    const [apiKey, setApiKey] = useState(null);
    const [message, setMessage] = useState("");

    let currentApiKey;
    useEffect(()=>{
        if (userData.api_key){
            currentApiKey = userData?.api_key;
        }
    },[]);

    const providers = [
        { value: "openai", label: "OpenAI" },
        { value: "anthropic", label: "Anthropic (coming soon)", disabled: true },
        { value: "gemini", label: "Google Gemini (coming soon)", disabled: true },
        { value: "groq", label: "Groq (coming soon)", disabled: true }
    ];

    const saveSettings = async () => {
        try {
            await invoke("saveUserSettings", { provider, api_key: apiKey, auth_token: localStorage.getItem('auth_token') });
            setMessage("Settings saved!");
        } catch (err) {
            console.error(err);
            setMessage("Error saving settings");
        }
    };

    return (
        <div className={styles.container}>
            <h2>LLM Provider Settings</h2>

            <label>Choose LLM Provider</label>
            <select
                className={styles.select}
                value={provider}
                onChange={(e) => setProvider(e.target.value)}
            >
                {providers.map((p) => (
                    <option key={p.value} value={p.value} disabled={p.disabled}>
                        {p.label}
                    </option>
                ))}
            </select>

            <label>API Key</label>
            <input
                type="password"
                className={styles.input}
                placeholder="Enter your API key"
                value={apiKey || currentApiKey}
                onChange={(e) => setApiKey(e.target.value)}
            />

            <button className={styles.saveBtn} onClick={saveSettings}>
                Save
            </button>

            {message && <p>{message}</p>}

            {/*<div className={styles.infoContainer}>
                <span className={styles.infoIcon}>ℹ️</span>
                <p className={styles.infoText}>
                    Your API key is stored securely. First, a public RSA key is fetched from the server. 
                    The API key is encrypted locally in your browser and stored on our server in encrypted form. 
                    When needed for inference, it is decrypted at runtime and sent over a secure connection to the API provider.
                </p>
            </div>*/}

        </div>
    );
}

export default LLMprovider;