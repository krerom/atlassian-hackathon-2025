import React, { useState, useEffect } from "react";
import styles from "./SettingsDisplay.module.css";
import LLMprovider from "./SettingsPages/LLMprovider/LLMprovider";
import LSTMsettings from "./SettingsPages/LSTMsettings/LSTMsettings";
import { useTour } from "../../../../utils/TourProvider";

function SettingsDisplay({ userData, sprints, setshowWelcome, showWelcome, handleTutorialRestart }) {
    const { startTour } = useTour();
    let content;
    const [settingsPage, setSettingsPage] = useState('llmProvider');

    switch (settingsPage) {
        case 'llmProvider': content = <LLMprovider userData={userData} />; break;
        case 'lstmSettings': content = <LSTMsettings userData={userData} sprints={sprints} />; break;
    }

    useEffect(() => {
        if (!localStorage.getItem("settings")) {
            startTour(
                [
                    {
                        target: "#settings-llm-nav",
                        content: (
                            <>
                                <strong>OpenAI API Key:</strong><br />
                                Add your API Key here to enable LLM features.
                                (More providers will be supported soon.)<br /><br />
                                <em>Hackathon Test Team Note:</em> An API Key is already provided for you,
                                so you do not need to enter anything unless you want to use your own key.
                            </>
                        ),
                    },
                    {
                        target: "#settings-lstm",
                        content: (
                            <>
                                <strong>LSTM Adapter:</strong><br />
                                Generate your LSTM Adapter here to enable sprint predictions.
                                This will analyze your past sprints and provide likely outcomes for future sprints.
                            </>
                        ),
                    }
                ]

                , "settings"
            );
        }
    }, []);

    const handleCheckboxChange = () => {
        if (!showWelcome == true) {
            localStorage.setItem("showWelcome", "true");
        } else {
            localStorage.setItem("showWelcome", "false");
        }
        setshowWelcome(!showWelcome);

    };

    return (
        <div className={styles.container}>
            <div className={styles.navBar}>
                <span
                    className={`${styles.menuItem} ${settingsPage === "llmProvider" ? styles.active : ""}`}
                    onClick={() => setSettingsPage("llmProvider")}
                    id="settings-llm-nav"
                >
                    LLM Provider
                </span>
                <span
                    className={`${styles.menuItem} ${styles.right} ${settingsPage === "lstmSettings" ? styles.active : ""}`}
                    onClick={() => setSettingsPage("lstmSettings")}
                    id="settings-lstm"
                >
                    LSTM Settings
                </span>
            </div>
            {content}
            <div className={styles.buttonsBottom}>
                <div className={styles.checkcont}>
                    <label htmlFor="checkbox">Show Welcome Message</label>
                    <input
                        type="checkbox"
                        id="checkbox"
                        name="welcomeMsg"
                        // Connect the input's 'checked' state to the React state
                        checked={showWelcome}
                        // Call the handler on change
                        onChange={handleCheckboxChange}
                    />
                </div>
                <div className={styles.restartTutCont} onClick={handleTutorialRestart}>
                    <span>Restart Tutorial Tour</span>
                </div>
            </div>
        </div>
    );
}

export default SettingsDisplay;