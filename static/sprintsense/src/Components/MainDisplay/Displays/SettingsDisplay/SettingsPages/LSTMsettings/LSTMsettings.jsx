import React, { useEffect, useState } from "react";
import { invoke } from "@forge/bridge";
import styles from "./LSTMsettings.module.css";
import AlertModal from "../../../../../AlertModal/AlertModal";

function LSTMsettings({ userData, sprints }) {
    const [loading, setLoading] = useState(true);
    const [training, setTraining] = useState(false);
    const [message, setMessage] = useState("");
    const [showalert, setshowAlert] = useState(false);
    const [alertHead, setAlertHead] = useState("");
    const [alertMsg, setAlertMsg] = useState("");
    const [userDataState, setUserDataState] = useState(userData);

    const handleTrainAdapter = async () => {
        if (!userDataState) return;

        const totalSprints = sprints?.reduce((sum, board) => sum + (board.sprints?.length || 0), 0);

        // uncomment in production - now using synthetic data if no sprints are available
        //if (totalSprints < 6) {
        //    setAlertHead("Error");
        //    setAlertMsg("You need at least 6 sprints to train an adapter!");
        //    setshowAlert(true);
        //    return;
        //}

        setTraining(true);
        setMessage("");

        try {
            const res = await invoke("trainAdapter", { auth_token: userDataState.auth_token, sprints });
            console.log("trainAdapter response:", res);

            if (res.success) {
                setMessage("Adapter trained successfully!");

                try {
                    const updatedUser = await invoke("getUserInfos", { auth_token: userDataState.auth_token });
                    console.log("getUserInfos response:", updatedUser);

                    if (updatedUser.success) {
                        setUserDataState(updatedUser.userData);
                    }
                } catch (e) {
                    console.error("Failed to fetch updated user info:", e);
                }

            } else {
                setMessage(res.message || "Training took a little longer and a timeout occured. Reload to see new adapter creation time.");
            }
        } catch (err) {
            console.error("Error in training adapter:", err);
            setMessage("Training took a little longer and a timeout occured. Reload to see new adapter creation time.");
        } finally {
            setTraining(false);
        }
    };

    return (
        <>
            {showalert && <AlertModal setshowAlert={setshowAlert} head={alertHead} message={alertMsg} />}
            <div className={styles.container}>
                <h2>LSTM Settings</h2>

                <div className={styles.userInfo}>
                    <p><strong>Adapter:</strong> {userDataState.adapter_path ? "Exists" : "Not created yet"}</p>
                    {userDataState.adapter_path && <p><strong>Adapter created at:</strong> {new Date(userDataState.adapter_created_at).toLocaleString()}</p>}
                </div>

                <button
                    onClick={handleTrainAdapter}
                    disabled={training}
                    className={styles.trainButton}
                >
                    {training ? "Training..." : "Train New Adapter"}
                </button>

                {message && <p className={styles.message}>{message}</p>}

                <div className={styles.lstmInfo}>
                    <h3>How the LSTM works:</h3>
                    <p>
                        The LSTM analyzes your team's last 6 sprints to predict the next sprint.
                        It fetches sprint data, organizes it chronologically, and fine-tunes a lightweight
                        adapter (dense layers) for your account. The base LSTM captures sequence patterns,
                        and your adapter personalizes predictions for your team's workflow.
                    </p>
                </div>
            </div>
        </>
    );
}

export default LSTMsettings;
