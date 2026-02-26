import React, { useEffect, useState } from 'react';
import { view, invoke, requestJira } from '@forge/bridge';
import styles from './App.module.css';
import SideBar from './Components/SideBar/SideBar';
import MainDisplay from './Components/MainDisplay/MainDisplay';
import Loader from './Components/Loader/Loader';
import TourProvider from './utils/TourProvider';

function App() {
    const [showWelcome, setshowWelcome] = useState(localStorage.getItem("showWelcome") ? (localStorage.getItem("showWelcome") == 'true' ? true : false) : (true));
    const [page, setPage] = useState(showWelcome == true ? "welcome" : "overview");

    const [allData, setAllData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [userData, setUserData] = useState({});

    const [predictedSimSprint, setPredictedSimSprint] = useState(null);
    const [llmSimResponse, setLlmSimResponse] = useState(null);

    const [authToken, setAuthToken] = useState('');

    const [tourResetKey, settourResKey] = useState(0);

    useEffect(() => {
        async function initUser() {
            setLoading(true);

            try {
                const ctx = await view.getContext();
                const accountId = ctx.accountId;

                // 1. Get Auth Token
                const authResponse = await invoke("getAuthToken", { accountId });
                const auth_token = authResponse.auth_token;

                if (!auth_token) throw new Error("Authentication token not received.");

                localStorage.setItem("auth_token", auth_token);
                setAuthToken(auth_token);

                // 2. Get User Info
                const userInfoResponse = await invoke("getUserInfos", { auth_token });

                // Check if the resolver returned a success flag or an empty object
                if (!userInfoResponse || userInfoResponse.success === false) {
                    // Log and continue, or throw an error if user data is essential
                    console.warn("Could not retrieve user data.");
                }
                setUserData(userInfoResponse.userData || {});

                // 3. Get All Boards and Sprints
                const data = await invoke("getAllBoardsAndSprints");
                console.log(data);
                setAllData(data);

            } catch (error) {
                console.error("Critical initialization failure:", error);

            } finally {
                setLoading(false);
            }
        }
        initUser();
    }, []);

    useEffect(() => {
        if (!loading) console.log(userData);
    }, [loading]);

    return (
        !loading ? (
            <>
                <TourProvider>
                    <div className={styles.container}>
                        <SideBar key={tourResetKey} setPage={setPage} page={page} settourResKey={settourResKey} />
                        <MainDisplay
                            page={page} sprints={allData} userData={userData}
                            setUserData={setUserData} setPage={setPage}
                            predictedSimSprint={predictedSimSprint}
                            setPredictedSimSprint={setPredictedSimSprint}
                            llmSimResponse={llmSimResponse}
                            setLlmSimResponse={setLlmSimResponse}
                            settourResKey={settourResKey} 
                            setshowWelcome={setshowWelcome}
                            showWelcome={showWelcome} />

                    </div>

                </TourProvider >
            </>
        ) : (<Loader />)
    );
}

export default App;